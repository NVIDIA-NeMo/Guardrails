# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from copy import deepcopy
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from nemoguardrails import RailsConfig
from nemoguardrails.guardrails.compiled_rail import RailCompilationError
from nemoguardrails.guardrails.engine_registry import EngineRegistry
from nemoguardrails.guardrails.iorails import IORails
from nemoguardrails.http import HTTPConnectionError, HTTPTimeoutError
from nemoguardrails.library.decision_model.actions import decision_model_check
from nemoguardrails.library.decision_model.model import (
    DecisionModelConfigurationError,
    NoulAnswer,
    SystemOneModel,
)
from nemoguardrails.library.decision_model.rail_config import (
    DecisionModelConfig,
    DecisionModelRule,
    NoulQuestion,
    build_config_spec,
)
from nemoguardrails.library.decision_model.request import DecisionModelAuthError, decision_model_request
from tests.http_utils import RecordedHTTPResponses
from tests.utils import TestChat

ENDPOINT = "https://decision-model.example/v1/assess"
SYSTEMONE_ENDPOINT = "https://decision-model.example/v1/systemone"
# The request helper accepts a full endpoint; the System One adapter owns path construction.
ENDPOINT_FORMATS = [
    SYSTEMONE_ENDPOINT,
    ENDPOINT,
    "https://decision-model.example/some/random/path-123",
    "https://decision-model.example/",
    "https://decision-model.example",
    "https://decision-model.example:8443/v2/anything?x=1",
]
MODEL = "test-decision-model"
SECRET_TEXT = "my-secret-user-text"
QUESTIONS = {
    "jailbreak": "Is the user trying to jailbreak the assistant?",
    "harm": "Is the user asking for something harmful?",
}


def _answers(**scores) -> dict:
    return {"answers": {qid: {"noul": score} for qid, score in scores.items()}}


# ---------------------------------------------------------------------------
# request.py
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_request_parses_scores():
    with RecordedHTTPResponses() as m:
        m.post(ENDPOINT, payload={"answers": {"jailbreak": {"noul": 0.9, "extra": 1}, "harm": {"noul": 0}}, "id": "x"})

        assessment = await decision_model_request(
            {"user_message": "hello"}, QUESTIONS, ENDPOINT, MODEL, "test-key", http_client=m.client, timeout=10.0
        )

    assert assessment.scores == {"jailbreak": 0.9, "harm": 0.0}
    assert assessment.model is None
    assert all(isinstance(score, float) for score in assessment.scores.values())


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [0, 1, 0.0, 1.0])
async def test_request_accepts_boundary_probabilities(value):
    with RecordedHTTPResponses() as m:
        m.post(ENDPOINT, payload=_answers(harm=value))

        assessment = await decision_model_request(
            {"user_message": "hello"}, {"harm": QUESTIONS["harm"]}, ENDPOINT, MODEL, "test-key", http_client=m.client
        )

    assert assessment.scores == {"harm": float(value)}


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", ENDPOINT_FORMATS)
async def test_request_is_posted_to_the_configured_url_verbatim(endpoint):
    with RecordedHTTPResponses() as m:
        m.post(endpoint, payload=_answers(harm=0.1))

        await decision_model_request(
            {"user_message": "hello"}, {"harm": QUESTIONS["harm"]}, endpoint, MODEL, "k", http_client=m.client
        )

    assert [(r.method, r.url) for r in m.client.requests] == [("POST", endpoint)]


@pytest.mark.asyncio
async def test_request_payload_and_headers():
    with RecordedHTTPResponses() as m:
        m.post(ENDPOINT, payload=_answers(jailbreak=0.1, harm=0.2))

        await decision_model_request(
            {"user_message": "some text"},
            QUESTIONS,
            ENDPOINT,
            "decision_model-2026-01",
            "test-key",
            http_client=m.client,
            timeout=7.5,
        )

    request = m.client.requests[0]
    assert request.method == "POST"
    assert request.url == ENDPOINT
    assert request.headers == {"Content-Type": "application/json", "Authorization": "Bearer test-key"}
    assert request.timeout == 7.5
    assert request.json == {
        "state": {"user_message": "some text"},
        "model": "decision_model-2026-01",
        "questions": {
            "jailbreak": {"type": "noul", "instructions": QUESTIONS["jailbreak"]},
            "harm": {"type": "noul", "instructions": QUESTIONS["harm"]},
        },
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [400, 401, 429, 500, 503])
async def test_request_non_200_raises_without_body_or_text(status):
    with RecordedHTTPResponses() as m:
        m.post(ENDPOINT, status=status, body=f"echoed input: {SECRET_TEXT}")

        with pytest.raises(ValueError) as exc_info:
            await decision_model_request(
                {"user_message": SECRET_TEXT}, QUESTIONS, ENDPOINT, MODEL, "test-key", http_client=m.client
            )

    message = str(exc_info.value)
    assert str(status) in message
    assert SECRET_TEXT not in message
    assert "echoed" not in message
    assert "test-key" not in message


@pytest.mark.asyncio
async def test_request_invalid_json_raises():
    with RecordedHTTPResponses() as m:
        m.post(ENDPOINT, body=f"not json {SECRET_TEXT}")

        with pytest.raises(ValueError) as exc_info:
            await decision_model_request(
                {"user_message": SECRET_TEXT}, QUESTIONS, ENDPOINT, MODEL, "test-key", http_client=m.client
            )

    assert "JSON" in str(exc_info.value)
    assert SECRET_TEXT not in str(exc_info.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("body", [{}, {"answers": None}, {"answers": []}, {"answers": "x"}, [], "text", None])
async def test_request_missing_answers_raises(body):
    with RecordedHTTPResponses() as m:
        m.post(ENDPOINT, body=json.dumps(body))

        with pytest.raises(ValueError, match="answers"):
            await decision_model_request(
                {"user_message": "hello"}, QUESTIONS, ENDPOINT, MODEL, "test-key", http_client=m.client
            )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "answers",
    [
        {"jailbreak": {"noul": 0.1}},  # question "harm" missing
        {"jailbreak": {"noul": 0.1}, "harm": 0.5},  # answer is not an object
        {"jailbreak": {"noul": 0.1}, "harm": {}},  # no noul value
        {"jailbreak": {"noul": 0.1}, "harm": {"noul": None}},
        {"jailbreak": {"noul": 0.1}, "harm": {"noul": True}},  # bool is not a probability
        {"jailbreak": {"noul": 0.1}, "harm": {"noul": False}},
        {"jailbreak": {"noul": 0.1}, "harm": {"noul": "0.5"}},
        {"jailbreak": {"noul": 0.1}, "harm": {"noul": -0.01}},
        {"jailbreak": {"noul": 0.1}, "harm": {"noul": 1.01}},
        {"jailbreak": {"noul": 0.1}, "harm": {"noul": float("nan")}},
        {"jailbreak": {"noul": 0.1}, "harm": {"yesno": 0.5}},  # some other answer type
    ],
)
async def test_request_bad_noul_answer_raises(answers):
    with RecordedHTTPResponses() as m:
        m.post(ENDPOINT, body=json.dumps({"answers": answers}))

        with pytest.raises(ValueError, match="harm"):
            await decision_model_request(
                {"user_message": "hello"}, QUESTIONS, ENDPOINT, MODEL, "test-key", http_client=m.client
            )


@pytest.mark.asyncio
async def test_request_transport_error_raises_without_details():
    with RecordedHTTPResponses() as m:
        m.post(ENDPOINT, exception=HTTPConnectionError(f"connection refused while sending {SECRET_TEXT}"))

        with pytest.raises(ValueError) as exc_info:
            await decision_model_request(
                {"user_message": SECRET_TEXT}, QUESTIONS, ENDPOINT, MODEL, "test-key", http_client=m.client
            )

    assert "failed" in str(exc_info.value)
    assert SECRET_TEXT not in str(exc_info.value)


@pytest.mark.asyncio
async def test_request_timeout_raises():
    with RecordedHTTPResponses() as m:
        m.post(ENDPOINT, exception=HTTPTimeoutError("timed out"))

        with pytest.raises(ValueError, match="timed out after 3.0 seconds"):
            await decision_model_request(
                {"user_message": "hello"}, QUESTIONS, ENDPOINT, MODEL, "test-key", http_client=m.client, timeout=3.0
            )


# ---------------------------------------------------------------------------
# actions.py
# ---------------------------------------------------------------------------


MODEL_CONFIG = {
    "type": "decider",
    "engine": "system_one",
    "model": MODEL,
    "api_key_env_var": "DECISION_MODEL_API_KEY",
    "parameters": {"base_url": "https://decision-model.example"},
}
RULES = {
    "jailbreak": {
        "question": {"type": "noul", "instructions": QUESTIONS["jailbreak"]},
        "block_if": {"probability_at_least": 0.5},
    },
    "harmful_request": {
        "question": {"type": "noul", "instructions": QUESTIONS["harm"]},
        "block_if": {"probability_at_least": 0.7},
    },
    "harmful_content": {
        "question": {"type": "noul", "instructions": "Does the assistant reply enable harm?"},
        "block_if": {"probability_at_least": 0.5},
    },
}


def _config(*, direction=None, fail_open=False, model=None, rules=None, exceptions=False):
    rails = {"config": {"decision_model": {"rules": deepcopy(rules or RULES), "fail_open": fail_open}}}
    if direction:
        names = ["jailbreak", "harmful_request"] if direction == "input" else ["harmful_content"]
        rails[direction] = {
            "flows": [f"decision model check {direction} $rule={name} $model=decider" for name in names]
        }
    return RailsConfig.from_content(
        config={
            "models": [deepcopy(model or MODEL_CONFIG)],
            "rails": rails,
            "enable_rails_exceptions": exceptions,
        }
    )


async def _check(config, client, *, source="input", text="hello", rule="jailbreak"):
    return await decision_model_check(
        source=source,
        text=text,
        rule=rule,
        model_name="decider",
        config=config,
        llms={"decider": SystemOneModel(config.models[0])},
        http_client=client,
    )


@pytest.fixture(autouse=True)
def _key(monkeypatch):
    monkeypatch.setenv("DECISION_MODEL_API_KEY", "test-key")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "source,key,rule", [("input", "user_message", "jailbreak"), ("output", "bot_message", "harmful_content")]
)
@pytest.mark.parametrize("score,blocked", [(0.49, False), (0.5, True), (1.0, True)])
async def test_named_rule_payload_and_inclusive_threshold(source, key, rule, score, blocked):
    config = _config()
    with RecordedHTTPResponses() as m:
        m.post(SYSTEMONE_ENDPOINT, payload={**_answers(**{rule: score}), "model": "resolved-version"})
        outcome = await _check(config, m.client, source=source, rule=rule)
    assert outcome.is_blocked is blocked
    assert outcome.metadata == {
        "source": source,
        "rule": rule,
        "scores": {rule: score},
        "thresholds": {rule: 0.5},
        "triggered": [rule] if blocked else [],
        "violations": [{"question": rule, "score": score, "threshold": 0.5}] if blocked else [],
        "model": "resolved-version",
    }
    assert m.client.requests[0].json == {
        "state": {key: "hello"},
        "model": MODEL,
        "questions": {rule: RULES[rule]["question"]},
    }
    assert m.client.requests[0].headers["Authorization"] == "Bearer test-key"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "threshold,score,blocked", [(0.7, 0.69, False), (0.7, 0.7, True), (0, 0, True), (1, 0.99, False), (1, 1, True)]
)
async def test_rule_threshold_boundaries(threshold, score, blocked):
    rules = deepcopy(RULES)
    rules["jailbreak"]["block_if"]["probability_at_least"] = threshold
    with RecordedHTTPResponses() as m:
        m.post(SYSTEMONE_ENDPOINT, payload=_answers(jailbreak=score))
        result = await _check(_config(rules=rules), m.client)
    assert result.is_blocked is blocked


@pytest.mark.asyncio
async def test_action_calls_decide_with_typed_question():
    model = AsyncMock()
    model.decide.return_value = NoulAnswer(0.1)
    config = _config()
    with RecordedHTTPResponses() as m:
        result = await decision_model_check(
            "input", "hello", "jailbreak", "decider", config, {"decider": model}, m.client
        )
    model.decide.assert_awaited_once_with(
        name="jailbreak",
        question=NoulQuestion(instructions=QUESTIONS["jailbreak"]),
        state={"user_message": "hello"},
        http_client=m.client,
    )
    assert not result.is_blocked
    assert "model" not in result.metadata


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["", "  ", None])
async def test_empty_text_skips_request_and_credentials(text, monkeypatch):
    config = _config()
    monkeypatch.delenv("DECISION_MODEL_API_KEY")
    with RecordedHTTPResponses() as m:
        result = await _check(config, m.client, text=text)
    assert not result.is_blocked
    assert not m.client.requests
    assert result.metadata["scores"] == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_open", [False, True])
@pytest.mark.parametrize("failure", [400, 429, 500, 503, "timeout", "connection", "json", "probability", "missing"])
async def test_provider_failures_and_privacy(fail_open, failure, caplog):
    with RecordedHTTPResponses() as m:
        if isinstance(failure, int):
            m.post(SYSTEMONE_ENDPOINT, status=failure, body=SECRET_TEXT)
        elif failure in {"timeout", "connection"}:
            cls = HTTPTimeoutError if failure == "timeout" else HTTPConnectionError
            m.post(SYSTEMONE_ENDPOINT, exception=cls(SECRET_TEXT))
        elif failure == "json":
            m.post(SYSTEMONE_ENDPOINT, body=SECRET_TEXT)
        else:
            m.post(SYSTEMONE_ENDPOINT, payload=_answers(jailbreak="invalid") if failure == "probability" else {})
        result = await _check(_config(fail_open=fail_open), m.client, text=SECRET_TEXT)
    assert result.is_blocked is not fail_open
    assert result.metadata["fail_open"] is fail_open
    assert result.metadata["error"]
    assert SECRET_TEXT not in caplog.text
    assert SECRET_TEXT not in str(result)


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_open", [False, True])
@pytest.mark.parametrize("status", [401, 403])
async def test_authentication_failure(fail_open, status, caplog):
    with RecordedHTTPResponses() as m:
        m.post(SYSTEMONE_ENDPOINT, status=status, body=SECRET_TEXT)
        if fail_open:
            result = await _check(_config(fail_open=True), m.client, text=SECRET_TEXT)
            assert not result.is_blocked
            assert result.metadata["fail_open"] is True
        else:
            with pytest.raises(DecisionModelAuthError):
                await _check(_config(), m.client, text=SECRET_TEXT)
    assert "authentication failed" in caplog.text
    assert SECRET_TEXT not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_open", [False, True])
async def test_missing_key_never_fails_open(fail_open, monkeypatch):
    config = _config(fail_open=fail_open)
    monkeypatch.delenv("DECISION_MODEL_API_KEY")
    with RecordedHTTPResponses() as m:
        with pytest.raises(DecisionModelConfigurationError, match="unset"):
            await _check(config, m.client)
    assert not m.client.requests


@pytest.mark.asyncio
async def test_custom_model_connection_and_optional_auth(monkeypatch):
    monkeypatch.delenv("DECISION_MODEL_API_KEY")
    model = deepcopy(MODEL_CONFIG)
    model["api_key_env_var"] = None
    model["model"] = "another-model"
    model["parameters"] = {"base_url": "http://internal.example/proxy/", "timeout": 2.5}
    with RecordedHTTPResponses() as m:
        m.post("http://internal.example/proxy/v1/systemone", payload=_answers(jailbreak=0.1))
        await _check(_config(model=model), m.client)
    request = m.client.requests[0]
    assert "Authorization" not in request.headers
    assert request.timeout == 2.5
    assert request.json["model"] == "another-model"


@pytest.mark.asyncio
async def test_custom_key_environment_variable(monkeypatch):
    monkeypatch.setenv("CUSTOM_DECIDER_KEY", "custom-test-key")
    model = deepcopy(MODEL_CONFIG)
    model["api_key_env_var"] = "CUSTOM_DECIDER_KEY"
    with RecordedHTTPResponses() as m:
        m.post(SYSTEMONE_ENDPOINT, payload=_answers(jailbreak=0.1))
        await _check(_config(model=model), m.client)
    assert m.client.requests[0].headers["Authorization"] == "Bearer custom-test-key"


@pytest.mark.asyncio
async def test_unverified_client_is_owned_but_injected_client_is_not(monkeypatch, caplog):
    from nemoguardrails.library.decision_model import request as request_module

    model = deepcopy(MODEL_CONFIG)
    model["parameters"]["verify_ssl"] = False
    created = []
    with RecordedHTTPResponses() as owned, RecordedHTTPResponses() as injected:
        owned.post(SYSTEMONE_ENDPOINT, payload=_answers(jailbreak=0.1))

        def create(**kwargs):
            created.append(kwargs)
            return owned.client

        monkeypatch.setattr(request_module, "create_http_client", create)
        await _check(_config(model=model), injected.client)
        assert not injected.client.requests
        assert owned.client.close_calls == 1
        assert injected.client.close_calls == 0
    assert created[0]["tls"].verify is False
    assert "TLS verification is disabled" in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "rule,model_name,source",
    [("unknown", "decider", "input"), ("jailbreak", "unknown", "input"), ("jailbreak", "decider", "retrieval")],
)
async def test_invalid_action_selection_raises(rule, model_name, source):
    with pytest.raises(ValueError):
        await decision_model_check(source, "hello", rule, model_name, _config(), {"decider": object()})


@pytest.mark.parametrize(
    "rules", [{}, {"x": {}}, {"x": {"question": {"instructions": "?"}}}, {" ": RULES["jailbreak"]}]
)
def test_rules_are_required_and_complete(rules):
    with pytest.raises(ValidationError):
        DecisionModelConfig(rules=rules)


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan"), float("inf")])
def test_config_and_schema_bound_threshold(threshold):
    data = deepcopy(RULES["jailbreak"])
    data["block_if"]["probability_at_least"] = threshold
    with pytest.raises(ValidationError):
        DecisionModelRule.model_validate(data)
    schema = DecisionModelRule.model_json_schema()["$defs"]["DecisionModelBlockCondition"]["properties"][
        "probability_at_least"
    ]
    assert schema["minimum"] == 0
    assert schema["maximum"] == 1


@pytest.mark.parametrize(
    "question",
    [
        {},
        {"instructions": ""},
        {"instructions": "  "},
        {"type": "choice", "instructions": "?"},
        {"instructions": "?", "extra": 1},
    ],
)
def test_only_nonblank_noul_questions_are_supported(question):
    with pytest.raises(ValidationError):
        NoulQuestion.model_validate(question)


@pytest.mark.parametrize(
    "parameters",
    [
        {},
        {"base_url": "ftp://example.org"},
        {"base_url": "https://"},
        {"base_url": "https://u:secret@example.org"},
        {"base_url": "https://example.org?key=secret"},
        {"base_url": "https://example.org#fragment"},
        {"base_url": "http://example.org"},
        {"base_url": "https://example.org:bad"},
        {"base_url": "https://example.org", "timeout": 0},
        {"base_url": "https://example.org", "unknown": 1},
    ],
)
def test_model_connection_validation(parameters):
    model = deepcopy(MODEL_CONFIG)
    model["parameters"] = parameters
    with pytest.raises(ValueError):
        SystemOneModel(_config(model=model).models[0])


def test_config_exports_and_no_implicit_configuration():
    assert set(build_config_spec().exports) == {
        "DecisionModelConfig",
        "DecisionModelRule",
        "DecisionModelBlockCondition",
        "NoulQuestion",
    }
    assert RailsConfig.from_content(config={"models": []}).rails.config.decision_model is None
    with pytest.raises(ValidationError):
        DecisionModelConfig.model_validate({"rules": RULES, "server_endpoint": SYSTEMONE_ENDPOINT})


@pytest.mark.parametrize("missing", ["rule", "model"])
def test_iorails_requires_rule_and_model_bindings(missing):
    config = _config()
    suffix = "$model=decider" if missing == "rule" else "$rule=jailbreak"
    config.rails.input.flows = [f"decision model check input {suffix}"]
    with pytest.raises(RailCompilationError):
        IORails(config)


def test_registry_builds_decision_adapter_without_chat_engine():
    registry = EngineRegistry(_config().models)
    assert isinstance(registry.llms["decider"], SystemOneModel)
    assert not registry._engines


@pytest.mark.parametrize("direction", ["input", "output"])
@pytest.mark.parametrize("blocked", [False, True])
def test_colang_1_rules_run_in_order_and_stop_at_first_block(direction, blocked):
    config = _config(direction=direction)
    chat = TestChat(config, llm_completions=["Safe reply"])
    with RecordedHTTPResponses() as m:
        chat.app.register_action_param("http_client", m.client)
        first = "jailbreak" if direction == "input" else "harmful_content"
        m.post(SYSTEMONE_ENDPOINT, payload=_answers(**{first: 0.9 if blocked else 0.1}))
        if direction == "input" and not blocked:
            m.post(SYSTEMONE_ENDPOINT, payload=_answers(harmful_request=0.1))
        chat >> "Hello"
        chat << ("I'm sorry, I can't respond to that." if blocked else "Safe reply")
    assert len(m.client.requests) == (2 if direction == "input" and not blocked else 1)
    assert list(m.client.requests[0].json["questions"]) == [first]
    assert m.client.requests[0].json["state"] == {
        "user_message" if direction == "input" else "bot_message": "Hello" if direction == "input" else "Safe reply"
    }


@pytest.mark.parametrize("direction", ["input", "output"])
@pytest.mark.parametrize("blocked", [False, True])
@pytest.mark.parametrize("exceptions", [False, True])
def test_colang_2_rules_run_in_order_and_stop_at_first_block(direction, blocked, exceptions):
    config_dict = _config(exceptions=exceptions).model_dump()
    config_dict["colang_version"] = "2.x"
    calls = (
        'decision model check input "jailbreak" "decider"\n  decision model check input "harmful_request" "decider"'
        if direction == "input"
        else 'decision model check output "harmful_content" "decider"'
    )
    config = RailsConfig.from_content(
        config=config_dict,
        colang_content=f"""
import core
import llm
import guardrails
import nemoguardrails.library.decision_model

flow {direction} rails $text
  {calls}

flow main
  activate llm continuation
  user said something
  bot say "Safe reply"
""",
    )
    chat = TestChat(config)
    with RecordedHTTPResponses() as m:
        chat.app.register_action_param("http_client", m.client)
        first = "jailbreak" if direction == "input" else "harmful_content"
        m.post(SYSTEMONE_ENDPOINT, payload=_answers(**{first: 0.9 if blocked else 0.1}))
        if direction == "input" and not blocked:
            m.post(SYSTEMONE_ENDPOINT, payload=_answers(harmful_request=0.1))
        chat >> "Hello"
        if blocked and exceptions:
            chat << {"type": "DecisionModelCheckRailException"}
        else:
            chat << ("I'm sorry, I can't respond to that." if blocked else "Safe reply")
    assert len(m.client.requests) == (2 if direction == "input" and not blocked else 1)
    assert m.client.requests[0].json["state"] == {
        "user_message" if direction == "input" else "bot_message": "Hello" if direction == "input" else "Safe reply"
    }


@pytest.mark.parametrize(
    "status,expected",
    [(503, "I'm sorry, I can't respond to that."), (401, "I'm sorry, an internal error has occurred.")],
)
def test_colang_provider_failure_is_fail_closed(status, expected, caplog):
    chat = TestChat(_config(direction="input"))
    with RecordedHTTPResponses() as m:
        chat.app.register_action_param("http_client", m.client)
        m.post(SYSTEMONE_ENDPOINT, status=status, body=SECRET_TEXT)
        chat >> SECRET_TEXT
        chat << expected
    assert SECRET_TEXT not in caplog.text
    assert len(m.client.requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("direction", ["input", "output"])
async def test_colang_1_block_emits_exception(direction):
    chat = TestChat(_config(direction=direction, exceptions=True), llm_completions=["Safe reply"])
    with RecordedHTTPResponses() as m:
        chat.app.register_action_param("http_client", m.client)
        rule = "jailbreak" if direction == "input" else "harmful_content"
        m.post(SYSTEMONE_ENDPOINT, payload=_answers(**{rule: 0.9}))
        result = await chat.app.generate_async(messages=[{"role": "user", "content": "Hello"}])
    assert result["role"] == "exception"
    assert result["content"]["type"] == "DecisionModelCheckRailException"
    assert f"triggered on {direction}" in result["content"]["message"]


@pytest.mark.asyncio
@pytest.mark.parametrize("block_first", [False, True])
async def test_example_config_public_check_stops_and_reports_reached_rule(block_first, monkeypatch):
    from nemoguardrails import Guardrails
    from nemoguardrails.rails.llm.options import RailStatus

    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    config = RailsConfig.from_path("examples/configs/decision_model")
    with RecordedHTTPResponses() as m:
        monkeypatch.setattr("nemoguardrails.guardrails.rails_manager.create_http_client", lambda **kwargs: m.client)
        m.post("https://api.typesafe.ai/v1/systemone", payload=_answers(jailbreak=0.9 if block_first else 0.1))
        if not block_first:
            m.post("https://api.typesafe.ai/v1/systemone", payload=_answers(harmful_request=0.7))
        async with Guardrails(config) as rails:
            assert isinstance(rails.rails_engine, IORails)
            result = await rails.check_async(messages=[{"role": "user", "content": "Hello"}])
    assert result.status == RailStatus.BLOCKED
    assert result.rail == "decision model check input"
    assert [list(r.json["questions"]) for r in m.client.requests] == (
        [["jailbreak"]] if block_first else [["jailbreak"], ["harmful_request"]]
    )


@pytest.mark.asyncio
async def test_model_selection_and_client_lifetime(monkeypatch):
    config = _config(direction="input")
    other = deepcopy(MODEL_CONFIG)
    other["type"] = "other_decider"
    other["model"] = "another-model"
    other["parameters"]["base_url"] = "https://other-decider.example"
    from nemoguardrails.rails.llm.config import Model

    config.models.append(Model.model_validate(other))
    config.rails.input.flows = ["decision model check input $rule=jailbreak $model=other_decider"]
    with RecordedHTTPResponses() as m:
        monkeypatch.setattr("nemoguardrails.guardrails.rails_manager.create_http_client", lambda **kwargs: m.client)
        m.post("https://other-decider.example/v1/systemone", payload=_answers(jailbreak=0.1))
        async with IORails(config) as rails:
            await rails.check_async(messages=[{"role": "user", "content": "Hello"}])
            assert m.client.close_calls == 0
        assert m.client.close_calls == 1
    assert m.client.requests[0].json["model"] == "another-model"


@pytest.mark.parametrize("model_type", ["main", "embeddings", "jailbreak_detection"])
def test_system_one_cannot_replace_generation_or_embedding_models(model_type):
    model = deepcopy(MODEL_CONFIG)
    model["type"] = model_type
    config = _config(model=model)
    with pytest.raises(ValueError, match="dedicated decision model type"):
        EngineRegistry(config.models)
    with pytest.raises(ValueError, match="dedicated decision model type"):
        TestChat(config)


@pytest.mark.parametrize("option", ["mode", "cache"])
def test_unsupported_system_one_modes_are_rejected(option):
    model = deepcopy(MODEL_CONFIG)
    model[option] = "text" if option == "mode" else {"enabled": True}
    with pytest.raises(ValueError, match="does not support"):
        SystemOneModel(_config(model=model).models[0])


def test_missing_key_rejected_at_config_load(monkeypatch):
    monkeypatch.delenv("DECISION_MODEL_API_KEY")
    with pytest.raises(ValidationError, match="environment variable"):
        _config()


@pytest.mark.asyncio
@pytest.mark.parametrize("model", [None, "", "   ", 42, {}, []])
async def test_invalid_optional_response_model_is_omitted(model):
    with RecordedHTTPResponses() as m:
        m.post(SYSTEMONE_ENDPOINT, payload={**_answers(jailbreak=0.1), "model": model})
        result = await _check(_config(), m.client)
    assert "model" not in result.metadata


def test_connection_validation_does_not_echo_configured_secrets():
    model = deepcopy(MODEL_CONFIG)
    model["parameters"]["api_key"] = "private-config-value"
    with pytest.raises(ValidationError) as error:
        SystemOneModel(_config(model=model).models[0])
    assert "private-config-value" not in str(error.value)
