# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from pathlib import Path

import pytest
import yaml

from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.decisions.model import DecisionError, decision_models
from nemoguardrails.decisions.types import ChoiceAnswer, ChoiceQuestion, NoulAnswer, NoulQuestion
from nemoguardrails.exceptions import StreamingNotSupportedError
from nemoguardrails.http import HTTPResponse
from nemoguardrails.library.decision_model.actions import decision_model_check_output
from nemoguardrails.streaming import StreamingHandler
from nemoguardrails.testing import RecordingHTTPClient
from nemoguardrails.testing.fake_model import FakeLLMModel
from tests.utils import TestChat

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/configs/decision_model/config.yml"

PASSTHROUGH_MAIN = """
import core

flow main
  user said something
  $reply = await PassthroughLLMAction(user_message="Refund?")
  bot say $reply
"""


def response(name, answer):
    return HTTPResponse(status_code=200, content=json.dumps({"answers": {name: answer}}).encode())


@pytest.fixture
def config(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-credential")
    return RailsConfig.from_path(str(EXAMPLE.parent))


@pytest.mark.asyncio
@pytest.mark.parametrize(("value", "blocked"), [(0.89, False), (0.90, True)])
async def test_noul_threshold(config, value, blocked):
    http = RecordingHTTPClient([response("password_request", {"type": "noul", "noul": value})])
    result = await decision_model_check_output(
        "password_request", "support_judge", config, decision_models(config.models), "reply", http_client=http
    )
    assert result.is_blocked is blocked
    assert result.metadata["rule"] == "password_request"
    (request,) = http.requests
    assert request.json["state"] == {"bot_message": "reply"}
    assert request.json["questions"]["password_request"]["type"] == "noul"
    assert request.url == "https://api.typesafe.ai/v1/systemone"
    assert request.timeout == 30
    assert http.close_calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("engine", "base_url", "endpoint", "option", "value"),
    [
        (
            "system_one",
            "http://localhost:8000/v1/",
            "http://localhost:8000/v1/systemone",
            "chat_template_kwargs",
            {"enable_thinking": False},
        ),
        (
            "openai_decisions",
            "https://proxy.example/openai",
            "https://proxy.example/openai/v1/decisions",
            "prompt_cache_options",
            {"ttl": "30m"},
        ),
    ],
)
async def test_decision_request_parameters(engine, base_url, endpoint, option, value, monkeypatch):
    monkeypatch.setenv("JUDGE_TOKEN", "test-token")
    config = RailsConfig.from_content(
        config={
            "models": [
                {
                    "type": "judge",
                    "engine": engine,
                    "model": "served-model",
                    "api_key_env_var": "JUDGE_TOKEN",
                    "parameters": {"base_url": base_url, "timeout": 2.5, option: value},
                }
            ]
        }
    )
    state = {"bot_message": "Hello."}
    if engine == "system_one":
        body = {"state": state, "questions": {"greeting": {"type": "noul", "instructions": "A greeting?"}}}
        answer = response("greeting", {"type": "noul", "noul": 0.9})
    else:
        body = {
            "input": [{"role": "user", "content": [{"type": "input_json", "data": state}]}],
            "questions": [{"type": "predicate", "name": "greeting", "instructions": "A greeting?"}],
        }
        answer = HTTPResponse(200, content=b'{"answers":[{"type":"predicate","name":"greeting","probability":0.9}]}')
    http = RecordingHTTPClient([answer])
    result = await decision_models(config.models)["judge"].decide(
        "greeting", NoulQuestion("A greeting?"), state, http_client=http
    )
    assert result == NoulAnswer(0.9)
    (request,) = http.requests
    assert (request.method, request.url, request.timeout) == ("POST", endpoint, 2.5)
    assert request.headers == {"Authorization": "Bearer test-token"}
    assert request.json == {"model": "served-model", option: value, **body}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("rule", "answer"),
    [
        ("password_request", {"type": "predicate", "probability": 0.95}),
        (
            "refund_promise",
            {
                "type": "choice",
                "choice": "promised",
                "probabilities": [
                    {"value": "promised", "probability": 0.95},
                    {"value": "conditional", "probability": 0.05},
                    {"value": "neither", "probability": 0.0},
                ],
            },
        ),
    ],
)
async def test_vllm_decisions_text_input(rule, answer, monkeypatch):
    data = yaml.safe_load(EXAMPLE.read_text())
    data["models"][0].update(
        engine="openai_decisions",
        model="served-judge",
        api_key_env_var="VLLM_TOKEN",
        parameters={"base_url": "http://localhost:8000/v1", "input_format": "text"},
    )
    monkeypatch.setenv("VLLM_TOKEN", "test-token")
    config = RailsConfig.from_content(config=data)
    reply = 'Your "refund" is guaranteed.\nCafé'
    response_body = {"answers": [{"name": rule, **answer}]}
    http = RecordingHTTPClient([HTTPResponse(200, content=json.dumps(response_body).encode())])
    outcome = await decision_model_check_output(
        rule, "support_judge", config, decision_models(config.models), reply, http_client=http
    )
    assert outcome.is_blocked
    (request,) = http.requests
    assert request.url == "http://localhost:8000/v1/decisions"
    assert request.json.keys() == {"model", "questions", "input"}
    assert json.loads(request.json["input"]) == {"bot_message": reply}
    (question,) = request.json["questions"]
    assert (question["name"], question["type"]) == (rule, answer["type"])


@pytest.mark.parametrize(
    ("engine", "parameters", "error"),
    [
        ("system_one", {"state": {}}, "rail-owned.*state"),
        ("openai_decisions", {"input": []}, "rail-owned.*input"),
        ("system_one", {"questions": {}}, "rail-owned.*questions"),
        ("openai_decisions", {"timeout": 0}, "timeout"),
        ("openai_decisions", {"input_format": "binary"}, "input_format"),
        ("system_one", {"api_key": "test-token"}, "api_key_env_var"),
        ("system_one", {"default_headers": {}}, "transport parameters"),
    ],
)
def test_invalid_decision_parameters_fail_at_config_load(engine, parameters, error):
    with pytest.raises(ValueError, match=error):
        RailsConfig.from_content(
            config={
                "models": [
                    {
                        "type": "judge",
                        "engine": engine,
                        "model": "served-model",
                        "parameters": parameters,
                    }
                ]
            }
        )


@pytest.mark.asyncio
async def test_system_one_choice_distribution(config):
    model = decision_models(config.models)["support_judge"]
    rule = config.rails.config.decision_model.rules["refund_promise"]
    probabilities = {"promised": 0.01, "conditional": 0.98, "neither": 0.01}
    http = RecordingHTTPClient(
        [response("refund_promise", {"type": "choice", "choice": "conditional", "probabilities": probabilities})]
    )
    answer = await model.decide(
        "refund_promise",
        ChoiceQuestion(rule.question.instructions, rule.question.choices),
        {"bot_message": "I can check eligibility."},
        http_client=http,
    )
    assert answer == ChoiceAnswer("conditional", probabilities)
    assert http.requests[0].json["questions"]["refund_promise"]["criteria"] == rule.question.choices


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "answer", [{"type": "refusal"}, {"type": "noul"}, {"type": "noul", "noul": True}, {"type": "noul", "noul": 1.01}]
)
async def test_unusable_answer(config, answer):
    model = decision_models(config.models)["support_judge"]
    question = NoulQuestion(config.rails.config.decision_model.rules["password_request"].question.instructions)
    with pytest.raises(DecisionError):
        await model.decide(
            "password_request",
            question,
            {"bot_message": "reply"},
            http_client=RecordingHTTPClient([response("password_request", answer)]),
        )


@pytest.mark.parametrize(
    ("params", "error"),
    [
        ("$rule=absent $model=support_judge", "Undefined decision rule"),
        ("$rule=refund_promise", "requires"),
        ("$check=refund_promise $model=support_judge", r"requires \$rule and \$model"),
        ("$model=support_judge $rule=refund_promise", None),
    ],
)
def test_binding_validation(params, error, monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-credential")
    data = yaml.safe_load(EXAMPLE.read_text())
    data["rails"]["output"]["flows"] = ["decision model check output " + params]
    if error:
        with pytest.raises(ValueError, match=error):
            RailsConfig.from_content(config=data)
    else:
        RailsConfig.from_content(config=data)


@pytest.mark.parametrize("v2", [False, True], ids=["colang1", "colang2"])
def test_colang_blocks_actual_reply(v2, monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-credential")
    data = yaml.safe_load(EXAMPLE.read_text())
    if v2:
        data["colang_version"] = "2.x"
        data["rails"].pop("output")
        flows = """
        import core
        import llm
        import guardrails
        import nemoguardrails.library.decision_model

        flow output rails $output_text
          decision model check output "refund_promise" "support_judge"

        flow main
          activate llm continuation
          user said something
          bot say "A full refund is guaranteed."
        """
    else:
        data["rails"]["output"]["flows"] = ["decision model check output $model=support_judge $rule=refund_promise"]
        flows = None
    chat = TestChat(
        RailsConfig.from_content(config=data, colang_content=flows), llm_completions=["A full refund is guaranteed."]
    )
    http = RecordingHTTPClient([response("refund_promise", {"type": "choice", "choice": "promised"})])
    chat.app.register_action_param("http_client", http)
    chat >> "Refund?"
    chat << "I'm sorry, I can't respond to that."
    assert http.requests[0].json["state"] == {"bot_message": "A full refund is guaranteed."}


@pytest.mark.parametrize(
    ("change", "error"),
    [
        ({"choice_in": ["issued"]}, "declared choices"),
        ({"probability_at_least": 0.9}, "question type"),
    ],
)
def test_reject_invalid_choice_policy(config, change, error):
    data = yaml.safe_load(EXAMPLE.read_text())
    data["rails"]["config"]["decision_model"]["rules"]["refund_promise"]["block_if"] = change
    with pytest.raises(ValueError, match=error):
        RailsConfig.from_content(config=data)


def test_reject_streaming_and_generation_models(config):
    data = yaml.safe_load(EXAMPLE.read_text())
    data["rails"]["output"]["streaming"] = {"enabled": True}
    with pytest.raises(ValueError, match="complete assistant reply"):
        RailsConfig.from_content(config=data)
    del data["rails"]["output"]["streaming"]
    data["models"][0]["engine"] = "openai"
    with pytest.raises(ValueError, match="native decision model"):
        RailsConfig.from_content(config=data)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "answer",
    [
        {"type": "refusal", "name": "refund_promise"},
        {"type": "choice", "name": "unrelated", "choice": "neither"},
        {"type": "choice", "name": "refund_promise", "choice": "unknown"},
        {"type": "choice", "name": "refund_promise", "choice": "neither", "probabilities": []},
    ],
)
async def test_openai_unusable_choice(config, answer):
    config.models[0].engine = "openai_decisions"
    deployment = decision_models(config.models)["support_judge"]
    spec = config.rails.config.decision_model.rules["refund_promise"].question
    question = ChoiceQuestion(spec.instructions, spec.choices)
    http = RecordingHTTPClient([HTTPResponse(200, content=json.dumps({"answers": [answer]}).encode())])
    with pytest.raises(DecisionError):
        await deployment.decide("refund_promise", question, {"bot_message": "reply"}, http_client=http)


def test_colang_provider_failure(config):
    chat = TestChat(config, llm_completions=["I can check eligibility."])
    http = RecordingHTTPClient([HTTPResponse(503)])
    chat.app.register_action_param("http_client", http)
    chat >> "Refund?"
    chat << "I'm sorry, an internal error has occurred."
    assert len(http.requests) == 1


@pytest.mark.asyncio
async def test_streaming_handler_cannot_release_unchecked_output(config):
    chat = TestChat(config, llm_completions=[])
    with pytest.raises(StreamingNotSupportedError, match="complete assistant reply"):
        await chat.app.generate_async(prompt="Refund?", streaming_handler=StreamingHandler())


@pytest.mark.asyncio
async def test_streaming_handler_with_output_rails_disabled(config):
    chat = TestChat(config, llm_completions=["Refund guaranteed."])
    http = RecordingHTTPClient([])
    chat.app.register_action_param("http_client", http)
    handler = StreamingHandler()

    await chat.app.generate_async(prompt="Refund?", options={"rails": {"output": False}}, streaming_handler=handler)

    assert [chunk async for chunk in handler] == ["Refund ", "guaranteed."]
    assert http.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize("reverse_params", [False, True])
async def test_colang2_example_runs_both_configured_checks(monkeypatch, tmp_path, reverse_params):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-credential")
    data = yaml.safe_load(EXAMPLE.read_text())
    data["colang_version"] = "2.x"
    refund_rule = "refund_promise"
    if reverse_params:
        # Names in YAML are data, even when they contain Colang interpolation syntax.
        refund_rule = "refund {len($name)}"
        rules = data["rails"]["config"]["decision_model"]["rules"]
        rules[refund_rule] = rules.pop("refund_promise")
        data["rails"]["output"]["flows"][1] = f'decision model check output $model=support_judge $rule="{refund_rule}"'
    main = tmp_path / "main.co"
    main.write_text(PASSTHROUGH_MAIN)
    if reverse_params:
        data["import_paths"] = [str(main)]
    config_file = tmp_path / "config.yml"
    config_file.write_text(yaml.safe_dump(data))
    config = RailsConfig.from_path(str(config_file if reverse_params else tmp_path))
    app = LLMRails(config, llm=FakeLLMModel(responses=["Refund guaranteed."]))
    http = RecordingHTTPClient(
        [
            response("password_request", {"type": "noul", "noul": 0.01}),
            response(refund_rule, {"type": "choice", "choice": "promised"}),
        ]
    )
    app.register_action_param("http_client", http)

    assert await app.generate_async(prompt="Refund?") == "I'm sorry, I can't respond to that."
    assert [list(request.json["questions"]) for request in http.requests] == [["password_request"], [refund_rule]]


@pytest.mark.asyncio
@pytest.mark.parametrize("options", [None, {"rails": {"output": False}}])
async def test_colang2_handwritten_check_prevents_passthrough_streaming(monkeypatch, options):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-credential")
    data = yaml.safe_load(EXAMPLE.read_text())
    data["colang_version"] = "2.x"
    del data["rails"]["output"]
    flows = """
import guardrails
import nemoguardrails.library.decision_model

flow output rails $output_text
  decision model check output "refund_promise" "support_judge"
"""
    config = RailsConfig.from_content(config=data, colang_content=PASSTHROUGH_MAIN + flows)
    model = FakeLLMModel(responses=["Refund guaranteed.", "Refund guaranteed."])
    app = LLMRails(config, llm=model)
    blocked = response("refund_promise", {"type": "choice", "choice": "promised"})
    http = RecordingHTTPClient([blocked, blocked])
    app.register_action_param("http_client", http)
    assert await app.generate_async(prompt="Refund?") == "I'm sorry, I can't respond to that."
    handler = StreamingHandler()
    with pytest.raises(StreamingNotSupportedError, match="complete assistant reply"):
        await app.generate_async(prompt="Refund?", streaming_handler=handler, options=options)
    assert handler.queue.empty()
    assert model.inference_count == len(http.requests) == 1

    async def external_generator():
        yield "Refund guaranteed."

    with pytest.raises(StreamingNotSupportedError, match="complete assistant reply"):
        app.stream_async(prompt="Refund?", generator=external_generator())
