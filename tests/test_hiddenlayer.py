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

"""Tests for the HiddenLayer rail's actions: auth, retries and the failure policy."""

import logging
from pathlib import Path
from uuid import UUID

import pytest
from pydantic import ValidationError

from nemoguardrails import Guardrails, RailsConfig
from nemoguardrails.actions.rail_outcome import RailDecision
from nemoguardrails.guardrails.tool_schema import Tool, ToolResult
from nemoguardrails.http import HTTPConnectionError, HTTPTimeoutError
from nemoguardrails.library.hiddenlayer import actions
from nemoguardrails.library.hiddenlayer._token import _reset_token_cache
from nemoguardrails.library.hiddenlayer._verdict import BLOCK_REASON, FAIL_OPEN_REASON
from nemoguardrails.library.hiddenlayer.actions import (
    NOTHING_TO_CHECK_REASON,
    _evaluate,
    _input_check,
    _settings,
    hiddenlayer_evaluate,
    hiddenlayer_evaluate_tool_call,
    hiddenlayer_evaluate_tool_result,
)
from nemoguardrails.library.hiddenlayer.errs import HiddenLayerCheckError
from nemoguardrails.library.hiddenlayer.rail_config import HiddenLayerRailConfig
from nemoguardrails.library.hiddenlayer.request import (
    _REQUEST_PATH,
    _RESPONSE_PATH,
)
from nemoguardrails.testing import RecordingHTTPClient
from nemoguardrails.types import ToolCall, ToolCallFunction
from tests.fixtures.hiddenlayer import (
    BOT_TEXT,
    INPUT_BLOCK,
    INPUT_MONITORED,
    INPUT_NONE,
    INPUT_REDACT,
    OUTPUT_BLOCK,
    OUTPUT_NONE,
    OUTPUT_REDACT,
    PII_TEXT,
    REDACTED_BOT_TEXT,
    REDACTED_PII_TEXT,
    TOOL_PROMOTED,
    headers,
    input_body,
)
from tests.http_utils import RecordedHTTPResponses
from tests.utils import TestChat

API_URL = "https://api.us.hiddenlayer.ai"
TOKEN_URL = f"{API_URL}/oauth2/token"
REQUEST_URL = f"{API_URL}{_REQUEST_PATH}"
RESPONSE_URL = f"{API_URL}{_RESPONSE_PATH}"
CLIENT_ID = "hl-client-id"
CLIENT_SECRET = "hl-client-secret-value"
TOKEN = "hl-access-token-1"
INPUT_TEXT = PII_TEXT
ERROR_BODY = '{"detail": "cannot parse interaction: My SSN is 536-90-4399"}'

_ENV_VARS = (
    "HIDDENLAYER_CLIENT_ID",
    "HIDDENLAYER_CLIENT_SECRET",
    "HIDDENLAYER_BASE_URL",
    "HIDDENLAYER_PROJECT_ID",
    "HIDDENLAYER_REQUESTER_ID",
)

RUN_SQL_TOOL = Tool(
    name="run_sql",
    arguments_schema={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
)


class _ForbiddenHTTPClient:
    """Stands in for every default transport so a test that forgets to inject a client fails offline."""

    async def request(self, method, url, **kwargs):
        raise AssertionError(f"unexpected network request: {method} {url}")

    async def close(self) -> None:
        pass


@pytest.fixture(autouse=True)
def hiddenlayer_env(monkeypatch):
    _reset_token_cache()
    for name in _ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HIDDENLAYER_CLIENT_ID", CLIENT_ID)
    monkeypatch.setenv("HIDDENLAYER_CLIENT_SECRET", CLIENT_SECRET)
    monkeypatch.setattr(actions, "create_http_client", lambda *args, **kwargs: _ForbiddenHTTPClient())
    monkeypatch.setattr(
        "nemoguardrails.guardrails.rails_manager.create_http_client",
        lambda *args, **kwargs: _ForbiddenHTTPClient(),
    )
    yield
    _reset_token_cache()


@pytest.fixture(params=[False, True], ids=["fail_closed", "fail_open"])
def fail_open(request) -> bool:
    return request.param


def _token(responses: RecordedHTTPResponses, token: str = TOKEN, times: int = 1) -> None:
    responses.post(TOKEN_URL, payload={"access_token": token, "expires_in": 3600}, times=times)


def _verdict(responses: RecordedHTTPResponses, fixture, *, url: str = REQUEST_URL, times: int = 1) -> None:
    """Answer the next evaluation with one of the fixtures: its status, verdict headers and payload."""
    status, response_headers, body = fixture
    responses.post(url, payload=body, headers=response_headers, status=status, times=times)


def _settings_for(**fields):
    return _settings(HiddenLayerRailConfig(retry_backoff_seconds=0, **fields))


async def _check_input(client, fail_open: bool = False, **fields):
    settings = _settings_for(fail_open=fail_open, **fields)
    body = _input_check("m", INPUT_TEXT)
    return await _evaluate("input", "openai", body, INPUT_TEXT, settings, client)


def _assert_failure_policy(outcome_or_error, fail_open: bool) -> None:
    if fail_open:
        assert outcome_or_error.decision is RailDecision.ALLOW
        assert outcome_or_error.metadata == {"fail_open": True}
        assert outcome_or_error.reason == FAIL_OPEN_REASON
    else:
        assert isinstance(outcome_or_error, HiddenLayerCheckError)


async def _outcome_or_error(client, fail_open: bool, **fields):
    try:
        return await _check_input(client, fail_open=fail_open, **fields)
    except HiddenLayerCheckError as error:
        return error


def _bearers(client: RecordingHTTPClient) -> list:
    return [request.headers["Authorization"] for request in client.requests if request.url != TOKEN_URL]


class TestAuth:
    @pytest.mark.asyncio
    async def test_token_reused_across_evaluations(self):
        with RecordedHTTPResponses() as responses:
            _token(responses)
            _verdict(responses, INPUT_NONE, times=2)
            for _ in range(2):
                assert (await _check_input(responses.client)).decision is RailDecision.ALLOW

    @pytest.mark.asyncio
    async def test_401_refreshes_the_token_and_retries_once(self):
        with RecordedHTTPResponses() as responses:
            _token(responses, "token-1")
            responses.post(REQUEST_URL, status=401, body="")
            _token(responses, "token-2")
            _verdict(responses, INPUT_NONE)

            assert (await _check_input(responses.client)).decision is RailDecision.ALLOW

        assert _bearers(responses.client) == ["Bearer token-1", "Bearer token-2"]

    @pytest.mark.asyncio
    async def test_401_twice_applies_the_failure_policy(self, fail_open):
        with RecordedHTTPResponses() as responses:
            _token(responses, "token-1")
            responses.post(REQUEST_URL, status=401, body="")
            _token(responses, "token-2")
            responses.post(REQUEST_URL, status=401, body="")

            _assert_failure_policy(await _outcome_or_error(responses.client, fail_open), fail_open)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("missing", ["HIDDENLAYER_CLIENT_ID", "HIDDENLAYER_CLIENT_SECRET"])
    async def test_missing_credentials_raise_value_error_even_with_fail_open(self, monkeypatch, missing, fail_open):
        monkeypatch.delenv(missing)
        with RecordedHTTPResponses() as responses:
            with pytest.raises(ValueError, match="HIDDENLAYER_CLIENT_ID and HIDDENLAYER_CLIENT_SECRET"):
                await _check_input(responses.client, fail_open=fail_open)

    @pytest.mark.asyncio
    async def test_insecure_base_url_from_env_raises_value_error_even_with_fail_open(self, monkeypatch):
        monkeypatch.setenv("HIDDENLAYER_BASE_URL", "http://evil.example")
        with RecordedHTTPResponses() as responses:
            with pytest.raises(ValueError, match="https://"):
                await _check_input(responses.client, fail_open=True)

    @pytest.mark.asyncio
    async def test_loopback_http_base_url_is_used(self, monkeypatch):
        monkeypatch.setenv("HIDDENLAYER_BASE_URL", "http://127.0.0.1:9")
        with RecordedHTTPResponses() as responses:
            responses.post("http://127.0.0.1:9/oauth2/token", payload={"access_token": TOKEN})
            _verdict(responses, INPUT_NONE, url=f"http://127.0.0.1:9{_REQUEST_PATH}")
            assert (await _check_input(responses.client)).decision is RailDecision.ALLOW


class TestTokenPath:
    """The token request uses the same retrying client and the same single connection-error retry."""

    @pytest.mark.asyncio
    async def test_connection_error_is_a_single_attempt(self, fail_open):
        with RecordedHTTPResponses() as responses:
            responses.post(TOKEN_URL, exception=HTTPConnectionError("refused"))
            _assert_failure_policy(await _outcome_or_error(responses.client, fail_open), fail_open)
            assert len(responses.client.requests) == 1

    @pytest.mark.asyncio
    async def test_503_then_success(self):
        with RecordedHTTPResponses() as responses:
            responses.post(TOKEN_URL, status=503, body="")
            _token(responses)
            _verdict(responses, INPUT_NONE)
            assert (await _check_input(responses.client)).decision is RailDecision.ALLOW

    @pytest.mark.asyncio
    async def test_timeout_is_a_single_attempt(self, fail_open):
        with RecordedHTTPResponses() as responses:
            responses.post(TOKEN_URL, exception=HTTPTimeoutError("timed out"))
            _assert_failure_policy(await _outcome_or_error(responses.client, fail_open), fail_open)

    @pytest.mark.asyncio
    async def test_401_applies_the_failure_policy_without_a_refresh_loop(self, fail_open):
        with RecordedHTTPResponses() as responses:
            responses.post(TOKEN_URL, status=401, body='{"error": "invalid_client"}')
            _assert_failure_policy(await _outcome_or_error(responses.client, fail_open), fail_open)


class TestFailures:
    @pytest.mark.asyncio
    async def test_connection_error_is_not_retried(self, fail_open):
        with RecordedHTTPResponses() as responses:
            _token(responses)
            responses.post(REQUEST_URL, exception=HTTPConnectionError("reset"))
            _assert_failure_policy(await _outcome_or_error(responses.client, fail_open), fail_open)
            assert len(_evaluations(responses.client)) == 1

    @pytest.mark.asyncio
    async def test_timeout_is_not_retried(self, fail_open):
        with RecordedHTTPResponses() as responses:
            _token(responses)
            responses.post(REQUEST_URL, exception=HTTPTimeoutError("timed out"))
            _assert_failure_policy(await _outcome_or_error(responses.client, fail_open), fail_open)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", [429, 502, 503, 504])
    async def test_retryable_status_then_success(self, status):
        with RecordedHTTPResponses() as responses:
            _token(responses)
            responses.post(REQUEST_URL, status=status, body="")
            _verdict(responses, INPUT_NONE)
            assert (await _check_input(responses.client)).decision is RailDecision.ALLOW

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", [429, 502, 503, 504])
    async def test_retryable_status_exhausts_max_retries(self, status, fail_open):
        with RecordedHTTPResponses() as responses:
            _token(responses)
            responses.post(REQUEST_URL, status=status, body="", times=3)
            _assert_failure_policy(await _outcome_or_error(responses.client, fail_open), fail_open)

    @pytest.mark.asyncio
    async def test_max_retries_zero_is_a_single_attempt(self):
        with RecordedHTTPResponses() as responses:
            _token(responses)
            responses.post(REQUEST_URL, status=503, body="")
            with pytest.raises(HiddenLayerCheckError, match="status=503"):
                await _check_input(responses.client, max_retries=0)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", [400, 403, 413, 422, 500])
    async def test_non_retryable_status_is_a_single_call(self, status, fail_open):
        with RecordedHTTPResponses() as responses:
            _token(responses)
            responses.post(REQUEST_URL, status=status, body=ERROR_BODY, content_type="application/json")
            _assert_failure_policy(await _outcome_or_error(responses.client, fail_open), fail_open)

    @pytest.mark.asyncio
    async def test_undecodable_body_applies_the_failure_policy(self, fail_open):
        with RecordedHTTPResponses() as responses:
            _token(responses)
            responses.post(REQUEST_URL, body="<html>bad gateway</html>", headers=headers("NONE"))
            _assert_failure_policy(await _outcome_or_error(responses.client, fail_open), fail_open)

    @pytest.mark.asyncio
    async def test_a_200_without_an_action_applies_the_failure_policy(self, fail_open):
        """No action is no verdict, so it joins the transport failures rather than being applied."""
        with RecordedHTTPResponses() as responses:
            _token(responses)
            responses.post(REQUEST_URL, payload=input_body(INPUT_TEXT), headers={})
            _assert_failure_policy(await _outcome_or_error(responses.client, fail_open), fail_open)

    @pytest.mark.asyncio
    async def test_an_action_the_rail_cannot_apply_blocks_whatever_the_failure_policy(self, fail_open):
        """The server decided something restrictive the rail does not know; allowing it would undo that."""
        with RecordedHTTPResponses() as responses:
            _token(responses)
            responses.post(REQUEST_URL, payload=input_body(INPUT_TEXT), headers=headers("QUARANTINE"))
            outcome = await _outcome_or_error(responses.client, fail_open)
            assert outcome.decision is RailDecision.BLOCK

    @pytest.mark.asyncio
    async def test_request_timeout_is_thirty_seconds(self):
        with RecordedHTTPResponses() as responses:
            _token(responses)
            _verdict(responses, INPUT_NONE)
            await _check_input(responses.client)
        assert [request.timeout for request in responses.client.requests] == [30.0, 30.0]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", [400, 422, 500])
    async def test_error_bodies_never_reach_logs_or_exceptions(self, caplog, status, fail_open):
        caplog.set_level(logging.DEBUG)
        with RecordedHTTPResponses() as responses:
            _token(responses)
            responses.post(REQUEST_URL, status=status, body=ERROR_BODY, content_type="application/json")
            result = await _outcome_or_error(responses.client, fail_open)

        rendered = caplog.text + str(result)
        for leaked in ("536-90-4399", "cannot parse", "detail", CLIENT_SECRET, TOKEN):
            assert leaked not in rendered
        if not fail_open:
            assert str(result) == (
                f"HiddenLayer input check failed: HiddenLayer evaluation failed: "
                f"status={status} content_type=application/json body_length={len(ERROR_BODY)}"
            )

    @pytest.mark.asyncio
    async def test_fail_open_logs_one_warning_without_content(self, caplog):
        caplog.set_level(logging.WARNING)
        with RecordedHTTPResponses() as responses:
            _token(responses)
            responses.post(REQUEST_URL, status=422, body=ERROR_BODY)
            await _check_input(responses.client, fail_open=True)

        (record,) = [record for record in caplog.records if record.name == actions.__name__]
        assert record.levelno == logging.WARNING
        assert record.getMessage() == (
            "HiddenLayer input check could not be completed (HiddenLayerTransportError, status=422); "
            "fail_open is enabled, allowing content."
        )


class TestVerdictsAreNeverFailedOpen:
    """A 200 carries HiddenLayer's verdict, including for an evaluation HiddenLayer could not run."""

    @pytest.mark.asyncio
    async def test_a_fail_closed_evaluation_failure_blocks(self, fail_open):
        with RecordedHTTPResponses() as responses:
            _token(responses)
            _verdict(responses, INPUT_BLOCK)
            outcome = await _check_input(responses.client, fail_open=fail_open)

        assert outcome.is_blocked
        assert "fail_open" not in outcome.metadata

    @pytest.mark.asyncio
    async def test_a_fail_open_evaluation_failure_allows_as_a_verdict(self, fail_open):
        with RecordedHTTPResponses() as responses:
            _token(responses)
            _verdict(responses, INPUT_NONE)
            outcome = await _check_input(responses.client, fail_open=fail_open)

        assert outcome.decision is RailDecision.ALLOW
        assert outcome.metadata == {"action": "NONE", "protection_mode": "AS_CONFIGURED"}

    @pytest.mark.asyncio
    async def test_monitor_mode_allows_without_a_fail_open_marker(self, fail_open):
        with RecordedHTTPResponses() as responses:
            _token(responses)
            _verdict(responses, INPUT_MONITORED)
            outcome = await _check_input(responses.client, fail_open=fail_open)

        assert outcome.decision is RailDecision.ALLOW
        assert outcome.metadata == {"action": "DETECT", "protection_mode": "MONITOR_MODE"}

    @pytest.mark.asyncio
    async def test_an_unusable_redaction_blocks_whatever_fail_open_says(self, fail_open):
        """The action is valid, so it is a verdict; the rail cannot apply it, so it fails closed."""
        with RecordedHTTPResponses() as responses:
            _token(responses)
            responses.post(REQUEST_URL, payload=input_body(INPUT_TEXT), headers=headers("REDACT"))
            outcome = await _check_input(responses.client, fail_open=fail_open)

        assert outcome.is_blocked
        assert outcome.reason == BLOCK_REASON
        assert "fail_open" not in outcome.metadata


class TestEvaluationId:
    @pytest.mark.asyncio
    async def test_each_check_sends_its_own_uuid(self):
        with RecordedHTTPResponses() as responses:
            _token(responses)
            _verdict(responses, INPUT_NONE, times=2)
            for _ in range(2):
                await _check_input(responses.client)

        first, second = [request.headers["HL-Roundtrip-Id"] for request in _evaluations(responses.client)]
        assert first != second
        assert str(UUID(first)) == first and str(UUID(second)) == second

    @pytest.mark.asyncio
    async def test_a_token_refresh_keeps_the_evaluation_id(self):
        with RecordedHTTPResponses() as responses:
            _token(responses, "token-1")
            responses.post(REQUEST_URL, status=401, body="")
            _token(responses, "token-2")
            _verdict(responses, INPUT_NONE)
            await _check_input(responses.client)

        assert _bearers(responses.client) == ["Bearer token-1", "Bearer token-2"]
        sent = [request.headers["HL-Roundtrip-Id"] for request in _evaluations(responses.client)]
        assert len(sent) == 2 and len(set(sent)) == 1

    @pytest.mark.asyncio
    async def test_a_retried_evaluation_keeps_its_id(self):
        with RecordedHTTPResponses() as responses:
            _token(responses)
            responses.post(REQUEST_URL, status=503, body="")
            _verdict(responses, INPUT_NONE)
            await _check_input(responses.client)

        sent = [request.headers["HL-Roundtrip-Id"] for request in _evaluations(responses.client)]
        assert len(sent) == 2 and len(set(sent)) == 1


class TestClientOwnership:
    @pytest.mark.asyncio
    async def test_injected_client_is_never_closed(self):
        # A failing check, so the close would happen on the error path if it happened at all.
        with RecordedHTTPResponses() as responses:
            responses.post(TOKEN_URL, status=503, body="", times=3)
            with pytest.raises(HiddenLayerCheckError):
                await _check_input(responses.client)
        assert responses.client.close_calls == 0

    @pytest.mark.asyncio
    @pytest.mark.parametrize("succeeds", [True, False])
    async def test_every_owned_client_is_closed(self, monkeypatch, succeeds):
        responses = RecordedHTTPResponses()
        _token(responses)
        if succeeds:
            _verdict(responses, INPUT_NONE)
        else:
            responses.post(REQUEST_URL, status=503, body="", times=3)
        created = []

        def create_http_client(**kwargs):
            created.append(kwargs)
            return responses.client

        monkeypatch.setattr(actions, "create_http_client", create_http_client)

        with responses:
            if succeeds:
                await _check_input(None)
            else:
                with pytest.raises(HiddenLayerCheckError):
                    await _check_input(None)

        # http_call builds a client per request and closes each one, so the count follows the
        # number of requests rather than being one shared client for the whole check.
        assert created and all(kwargs == {"timeout": 30.0} for kwargs in created)
        assert responses.client.close_calls == len(created)


class TestActions:
    @pytest.mark.asyncio
    async def test_input_posts_the_user_message_and_applies_a_redaction(self, monkeypatch):
        monkeypatch.setenv("HIDDENLAYER_PROJECT_ID", "my-project")
        with RecordedHTTPResponses() as responses:
            _token(responses)
            _verdict(responses, INPUT_REDACT)
            outcome = await hiddenlayer_evaluate(surface="input", text=INPUT_TEXT, http_client=responses.client)

        assert outcome.transform_text == {"user_message": REDACTED_PII_TEXT}
        evaluation = _evaluations(responses.client)[0]
        assert evaluation.url == REQUEST_URL
        assert evaluation.json == {"check": "input", "user_message": INPUT_TEXT}
        assert evaluation.headers["HL-Project-Id"] == "my-project"
        assert evaluation.headers["HL-Requester-Id"] == "hiddenlayer-nemo"
        assert "HL-Provider-Id" not in evaluation.headers
        assert evaluation.headers["Authorization"] == f"Bearer {TOKEN}"

    @pytest.mark.asyncio
    async def test_output_block(self):
        with RecordedHTTPResponses() as responses:
            _token(responses)
            _verdict(responses, OUTPUT_BLOCK, url=RESPONSE_URL)
            outcome = await hiddenlayer_evaluate(surface="output", text="hi", http_client=responses.client)

        assert outcome.is_blocked
        evaluation = _evaluations(responses.client)[0]
        assert evaluation.url == RESPONSE_URL
        assert evaluation.json == {"check": "output", "bot_message": "hi"}

    @pytest.mark.asyncio
    @pytest.mark.parametrize("surface", ["input", "output"])
    async def test_an_empty_message_is_allowed_without_calling_hiddenlayer(self, surface):
        """A check carrying no text is rejected, and there is nothing in it to evaluate."""
        client = RecordingHTTPClient()

        outcome = await hiddenlayer_evaluate(surface=surface, text="", http_client=client)

        assert outcome.decision is RailDecision.ALLOW
        assert outcome.reason == NOTHING_TO_CHECK_REASON
        assert "fail_open" not in outcome.metadata
        assert client.requests == []

    @pytest.mark.asyncio
    async def test_unknown_surface_raises(self):
        with pytest.raises(ValueError, match="surface"):
            await hiddenlayer_evaluate(surface="tool_call", text="hi", http_client=RecordingHTTPClient())

    @pytest.mark.asyncio
    async def test_insecure_base_url_raises_before_any_request(self, monkeypatch):
        monkeypatch.setenv("HIDDENLAYER_BASE_URL", "http://evil.example")
        client = RecordingHTTPClient()
        with pytest.raises(ValueError, match="https://"):
            await hiddenlayer_evaluate(surface="input", text="hi", http_client=client)
        assert client.requests == []

    @pytest.mark.asyncio
    async def test_undeclared_tool_call_is_blocked_without_calling_hiddenlayer(self):
        client = RecordingHTTPClient()
        call = ToolCall(id="call_1", function=ToolCallFunction(name="run_sql", arguments={"query": "SELECT 1"}))

        outcome = await hiddenlayer_evaluate_tool_call(tool_call=call, tool_definition=None, http_client=client)

        assert outcome.is_blocked
        assert client.requests == []

    @pytest.mark.asyncio
    async def test_tool_call_is_evaluated(self):
        call = ToolCall(id="call_1", function=ToolCallFunction(name="run_sql", arguments={"query": "SELECT 1"}))
        with RecordedHTTPResponses() as responses:
            _token(responses)
            _verdict(responses, OUTPUT_NONE, url=RESPONSE_URL)
            outcome = await hiddenlayer_evaluate_tool_call(
                tool_call=call, tool_definition=RUN_SQL_TOOL, http_client=responses.client
            )

        assert outcome.decision is RailDecision.ALLOW
        assert _evaluations(responses.client)[0].url == RESPONSE_URL
        assert _evaluation_bodies(responses.client)[0] == {
            "check": "tool_call",
            "tool_call": {"id": "call_1", "name": "run_sql", "arguments": {"query": "SELECT 1"}},
            "tool_definition": {"name": "run_sql", "parameters": RUN_SQL_TOOL.arguments_schema},
        }

    @pytest.mark.asyncio
    async def test_tool_result_takes_the_request_route_and_a_promoted_block(self):
        call = ToolCall(id="", function=ToolCallFunction(name="lookup", arguments={"q": "jane"}))
        result = ToolResult(call_id="call_probe2", name="lookup", content="SSN 536-90-4399")
        with RecordedHTTPResponses() as responses:
            _token(responses)
            _verdict(responses, TOOL_PROMOTED)
            outcome = await hiddenlayer_evaluate_tool_result(
                tool_call=call, tool_result=result, tool_definition=RUN_SQL_TOOL, http_client=responses.client
            )

        assert outcome.is_blocked and not outcome.is_transform
        assert outcome.metadata["protection_mode"] == "INTEGRATION_LIMITED_PROMOTE"
        evaluation = _evaluations(responses.client)[0]
        assert evaluation.url == REQUEST_URL
        assert evaluation.json["tool_definition"]["name"] == "run_sql"
        # The empty call ID is sent as NeMo held it; HiddenLayer owns the fallback to the result's.
        assert evaluation.json["tool_call"] == {"id": "", "name": "lookup", "arguments": {"q": "jane"}}
        assert evaluation.json["tool_result"] == {
            "content": "SSN 536-90-4399",
            "is_error": False,
            "call_id": "call_probe2",
            "name": "lookup",
        }


REFUSAL = "I'm sorry, I can't respond to that."
INPUT_REDACTED = REDACTED_PII_TEXT
OUTPUT_TEXT = BOT_TEXT
OUTPUT_REDACTED = REDACTED_BOT_TEXT
EXAMPLE_CONFIG = Path(__file__).resolve().parents[1] / "examples" / "configs" / "hiddenlayer"

V1_CONFIG = """
rails:
  config:
    hiddenlayer:
      retry_backoff_seconds: 0
  input:
    flows:
      - hiddenlayer check input
  output:
    flows:
      - hiddenlayer check output
"""

V2_CONFIG = """
colang_version: 2.x
models: []
rails:
  config:
    hiddenlayer:
      retry_backoff_seconds: 0
"""

V2_FLOWS = """
import core
import llm
import guardrails
import nemoguardrails.library.hiddenlayer

flow input rails $input_text
  hiddenlayer check input

flow output rails $output_text
  hiddenlayer check output

flow main
  activate llm continuation
  user said something
  global $user_message
  bot say $user_message
"""


def _evaluations(client: RecordingHTTPClient) -> list:
    """Every evaluation request the rail made, in order; the token requests are left out."""
    return [request for request in client.requests if request.url != TOKEN_URL]


def _evaluation_bodies(client: RecordingHTTPClient) -> list:
    return [request.json for request in _evaluations(client)]


def _v1_chat(llm_completions, enable_rails_exceptions: bool = False) -> TestChat:
    config = RailsConfig.from_content(
        yaml_content=V1_CONFIG + ("enable_rails_exceptions: true\n" if enable_rails_exceptions else "")
    )
    return TestChat(config, llm_completions=llm_completions)


class TestColang1:
    def test_input_block_refuses_without_calling_the_main_model(self):
        chat = _v1_chat(llm_completions=[])
        with RecordedHTTPResponses() as responses:
            _token(responses)
            _verdict(responses, INPUT_BLOCK)
            chat.app.register_action_param("http_client", responses.client)
            chat >> INPUT_TEXT
            chat << REFUSAL

        assert _evaluation_bodies(responses.client)[0]["user_message"] == INPUT_TEXT

    def test_input_block_raises_the_rail_exception_when_enabled(self):
        chat = _v1_chat(llm_completions=[], enable_rails_exceptions=True)
        with RecordedHTTPResponses() as responses:
            _token(responses)
            _verdict(responses, INPUT_BLOCK)
            chat.app.register_action_param("http_client", responses.client)
            response = chat.app.generate(messages=[{"role": "user", "content": INPUT_TEXT}])

        assert response["role"] == "exception"
        assert response["content"]["type"] == "HiddenlayerRailException"

    def test_output_block_raises_an_exception_when_enabled(self):
        chat = _v1_chat(llm_completions=[OUTPUT_TEXT], enable_rails_exceptions=True)
        with RecordedHTTPResponses() as responses:
            _token(responses)
            _verdict(responses, INPUT_NONE)
            _verdict(responses, OUTPUT_BLOCK, url=RESPONSE_URL)
            chat.app.register_action_param("http_client", responses.client)
            response = chat.app.generate(messages=[{"role": "user", "content": INPUT_TEXT}])

        assert response["role"] == "exception"
        assert response["content"]["type"] == "HiddenlayerRailException"

    def test_input_redact_replaces_the_user_message(self):
        chat = _v1_chat(llm_completions=["Hello!"])
        with RecordedHTTPResponses() as responses:
            _token(responses)
            _verdict(responses, INPUT_REDACT)
            _verdict(responses, OUTPUT_NONE, url=RESPONSE_URL)
            chat.app.register_action_param("http_client", responses.client)
            response = chat.app.generate(
                messages=[{"role": "user", "content": INPUT_TEXT}], options={"output_vars": ["user_message"]}
            )

        assert response.output_data == {"user_message": INPUT_REDACTED}
        assert response.response[0]["content"] == "Hello!"

    def test_output_block_refuses(self):
        chat = _v1_chat(llm_completions=[OUTPUT_TEXT])
        with RecordedHTTPResponses() as responses:
            _token(responses)
            _verdict(responses, INPUT_NONE)
            _verdict(responses, OUTPUT_BLOCK, url=RESPONSE_URL)
            chat.app.register_action_param("http_client", responses.client)
            chat >> "Hello!"
            chat << REFUSAL

        assert _evaluations(responses.client)[1].url == RESPONSE_URL
        assert _evaluation_bodies(responses.client)[1]["check"] == "output"

    def test_output_redact_replaces_the_bot_message(self):
        chat = _v1_chat(llm_completions=[OUTPUT_TEXT])
        with RecordedHTTPResponses() as responses:
            _token(responses)
            _verdict(responses, INPUT_NONE)
            _verdict(responses, OUTPUT_REDACT, url=RESPONSE_URL)
            chat.app.register_action_param("http_client", responses.client)
            chat >> "Hello!"
            chat << OUTPUT_REDACTED

        assert _evaluation_bodies(responses.client)[1]["bot_message"] == OUTPUT_TEXT

    def test_missing_credentials_fail_closed(self, monkeypatch):
        monkeypatch.delenv("HIDDENLAYER_CLIENT_SECRET")
        chat = _v1_chat(llm_completions=[])
        chat.app.register_action_param("http_client", RecordingHTTPClient())
        chat >> "Hello!"
        chat << "I'm sorry, an internal error has occurred."


class TestColang2:
    @staticmethod
    def _chat(enable_rails_exceptions: bool = False) -> TestChat:
        yaml_content = V2_CONFIG + ("enable_rails_exceptions: true\n" if enable_rails_exceptions else "")
        return TestChat(RailsConfig.from_content(yaml_content=yaml_content, colang_content=V2_FLOWS))

    def test_input_block_sends_an_exception_instead_of_refusing_when_enabled(self):
        chat = self._chat(enable_rails_exceptions=True)
        with RecordedHTTPResponses() as responses:
            _token(responses)
            _verdict(responses, INPUT_BLOCK)
            chat.app.register_action_param("http_client", responses.client)
            response = chat.app.generate(messages=[{"role": "user", "content": INPUT_TEXT}])

        # The flow takes the exception branch rather than `bot refuse to respond`, so no refusal
        # is uttered and the output rail never runs. Colang 2 does not promote a rail exception
        # to a message role the way Colang 1 does, so the absence of the refusal is what is
        # observable here.
        assert REFUSAL not in response["content"]
        assert len(_evaluations(responses.client)) == 1

    def test_output_block_sends_an_exception_instead_of_refusing_when_enabled(self):
        chat = self._chat(enable_rails_exceptions=True)
        with RecordedHTTPResponses() as responses:
            _token(responses)
            _verdict(responses, INPUT_NONE)
            _verdict(responses, OUTPUT_BLOCK, url=RESPONSE_URL)
            chat.app.register_action_param("http_client", responses.client)
            response = chat.app.generate(messages=[{"role": "user", "content": INPUT_TEXT}])

        assert REFUSAL not in response["content"]

    def test_input_redact_replaces_the_user_message(self):
        chat = self._chat()
        with RecordedHTTPResponses() as responses:
            _token(responses)
            _verdict(responses, INPUT_REDACT)
            _verdict(responses, OUTPUT_NONE, url=RESPONSE_URL)
            chat.app.register_action_param("http_client", responses.client)
            chat >> INPUT_TEXT
            chat << INPUT_REDACTED

        assert _evaluation_bodies(responses.client)[1]["bot_message"] == INPUT_REDACTED

    def test_output_block_refuses(self):
        chat = self._chat()
        with RecordedHTTPResponses() as responses:
            _token(responses)
            _verdict(responses, INPUT_NONE)
            _verdict(responses, OUTPUT_BLOCK, url=RESPONSE_URL)
            chat.app.register_action_param("http_client", responses.client)
            chat >> "Hello!"
            chat << REFUSAL

    def test_output_redact_replaces_the_bot_message(self):
        chat = self._chat()
        with RecordedHTTPResponses() as responses:
            _token(responses)
            _verdict(responses, INPUT_NONE)
            _verdict(responses, OUTPUT_REDACT, url=RESPONSE_URL)
            chat.app.register_action_param("http_client", responses.client)
            chat >> OUTPUT_TEXT
            chat << OUTPUT_REDACTED


class TestRailsConfigPrecedence:
    @staticmethod
    def _config(fields: str = "") -> RailsConfig:
        return RailsConfig.from_content(
            yaml_content=f"""
            models:
              - type: main
                engine: openai
                model: gpt-5-nano
            rails:
              config:
                hiddenlayer:
                  retry_backoff_seconds: 0
                  {fields}
            """
        )

    @staticmethod
    async def _sent_request(config: RailsConfig, api_url: str = API_URL):
        with RecordedHTTPResponses() as responses:
            responses.post(f"{api_url}/oauth2/token", payload={"access_token": TOKEN, "expires_in": 3600})
            _verdict(responses, INPUT_NONE, url=f"{api_url}{_REQUEST_PATH}")
            await hiddenlayer_evaluate(surface="input", text="hi", config=config, http_client=responses.client)
        return responses.client.requests[1]

    @pytest.fixture
    def env_settings(self, monkeypatch):
        monkeypatch.setenv("HIDDENLAYER_BASE_URL", "https://api.eu.hiddenlayer.ai")
        monkeypatch.setenv("HIDDENLAYER_PROJECT_ID", "env-project")
        monkeypatch.setenv("HIDDENLAYER_REQUESTER_ID", "env-requester")

    @pytest.mark.asyncio
    async def test_config_fields_win_over_the_environment(self, env_settings):
        config = self._config(
            "api_url: https://hl.example.com/\n                  project_id: field-project"
            "\n                  requester_id: field-requester"
        )

        request = await self._sent_request(config, api_url="https://hl.example.com")

        assert request.headers["HL-Project-Id"] == "field-project"
        assert request.headers["HL-Requester-Id"] == "field-requester"
        assert request.headers["HL-Provider-Id"] == "openai"
        assert request.json["model"] == "gpt-5-nano"

    @pytest.mark.asyncio
    async def test_environment_fills_unset_fields(self, env_settings):
        request = await self._sent_request(self._config(), api_url="https://api.eu.hiddenlayer.ai")

        assert request.headers["HL-Project-Id"] == "env-project"
        assert request.headers["HL-Requester-Id"] == "env-requester"

    @pytest.mark.asyncio
    async def test_defaults_apply_without_field_or_environment(self):
        request = await self._sent_request(self._config())

        assert "HL-Project-Id" not in request.headers
        assert request.headers["HL-Requester-Id"] == "hiddenlayer-nemo"

    def test_unknown_key_fails_the_load(self):
        with pytest.raises(ValidationError, match="projectid"):
            self._config("projectid: typo")

    def test_insecure_api_url_fails_the_load(self):
        with pytest.raises(ValidationError, match="https://"):
            self._config("api_url: http://api.us.hiddenlayer.ai")


class TestExampleConfig:
    def test_guardrails_selects_iorails_and_compiles_all_four_surfaces(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        rails = Guardrails(RailsConfig.from_path(str(EXAMPLE_CONFIG)), require_iorails=True)

        assert rails.use_iorails_engine
        manager = rails.rails_engine.rails_manager
        assert {flow for _, flow in manager._rails} == {"hiddenlayer check input", "hiddenlayer check output"}
        assert {flow for _, flow in manager._per_tool_rails} == {
            "hiddenlayer check tool call",
            "hiddenlayer check tool result",
        }
