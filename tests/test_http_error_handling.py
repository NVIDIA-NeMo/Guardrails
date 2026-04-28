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

"""HTTP error handling tests.

Tests the full error propagation path: status extraction from provider
exceptions → LLMCallException / ModelEngineError / APIEngineError →
RailAction routing → API endpoint HTTP status codes.
"""

from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("openai", reason="openai is required for these tests")

from fastapi.testclient import TestClient

from nemoguardrails.actions.llm.utils import _extract_http_status, _raise_llm_call_exception
from nemoguardrails.exceptions import (
    LLMAuthenticationError,
    LLMCallException,
    LLMConnectionError,
    LLMRateLimitError,
    LLMServerError,
    LLMTimeoutError,
)
from nemoguardrails.guardrails.api_engine import APIEngineError
from nemoguardrails.guardrails.guardrails_types import RailResult
from nemoguardrails.guardrails.model_engine import ModelEngineError
from nemoguardrails.guardrails.rail_action import RailAction
from nemoguardrails.server import api
from nemoguardrails.server.api import ChunkError, process_chunk

# ---------------------------------------------------------------------------
# 1. _extract_http_status
# ---------------------------------------------------------------------------


class TestExtractHttpStatus:
    @pytest.mark.parametrize(
        "exception,expected",
        [
            (LLMAuthenticationError(401, "Unauthorized"), 401),
            (LLMRateLimitError(429, "Rate limited"), 429),
            (LLMServerError(503, "Unavailable"), 503),
        ],
        ids=["auth-401", "rate-limit-429", "server-503"],
    )
    def test_llm_client_error_returns_status(self, exception, expected):
        assert _extract_http_status(exception) == expected

    @pytest.mark.parametrize(
        "exception",
        [LLMTimeoutError(0, "Timed out"), LLMConnectionError(0, "Refused")],
        ids=["timeout", "connection"],
    )
    def test_zero_status_code_returns_none(self, exception):
        assert _extract_http_status(exception) is None

    def test_generic_exception_returns_none(self):
        assert _extract_http_status(ValueError("boom")) is None

    def test_third_party_status_code_attr(self):
        class OpenAIError(Exception):
            status_code = 401

        assert _extract_http_status(OpenAIError()) == 401

    def test_response_status_code_attr(self):
        class FakeResponse:
            status_code = 503

        class RequestsError(Exception):
            response = FakeResponse()

        assert _extract_http_status(RequestsError()) == 503

    def test_non_int_status_code_ignored(self):
        class WeirdError(Exception):
            status_code = "not-a-number"

        assert _extract_http_status(WeirdError()) is None


# ---------------------------------------------------------------------------
# 2. _raise_llm_call_exception
# ---------------------------------------------------------------------------


class _FakeLLMModel:
    model_name = "test-model"
    provider_name = "test-provider"
    provider_url = "http://localhost:8000"


class TestRaiseLLMCallException:
    def test_propagates_status_from_inner(self):
        with pytest.raises(LLMCallException) as exc_info:
            _raise_llm_call_exception(LLMAuthenticationError(401, "Bad key"), _FakeLLMModel())
        exc = exc_info.value
        assert exc.status == 401
        assert exc.inner_exception.status_code == 401
        assert exc.__cause__ is exc.inner_exception

    def test_none_status_from_generic_exception(self):
        with pytest.raises(LLMCallException) as exc_info:
            _raise_llm_call_exception(ValueError("broke"), _FakeLLMModel())
        assert exc_info.value.status is None

    def test_detail_includes_model_context(self):
        with pytest.raises(LLMCallException) as exc_info:
            _raise_llm_call_exception(LLMServerError(500, "fail"), _FakeLLMModel())
        assert "test-model" in str(exc_info.value)
        assert "test-provider" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 3. LLMCallException construction
# ---------------------------------------------------------------------------


class TestLLMCallException:
    def test_default_status_is_none(self):
        exc = LLMCallException(ValueError("boom"))
        assert exc.status is None
        assert exc.detail is None

    def test_all_fields(self):
        inner = RuntimeError("fail")
        exc = LLMCallException(inner, detail="model=gpt-4", status=503)
        assert exc.status == 503
        assert exc.detail == "model=gpt-4"
        assert exc.inner_exception is inner
        assert "fail" in str(exc)


# ---------------------------------------------------------------------------
# 4. RailAction exception routing
# ---------------------------------------------------------------------------


class _ErrorRailAction(RailAction):
    action_name = "test rail"
    requires_model = False
    exception_to_raise: Optional[Exception] = None

    def _extract_messages(self, messages, bot_response):
        return {"user_input": "test"}

    def _create_prompt(self, model_type, extracted):
        return [{"role": "user", "content": "test"}]

    async def _get_response(self, model_type, prompt):
        if self.exception_to_raise is not None:
            raise self.exception_to_raise
        return "safe"

    def _parse_response(self, response):
        return RailResult(is_safe=True)


_MESSAGES = [{"role": "user", "content": "hi"}]


@pytest.fixture
def rail_action():
    return _ErrorRailAction(engine_registry=MagicMock(), task_manager=MagicMock())


class TestRailActionExceptionRouting:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "exception,expected_status",
        [
            (ModelEngineError("HTTP 401", "model", status=401), 401),
            (APIEngineError("HTTP 404", "/ep", status=404), 404),
        ],
        ids=["model-401", "api-404"],
    )
    async def test_engine_error_with_status_reraises(self, rail_action, exception, expected_status):
        rail_action.exception_to_raise = exception
        with pytest.raises(type(exception)) as exc_info:
            await rail_action.run("test rail", _MESSAGES)
        assert exc_info.value.status == expected_status

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "exception",
        [
            ModelEngineError("Connection refused", "model", status=None),
            APIEngineError("DNS failed", "/ep", status=None),
            RuntimeError("unexpected"),
        ],
        ids=["model-no-status", "api-no-status", "generic"],
    )
    async def test_error_without_status_fails_closed(self, rail_action, exception):
        rail_action.exception_to_raise = exception
        result = await rail_action.run("test rail", _MESSAGES)
        assert not result.is_safe

    @pytest.mark.asyncio
    async def test_happy_path(self, rail_action):
        result = await rail_action.run("test rail", _MESSAGES)
        assert result.is_safe


# ---------------------------------------------------------------------------
# 5. API endpoint integration
# ---------------------------------------------------------------------------

_client = TestClient(api.app)

_REQUEST = {
    "model": "test-model",
    "messages": [{"role": "user", "content": "hello"}],
    "guardrails": {"config_id": "test-config"},
}


def _post(side_effect=None, return_value=None):
    mock_rails = AsyncMock()
    if side_effect is not None:
        mock_rails.generate_async.side_effect = side_effect
    else:
        mock_rails.generate_async.return_value = return_value
    with patch("nemoguardrails.server.api._get_rails", new_callable=AsyncMock, return_value=mock_rails):
        return _client.post("/v1/chat/completions", json=_REQUEST)


class TestAPIErrorPropagation:
    @pytest.mark.parametrize(
        "exception,expected_status",
        [
            (LLMCallException("Unauthorized", status=401), 401),
            (LLMCallException("Forbidden", status=403), 403),
            (ModelEngineError("Not found", "m", status=404), 404),
            (APIEngineError("Rate limited", "/ep", status=429), 429),
            (ModelEngineError("Internal server error", "m", status=500), 500),
            (ModelEngineError("Bad gateway", "m", status=502), 502),
            (APIEngineError("Service unavailable", "/ep", status=503), 503),
        ],
        ids=[
            "401-unauthorized",
            "403-forbidden",
            "404-not-found",
            "429-rate-limit",
            "500-server",
            "502-bad-gateway",
            "503-unavailable",
        ],
    )
    def test_downstream_error_returns_status(self, exception, expected_status):
        response = _post(side_effect=exception)
        assert response.status_code == expected_status
        body = response.json()
        assert body["id"].startswith("chatcmpl-")
        assert body["object"] == "chat.completion"
        assert body["model"] == "test-model"
        assert body["choices"][0]["message"]["role"] == "assistant"

    def test_no_status_returns_500(self):
        response = _post(side_effect=ModelEngineError("Connection refused", "m", status=None))
        assert response.status_code == 500

    def test_generic_exception_returns_500(self):
        response = _post(side_effect=RuntimeError("unexpected"))
        assert response.status_code == 500
        assert response.json()["choices"][0]["message"]["content"] == "Internal server error"

    def test_happy_path_returns_200(self):
        response = _post(return_value={"role": "assistant", "content": "Hello!"})
        assert response.status_code == 200
        assert response.json()["choices"][0]["message"]["content"] == "Hello!"


# ---------------------------------------------------------------------------
# 6. Streaming SSE error chunks
# ---------------------------------------------------------------------------


class TestStreamingErrorChunks:
    @pytest.mark.parametrize(
        "status,error_type",
        [(401, "downstream_error"), (429, "downstream_error"), (503, "downstream_error")],
        ids=["401", "429", "503"],
    )
    def test_iorails_error_chunk_carries_status(self, status, error_type):
        import json

        from nemoguardrails.llm.clients._errors import _redact_secrets

        error_payload = json.dumps(
            {
                "error": {
                    "message": _redact_secrets("HTTP error"),
                    "type": error_type,
                    "code": status,
                }
            }
        )
        result = process_chunk(error_payload)
        assert isinstance(result, ChunkError)
        assert result.error.code == status
        assert result.error.type == "downstream_error"

    def test_generation_error_chunk_has_string_code(self):
        import json

        error_payload = json.dumps(
            {"error": {"message": "Connection refused", "type": "generation_error", "code": "generation_failed"}}
        )
        result = process_chunk(error_payload)
        assert isinstance(result, ChunkError)
        assert result.error.code == "generation_failed"
        assert result.error.type == "generation_error"

    def test_normal_token_not_parsed_as_error(self):
        result = process_chunk("Hello")
        assert not isinstance(result, ChunkError)
        assert result == "Hello"

    def test_secret_redacted_in_error_chunk(self):
        import json

        from nemoguardrails.llm.clients._errors import _redact_secrets

        raw = "Auth failed with key sk-proj-abc123"
        error_payload = json.dumps({"error": {"message": _redact_secrets(raw), "code": 401}})
        result = process_chunk(error_payload)
        assert isinstance(result, ChunkError)
        assert "sk-***" in result.error.message
        assert "abc123" not in result.error.message
