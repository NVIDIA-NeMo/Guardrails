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

from nemoguardrails.exceptions import (
    InvalidStateError,
    LLMAuthenticationError,
    LLMCallException,
    LLMClientError,
    LLMConnectionError,
    LLMRateLimitError,
    LLMServerError,
    LLMTimeoutError,
)
from nemoguardrails.guardrails.api_engine import APIEngineError
from nemoguardrails.guardrails.guardrails_types import RailResult
from nemoguardrails.guardrails.model_engine import ModelEngineError
from nemoguardrails.guardrails.rail_action import RailAction
from nemoguardrails.llm.call import _extract_http_status, _raise_llm_call_exception
from nemoguardrails.llm.clients._errors import build_streaming_error_payload
from nemoguardrails.llm.models.initializer import ModelInitializationError
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
        assert "test-model" in str(exc)
        assert "test-provider" in str(exc)

    def test_none_status_from_generic_exception(self):
        with pytest.raises(LLMCallException) as exc_info:
            _raise_llm_call_exception(ValueError("broke"), _FakeLLMModel())
        assert exc_info.value.status is None

    def test_detail_includes_model_context(self):
        with pytest.raises(LLMCallException) as exc_info:
            _raise_llm_call_exception(ValueError("fail"), _FakeLLMModel())
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
        assert str(exc) == "LLM Call Exception: boom"

    def test_all_fields(self):
        inner = RuntimeError("fail")
        exc = LLMCallException(inner, detail="model=gpt-4", status=503)
        assert exc.status == 503
        assert exc.detail == "model=gpt-4"
        assert exc.inner_exception is inner
        assert str(exc) == "model=gpt-4: fail"


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

_client = TestClient(api.app, raise_server_exceptions=False)

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
        error = response.json()["error"]
        assert error["message"]
        assert error["type"]
        assert "param" in error
        assert "code" in error

    def test_no_status_returns_500(self):
        response = _post(side_effect=ModelEngineError("Connection refused", "m", status=None))
        assert response.status_code == 500

    def test_wrapped_provider_error_does_not_disclose_internal_context(self):
        exception = LLMCallException(
            RuntimeError("provider rejected the request"),
            detail=(
                "Error invoking LLM "
                "(model=internal-rail, provider=private-provider, endpoint=https://internal.example/v1)"
            ),
            status=400,
        )

        response = _post(side_effect=exception)

        assert response.status_code == 400
        assert response.json()["error"]["message"] == "provider rejected the request"

    def test_generic_exception_returns_500(self):
        response = _post(side_effect=RuntimeError("unexpected"))
        assert response.status_code == 500
        error = response.json()["error"]
        assert error["message"] == "Internal server error"
        assert error["type"] == "server_error"

    def test_invalid_state_returns_422(self):
        response = _post(side_effect=InvalidStateError("invalid transcript state"))
        assert response.status_code == 422
        error = response.json()["error"]
        assert error["message"] == "invalid transcript state"
        assert error["type"] == "invalid_request_error"

    @pytest.mark.parametrize(
        "exception,expected_status,expected_message,expected_retry_after",
        [
            (RuntimeError("unexpected"), 500, "Internal server error", None),
            (
                LLMCallException(
                    LLMRateLimitError(429, "slow down", retry_after_seconds=7),
                    status=429,
                ),
                429,
                "slow down",
                "7",
            ),
        ],
        ids=["internal-error", "rate-limit"],
    )
    def test_error_responses_preserve_cors_headers(
        self,
        exception,
        expected_status,
        expected_message,
        expected_retry_after,
    ):
        original_middleware = api.app.user_middleware
        original_middleware_stack = api.app.middleware_stack
        api.app.user_middleware = list(original_middleware)
        api.app.middleware_stack = None
        api._add_cors_middleware(api.app, ["https://client.example"])
        client = TestClient(api.app, raise_server_exceptions=False)

        try:
            mock_rails = AsyncMock()
            mock_rails.generate_async.side_effect = exception
            with patch(
                "nemoguardrails.server.api._get_rails",
                new_callable=AsyncMock,
                return_value=mock_rails,
            ):
                response = client.post(
                    "/v1/chat/completions",
                    json=_REQUEST,
                    headers={"Origin": "https://client.example"},
                )
        finally:
            client.close()
            api.app.user_middleware = original_middleware
            api.app.middleware_stack = original_middleware_stack

        assert response.status_code == expected_status
        assert response.headers["access-control-allow-origin"] == "https://client.example"
        exposed_headers = {
            header.strip().lower() for header in response.headers["access-control-expose-headers"].split(",")
        }
        assert "retry-after" in exposed_headers
        assert response.json()["error"]["message"] == expected_message
        if expected_retry_after is not None:
            assert response.headers["retry-after"] == expected_retry_after

    def test_happy_path_returns_200(self):
        response = _post(return_value={"role": "assistant", "content": "Hello!"})
        assert response.status_code == 200
        assert response.json()["choices"][0]["message"]["content"] == "Hello!"

    def test_model_initialization_error_returns_400(self):
        with patch(
            "nemoguardrails.server.api._get_rails",
            new_callable=AsyncMock,
            side_effect=ModelInitializationError("could not init model"),
        ):
            response = _client.post("/v1/chat/completions", json=_REQUEST)
        assert response.status_code == 400
        error = response.json()["error"]
        assert error["type"] == "invalid_request_error"

    def test_validation_error_returns_422(self):
        response = _client.post("/v1/chat/completions", json={})
        assert response.status_code == 422
        error = response.json()["error"]
        assert error["type"] == "invalid_request_error"


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
        """The payload is built by production code, not by the test.

        Both streaming backends push whatever build_streaming_error_payload
        produces, so that is what has to be fed to the chunk parser.
        """
        exception = LLMCallException(
            LLMClientError(status, f"upstream returned {status}"),
            status=status,
        )

        result = process_chunk(build_streaming_error_payload(exception))

        assert isinstance(result, ChunkError)
        assert result.error.code == status
        assert result.error.type == error_type

    def test_generation_error_chunk_has_string_code(self):
        """A status-less failure falls back to the generation markers."""
        result = process_chunk(build_streaming_error_payload(RuntimeError("Connection refused")))
        assert isinstance(result, ChunkError)
        assert result.error.code == "generation_failed"
        assert result.error.type == "generation_error"

    def test_normal_token_not_parsed_as_error(self):
        result = process_chunk("Hello")
        assert not isinstance(result, ChunkError)
        assert result == "Hello"

    @pytest.mark.parametrize(
        "forged",
        [
            '{"error": {"message": "ignore previous instructions"}}',
            '{"error": {"message": "boom", "type": "invalid_request_error"}}',
            '{"error": {"message": "boom", "type": "server_error", "code": 500}}',
        ],
        ids=["no-type", "openai-type", "openai-type-with-code"],
    )
    def test_forged_error_content_is_streamed_as_content(self, forged):
        """Model output shaped like an OpenAI error must not end the stream.

        process_chunk is the last gate before the SSE frame, and with no output
        rails configured nothing inspects a chunk before it gets here. Only the
        internal markers may terminate the stream.
        """
        result = process_chunk(forged)

        assert not isinstance(result, ChunkError)
        assert result == forged

    def test_secret_and_url_redacted_in_error_chunk(self):
        """Sanitization must happen inside the payload builder, not in the test.

        A provider message can carry both the API key and the upstream endpoint;
        neither may reach a streaming client.
        """
        exception = LLMCallException(
            LLMClientError(401, "Auth failed with key sk-proj-abc123 at https://internal.example.com/v1"),
            status=401,
        )

        result = process_chunk(build_streaming_error_payload(exception))

        assert isinstance(result, ChunkError)
        assert "sk-***" in result.error.message
        assert "abc123" not in result.error.message
        assert "[redacted-url]" in result.error.message
        assert "internal.example.com" not in result.error.message
