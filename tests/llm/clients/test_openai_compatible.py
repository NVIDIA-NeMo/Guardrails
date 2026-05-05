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

import asyncio
from contextlib import asynccontextmanager
from unittest import mock

import httpx
import pytest

from nemoguardrails.exceptions import (
    LLMAuthenticationError,
    LLMBadRequestError,
    LLMClientError,
    LLMConnectionError,
    LLMContextWindowError,
    LLMRateLimitError,
    LLMServerError,
    LLMTimeoutError,
    LLMUnsupportedParamsError,
)
from nemoguardrails.llm.clients.base import BaseClient
from nemoguardrails.llm.clients.constants import INITIAL_RETRY_DELAY, MAX_RETRY_AFTER, MAX_RETRY_DELAY
from nemoguardrails.llm.clients.openai_compatible import OpenAICompatibleClient
from tests.llm.clients._helpers import (
    consume,
    low_retry_delay,
    make_client,
    mock_httpx_post,
    ok_response,
    stream_client,
    tracking_mock_stream,
)


class TestProviderUrl:
    @pytest.mark.asyncio
    async def test_provider_url_returns_base_url(self):
        async with OpenAICompatibleClient(base_url="https://api.openai.com/v1") as c:
            assert c.provider_url == "https://api.openai.com/v1"

    @pytest.mark.asyncio
    async def test_provider_url_strips_trailing_slash(self):
        async with OpenAICompatibleClient(base_url="https://api.openai.com/v1/") as c:
            assert c.provider_url == "https://api.openai.com/v1"


class TestChatCompletion:
    @pytest.mark.asyncio
    async def test_returns_http_response(self):
        from nemoguardrails.llm.clients.base import HTTPResponse

        client = make_client()
        mock_httpx_post(client, [(200, ok_response(), {})])
        result = await client.chat_completion("gpt-4o", [{"role": "user", "content": "Hi"}])
        assert isinstance(result, HTTPResponse)
        assert result.body["choices"][0]["message"]["content"] == "Hello"
        assert result.headers is not None

    @pytest.mark.asyncio
    async def test_builds_correct_payload(self):
        client = make_client()
        captured = {}

        async def capturing_post(*args, **kwargs):
            captured.update(kwargs)
            return httpx.Response(200, json=ok_response(), request=httpx.Request("POST", "url"))

        client._client = type("MockClient", (), {"post": capturing_post})()
        await client.chat_completion("gpt-4o", [{"role": "user", "content": "Hi"}], stop=["END"], temperature=0.5)

        payload = captured["json"]
        assert payload["model"] == "gpt-4o"
        assert payload["messages"] == [{"role": "user", "content": "Hi"}]
        assert payload["stop"] == ["END"]
        assert payload["temperature"] == 0.5
        assert "stream" not in payload

    @pytest.mark.asyncio
    async def test_stream_sets_stream_options(self):
        client = make_client()
        captured = {}

        @asynccontextmanager
        async def capturing_stream(*args, **kwargs):
            captured.update(kwargs)

            class FakeResponse:
                status_code = 200
                headers = {}

                async def aread(self):
                    pass

                async def aiter_lines(self):
                    yield 'data: {"id":"c","choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":"stop"}]}'
                    yield ""
                    yield "data: [DONE]"
                    yield ""

            yield FakeResponse()

        client._client = type("MockClient", (), {"stream": capturing_stream})()
        async for _ in client.stream_chat_completion("gpt-4o", [{"role": "user", "content": "Hi"}]):
            pass

        payload = captured["json"]
        assert payload["stream"] is True
        assert payload["stream_options"] == {"include_usage": True}


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_401_raises_auth_error(self):
        client = make_client()
        mock_httpx_post(client, [(401, {"error": {"message": "Invalid key"}}, {})])
        with pytest.raises(LLMAuthenticationError) as exc_info:
            await client.chat_completion("gpt-4o", [])
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_403_raises_auth_error(self):
        client = make_client()
        mock_httpx_post(client, [(403, {"error": {"message": "Denied"}}, {})])
        with pytest.raises(LLMAuthenticationError) as exc_info:
            await client.chat_completion("gpt-4o", [])
        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_429_parses_retry_after(self):
        client = make_client(max_retries=0)
        mock_httpx_post(client, [(429, {"error": {"message": "Rate limited"}}, {"retry-after": "20"})])
        with pytest.raises(LLMRateLimitError) as exc_info:
            await client.chat_completion("gpt-4o", [])
        assert exc_info.value.retry_after_seconds == 20.0

    @pytest.mark.asyncio
    async def test_400_bad_request_with_param(self):
        client = make_client()
        mock_httpx_post(client, [(400, {"error": {"message": "Invalid temperature", "param": "temperature"}}, {})])
        with pytest.raises(LLMBadRequestError) as exc_info:
            await client.chat_completion("gpt-4o", [])
        assert exc_info.value.param == "temperature"

    @pytest.mark.asyncio
    async def test_400_context_window(self):
        client = make_client()
        mock_httpx_post(client, [(400, {"error": {"message": "maximum context length exceeded"}}, {})])
        with pytest.raises(LLMContextWindowError):
            await client.chat_completion("gpt-4o", [])

    @pytest.mark.asyncio
    async def test_400_unsupported_params(self):
        client = make_client()
        mock_httpx_post(client, [(400, {"error": {"message": "temperature is not supported"}}, {})])
        with pytest.raises(LLMUnsupportedParamsError):
            await client.chat_completion("gpt-4o", [])

    @pytest.mark.asyncio
    async def test_422_bad_request(self):
        client = make_client()
        mock_httpx_post(client, [(422, {"error": {"message": "Invalid schema"}}, {})])
        with pytest.raises(LLMBadRequestError) as exc_info:
            await client.chat_completion("gpt-4o", [])
        assert exc_info.value.status_code == 422

    @pytest.mark.asyncio
    async def test_500_server_error(self):
        client = make_client(max_retries=0)
        mock_httpx_post(client, [(500, {"error": {"message": "Server error"}}, {})])
        with pytest.raises(LLMServerError):
            await client.chat_completion("gpt-4o", [])

    @pytest.mark.asyncio
    async def test_503_server_error(self):
        client = make_client(max_retries=0)
        mock_httpx_post(client, [(503, {"error": {"message": "Overloaded"}}, {})])
        with pytest.raises(LLMServerError):
            await client.chat_completion("gpt-4o", [])

    @pytest.mark.asyncio
    async def test_redacts_api_key_in_message(self):
        client = make_client()
        mock_httpx_post(client, [(401, {"error": {"message": "Key: sk-proj-abc123"}}, {})])
        with pytest.raises(LLMAuthenticationError) as exc_info:
            await client.chat_completion("gpt-4o", [])
        assert "sk-proj-abc123" not in exc_info.value.error_message
        assert "sk-***" in exc_info.value.error_message

    @pytest.mark.asyncio
    async def test_vllm_flat_format(self):
        client = make_client()
        mock_httpx_post(client, [(400, {"message": "Not supported", "object": "error"}, {})])
        with pytest.raises(LLMBadRequestError) as exc_info:
            await client.chat_completion("gpt-4o", [])
        assert "Not supported" in exc_info.value.error_message

    @pytest.mark.asyncio
    async def test_detail_format(self):
        client = make_client()
        mock_httpx_post(client, [(422, {"detail": "Validation error"}, {})])
        with pytest.raises(LLMBadRequestError) as exc_info:
            await client.chat_completion("gpt-4o", [])
        assert "Validation error" in exc_info.value.error_message

    @pytest.mark.asyncio
    async def test_unparseable_body(self):
        client = make_client(max_retries=0)
        mock_httpx_post(client, [(500, "Internal Server Error", {})])
        with pytest.raises(LLMServerError) as exc_info:
            await client.chat_completion("gpt-4o", [])
        assert "Internal Server Error" in exc_info.value.error_message

    @pytest.mark.asyncio
    async def test_error_has_body_and_headers(self):
        client = make_client()
        mock_httpx_post(client, [(401, {"error": {"message": "Bad"}}, {"x-request-id": "req-1"})])
        with pytest.raises(LLMAuthenticationError) as exc_info:
            await client.chat_completion("gpt-4o", [])
        assert exc_info.value.body is not None
        assert exc_info.value.response_headers is not None

    @pytest.mark.asyncio
    async def test_408_maps_to_timeout_error(self):
        client = make_client(max_retries=0)
        mock_httpx_post(client, [(408, {"error": {"message": "Request timeout"}}, {})])
        with pytest.raises(LLMTimeoutError) as exc_info:
            await client.chat_completion("gpt-4o", [])
        assert exc_info.value.status_code == 408


class TestRetry:
    @pytest.mark.asyncio
    async def test_retries_on_500(self):
        client = make_client()
        get_count = mock_httpx_post(
            client,
            [
                (500, {"error": {"message": "Error"}}, {}),
                (200, ok_response(), {}),
            ],
        )
        await client.chat_completion("gpt-4o", [])
        assert get_count() == 2

    @pytest.mark.asyncio
    async def test_retries_on_429(self):
        client = make_client()
        get_count = mock_httpx_post(
            client,
            [
                (429, {"error": {"message": "Rate limited"}}, {"retry-after": "0.01"}),
                (200, ok_response(), {}),
            ],
        )
        await client.chat_completion("gpt-4o", [])
        assert get_count() == 2

    @pytest.mark.asyncio
    async def test_retries_on_502(self):
        client = make_client()
        get_count = mock_httpx_post(
            client,
            [
                (502, {"error": {"message": "Bad gateway"}}, {}),
                (200, ok_response(), {}),
            ],
        )
        await client.chat_completion("gpt-4o", [])
        assert get_count() == 2

    @pytest.mark.asyncio
    async def test_no_retry_on_400(self):
        client = make_client()
        get_count = mock_httpx_post(client, [(400, {"error": {"message": "Bad"}}, {})])
        with pytest.raises(LLMBadRequestError):
            await client.chat_completion("gpt-4o", [])
        assert get_count() == 1

    @pytest.mark.asyncio
    async def test_no_retry_on_401(self):
        client = make_client()
        get_count = mock_httpx_post(client, [(401, {"error": {"message": "Unauth"}}, {})])
        with pytest.raises(LLMAuthenticationError):
            await client.chat_completion("gpt-4o", [])
        assert get_count() == 1

    @pytest.mark.asyncio
    async def test_max_retries_exhausted(self):
        client = make_client(max_retries=2)
        get_count = mock_httpx_post(
            client,
            [
                (500, {"error": {"message": "Error"}}, {}),
                (500, {"error": {"message": "Error"}}, {}),
                (500, {"error": {"message": "Error"}}, {}),
            ],
        )
        with pytest.raises(LLMServerError):
            await client.chat_completion("gpt-4o", [])
        assert get_count() == 3

    @pytest.mark.asyncio
    async def test_zero_retries(self):
        client = make_client(max_retries=0)
        get_count = mock_httpx_post(client, [(500, {"error": {"message": "Error"}}, {})])
        with pytest.raises(LLMServerError):
            await client.chat_completion("gpt-4o", [])
        assert get_count() == 1

    @pytest.mark.asyncio
    async def test_x_should_retry_false(self):
        client = make_client()
        get_count = mock_httpx_post(client, [(500, {"error": {"message": "Error"}}, {"x-should-retry": "false"})])
        with pytest.raises(LLMServerError):
            await client.chat_completion("gpt-4o", [])
        assert get_count() == 1

    @pytest.mark.asyncio
    async def test_x_should_retry_true(self):
        client = make_client()
        get_count = mock_httpx_post(
            client,
            [
                (400, {"error": {"message": "Bad"}}, {"x-should-retry": "true"}),
                (200, ok_response(), {}),
            ],
        )
        await client.chat_completion("gpt-4o", [])
        assert get_count() == 2


class TestStreamErrorDetection:
    @pytest.mark.asyncio
    async def test_mid_stream_error_after_chunks_raises(self):
        client = stream_client(
            [
                'data: {"id":"c","choices":[{"index":0,"delta":{"content":"hi"}}]}',
                "",
                'data: {"error": {"message": "Server overloaded", "type": "overloaded_error"}}',
                "",
            ]
        )
        chunks = []
        with pytest.raises(LLMServerError):
            async for chunk in client.stream_chat_completion("gpt-4o", [{"role": "user", "content": "Hi"}]):
                chunks.append(chunk)
        assert len(chunks) == 1

    @pytest.mark.parametrize(
        "error_type,expected_cls",
        [
            ("invalid_request_error", LLMBadRequestError),
            ("authentication_error", LLMAuthenticationError),
            ("permission_error", LLMAuthenticationError),
            ("rate_limit_error", LLMRateLimitError),
            ("api_error", LLMServerError),
            ("server_error", LLMServerError),
            ("overloaded_error", LLMServerError),
        ],
    )
    @pytest.mark.asyncio
    async def test_mid_stream_error_type_maps_to_class(self, error_type, expected_cls):
        client = stream_client(
            [
                f'data: {{"error": {{"message": "err", "type": "{error_type}"}}}}',
                "",
            ]
        )
        with pytest.raises(expected_cls):
            await consume(client)

    @pytest.mark.asyncio
    async def test_mid_stream_error_numeric_code_fallback(self):
        client = stream_client(
            [
                'data: {"error": {"message": "Gateway down", "code": 502}}',
                "",
            ]
        )
        with pytest.raises(LLMServerError) as exc_info:
            await consume(client)
        assert exc_info.value.status_code == 502

    @pytest.mark.asyncio
    async def test_mid_stream_error_unknown_type_raises_generic(self):
        client = stream_client(
            [
                'data: {"error": {"message": "Something broke"}}',
                "",
            ]
        )
        with pytest.raises(LLMClientError) as exc_info:
            await consume(client)
        assert exc_info.value.status_code == 0
        assert "Something broke" in exc_info.value.error_message

    @pytest.mark.asyncio
    async def test_mid_stream_error_string_code_raises_generic(self):
        client = stream_client(
            [
                'data: {"error": {"message": "Bad request", "code": "invalid_request"}}',
                "",
            ]
        )
        with pytest.raises(LLMClientError) as exc_info:
            await consume(client)
        assert exc_info.value.status_code == 0

    @pytest.mark.asyncio
    async def test_mid_stream_rate_limit_retry_after_parsed(self):
        client = stream_client(
            [
                'data: {"error": {"message": "Slow down", "type": "rate_limit_error"}}',
                "",
            ],
            response_headers={"retry-after": "12"},
        )
        with pytest.raises(LLMRateLimitError) as exc_info:
            await consume(client)
        assert exc_info.value.retry_after_seconds == 12.0


@mock.patch(
    "nemoguardrails.llm.clients.base.BaseClient._calculate_retry_delay",
    staticmethod(low_retry_delay),
)
class TestNetworkExceptionRetry:
    @pytest.mark.asyncio
    async def test_timeout_retries_then_succeeds(self):
        client = make_client()
        call_count = 0

        async def flaky_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise httpx.ReadTimeout("timed out")
            return httpx.Response(
                200,
                json=ok_response(),
                request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
            )

        client._client = type("MockClient", (), {"post": flaky_post})()
        await client.chat_completion("gpt-4o", [])
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_connect_error_retries_then_succeeds(self):
        client = make_client()
        call_count = 0

        async def flaky_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise httpx.ConnectError("connection refused")
            return httpx.Response(
                200,
                json=ok_response(),
                request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
            )

        client._client = type("MockClient", (), {"post": flaky_post})()
        await client.chat_completion("gpt-4o", [])
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_timeout_exhausts_retries_raises_timeout(self):
        client = make_client(max_retries=2)
        call_count = 0

        async def always_timeout(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise httpx.ReadTimeout("timed out")

        client._client = type("MockClient", (), {"post": always_timeout})()
        with pytest.raises(LLMTimeoutError) as exc_info:
            await client.chat_completion("gpt-4o", [])
        assert call_count == 3
        assert exc_info.value.status_code == 0

    @pytest.mark.asyncio
    async def test_network_error_exhausts_retries_raises_connection(self):
        client = make_client(max_retries=1)
        call_count = 0

        async def always_connect_err(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise httpx.ConnectError("unreachable")

        client._client = type("MockClient", (), {"post": always_connect_err})()
        with pytest.raises(LLMConnectionError) as exc_info:
            await client.chat_completion("gpt-4o", [])
        assert call_count == 2
        assert exc_info.value.status_code == 0

    @pytest.mark.asyncio
    async def test_zero_retries_timeout_raises_immediately(self):
        client = make_client(max_retries=0)

        async def always_timeout(*args, **kwargs):
            raise httpx.ConnectTimeout("timed out")

        client._client = type("MockClient", (), {"post": always_timeout})()
        with pytest.raises(LLMTimeoutError):
            await client.chat_completion("gpt-4o", [])

    @pytest.mark.asyncio
    async def test_invalid_url_does_not_retry(self):
        client = make_client(max_retries=3)
        call_count = 0

        async def invalid_url(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise httpx.InvalidURL("bad url")

        client._client = type("MockClient", (), {"post": invalid_url})()
        with pytest.raises(httpx.InvalidURL):
            await client.chat_completion("gpt-4o", [])
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_stream_timeout_before_first_chunk_retries(self):
        call_count = 0

        @asynccontextmanager
        async def flaky_stream(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise httpx.ConnectTimeout("timed out")

            class FakeResponse:
                status_code = 200
                headers = {}

                async def aread(self):
                    pass

                async def aiter_lines(self):
                    yield 'data: {"id":"c","choices":[{"index":0,"delta":{"content":"hi"}}]}'
                    yield ""
                    yield "data: [DONE]"
                    yield ""

            yield FakeResponse()

        client = make_client()
        client._client = type("MockClient", (), {"stream": flaky_stream})()
        chunks = []
        async for chunk in client.stream_chat_completion("gpt-4o", [{"role": "user", "content": "Hi"}]):
            chunks.append(chunk)
        assert call_count == 2
        assert len(chunks) == 1

    @pytest.mark.asyncio
    async def test_stream_timeout_after_first_chunk_does_not_retry(self):
        call_count = 0

        @asynccontextmanager
        async def stream_then_timeout(*args, **kwargs):
            nonlocal call_count
            call_count += 1

            class FakeResponse:
                status_code = 200
                headers = {}

                async def aread(self):
                    pass

                async def aiter_lines(self):
                    yield 'data: {"id":"c","choices":[{"index":0,"delta":{"content":"hi"}}]}'
                    yield ""
                    raise httpx.ReadTimeout("mid-stream timeout")

            yield FakeResponse()

        client = make_client()
        client._client = type("MockClient", (), {"stream": stream_then_timeout})()
        chunks = []
        with pytest.raises(LLMTimeoutError) as exc_info:
            async for chunk in client.stream_chat_completion("gpt-4o", [{"role": "user", "content": "Hi"}]):
                chunks.append(chunk)
        assert call_count == 1
        assert len(chunks) == 1
        assert isinstance(exc_info.value.__cause__, httpx.ReadTimeout)

    @pytest.mark.asyncio
    async def test_stream_timeout_before_first_chunk_exhausts_retries_wrapped(self):
        call_count = 0

        @asynccontextmanager
        async def always_timeout(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise httpx.ConnectTimeout("pre-yield timeout")
            yield  # pragma: no cover — make this an async generator

        client = make_client(max_retries=2)
        client._client = type("MockClient", (), {"stream": always_timeout})()
        with pytest.raises(LLMTimeoutError) as exc_info:
            async for _ in client.stream_chat_completion("gpt-4o", [{"role": "user", "content": "Hi"}]):
                pass
        assert call_count == 3
        assert isinstance(exc_info.value.__cause__, httpx.ConnectTimeout)

    @pytest.mark.asyncio
    async def test_stream_network_error_before_first_chunk_exhausts_retries_wrapped(self):
        call_count = 0

        @asynccontextmanager
        async def always_connect_error(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise httpx.ConnectError("connection refused")
            yield  # pragma: no cover — make this an async generator

        client = make_client(max_retries=1)
        client._client = type("MockClient", (), {"stream": always_connect_error})()
        with pytest.raises(LLMConnectionError) as exc_info:
            async for _ in client.stream_chat_completion("gpt-4o", [{"role": "user", "content": "Hi"}]):
                pass
        assert call_count == 2
        assert isinstance(exc_info.value.__cause__, httpx.ConnectError)

    @pytest.mark.asyncio
    async def test_runtime_error_retries_then_succeeds(self):
        """RuntimeError (e.g. 'Event loop is closed') is treated as transient.

        When a cached httpx.AsyncClient is reused on a new event loop, the first
        send raises RuntimeError because the underlying transport is bound to
        the dead loop. httpx's connection pool invalidates the stale connection
        and the retry opens a fresh one in the running loop. The retry path
        only works if RuntimeError is caught by the loop in _apost.
        """
        client = make_client()
        call_count = 0

        async def flaky_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise RuntimeError("Event loop is closed")
            return httpx.Response(
                200,
                json=ok_response(),
                request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
            )

        client._client = type("MockClient", (), {"post": flaky_post})()
        await client.chat_completion("gpt-4o", [])
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_runtime_error_exhausts_retries_raises_connection(self):
        client = make_client(max_retries=1)
        call_count = 0

        async def always_runtime_error(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise RuntimeError("Event loop is closed")

        client._client = type("MockClient", (), {"post": always_runtime_error})()
        with pytest.raises(LLMConnectionError) as exc_info:
            await client.chat_completion("gpt-4o", [])
        assert call_count == 2
        assert isinstance(exc_info.value.__cause__, RuntimeError)

    @pytest.mark.asyncio
    async def test_stream_runtime_error_before_first_chunk_retries(self):
        call_count = 0

        @asynccontextmanager
        async def flaky_stream(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise RuntimeError("Event loop is closed")

            class FakeResponse:
                status_code = 200
                headers = {}

                async def aread(self):
                    pass

                async def aiter_lines(self):
                    yield 'data: {"id":"c","choices":[{"index":0,"delta":{"content":"hi"}}]}'
                    yield ""
                    yield "data: [DONE]"
                    yield ""

            yield FakeResponse()

        client = make_client()
        client._client = type("MockClient", (), {"stream": flaky_stream})()
        chunks = []
        async for chunk in client.stream_chat_completion("gpt-4o", [{"role": "user", "content": "Hi"}]):
            chunks.append(chunk)
        assert call_count == 2
        assert len(chunks) == 1

    @pytest.mark.asyncio
    async def test_stream_runtime_error_before_first_chunk_exhausts_retries_wrapped(self):
        call_count = 0

        @asynccontextmanager
        async def always_runtime_error(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise RuntimeError("Event loop is closed")
            yield  # pragma: no cover — make this an async generator

        client = make_client(max_retries=1)
        client._client = type("MockClient", (), {"stream": always_runtime_error})()
        with pytest.raises(LLMConnectionError) as exc_info:
            async for _ in client.stream_chat_completion("gpt-4o", [{"role": "user", "content": "Hi"}]):
                pass
        assert call_count == 2
        assert isinstance(exc_info.value.__cause__, RuntimeError)

    @pytest.mark.asyncio
    async def test_unrelated_runtime_error_propagates_without_retry(self):
        """Only stale-event-loop RuntimeErrors are treated as transient.

        A RuntimeError with any other message must propagate untouched so
        programmer bugs are not silently retried and re-wrapped as a
        connection error.
        """
        client = make_client()
        call_count = 0

        async def raise_unrelated(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise RuntimeError("dictionary changed size during iteration")

        client._client = type("MockClient", (), {"post": raise_unrelated})()
        with pytest.raises(RuntimeError, match="dictionary changed"):
            await client.chat_completion("gpt-4o", [])
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_runtime_error_with_max_retries_zero_raises_immediately(self):
        """With retries disabled, a stale-event-loop error surfaces clearly.

        The error is wrapped in LLMConnectionError so callers see a typed
        exception with the original RuntimeError as __cause__.
        """
        client = make_client(max_retries=0)
        call_count = 0

        async def stale_loop(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise RuntimeError("Event loop is closed")

        client._client = type("MockClient", (), {"post": stale_loop})()
        with pytest.raises(LLMConnectionError) as exc_info:
            await client.chat_completion("gpt-4o", [])
        assert call_count == 1
        assert isinstance(exc_info.value.__cause__, RuntimeError)
        assert "event loop" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_stream_unrelated_runtime_error_propagates_without_retry(self):
        call_count = 0

        @asynccontextmanager
        async def raise_unrelated(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise RuntimeError("dictionary changed size during iteration")
            yield  # pragma: no cover

        client = make_client()
        client._client = type("MockClient", (), {"stream": raise_unrelated})()
        with pytest.raises(RuntimeError, match="dictionary changed"):
            async for _ in client.stream_chat_completion("gpt-4o", [{"role": "user", "content": "Hi"}]):
                pass
        assert call_count == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "message",
        [
            "Event loop is closed",
            "got Future attached to a different loop",
            "Task got Future <Future pending> attached to a different loop",
            "<asyncio.locks.Event object at 0x10731c8a0 [unset]> is bound to a different event loop",
        ],
    )
    async def test_stale_loop_messages_all_retry(self, message):
        """All known stale-loop RuntimeError variants are retried.

        These are flavors of the same defect (a transport bound to one
        loop being reused on another). Verified empirically:
          * 'Event loop is closed' fires after asyncio.run closes its loop.
          * 'is bound to a different event loop' fires when an asyncio
            primitive (Lock/Event/Semaphore/Queue) is reused across loops.
          * 'got Future attached to a different loop' fires when a Future
            created in another loop is awaited.
        The retry path must absorb all of them.
        """
        client = make_client()
        call_count = 0

        async def flaky_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise RuntimeError(message)
            return httpx.Response(
                200,
                json=ok_response(),
                request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
            )

        client._client = type("MockClient", (), {"post": flaky_post})()
        await client.chat_completion("gpt-4o", [])
        assert call_count == 2

    @pytest.mark.parametrize("runs", [2, 5, 10])
    def test_recovers_when_same_client_used_across_asyncio_run_calls(self, runs):
        """Same client reused across N asyncio.run() boundaries.

        Each fresh loop sees one stale-transport RuntimeError on its first
        request and recovers via the retry path. Without the fix, the second
        asyncio.run raises 'Event loop is closed'.
        """
        client = make_client()
        first_call_per_loop = {"is_first": True}
        post_calls = 0

        async def post_simulating_loop_binding(*args, **kwargs):
            nonlocal post_calls
            post_calls += 1
            if first_call_per_loop["is_first"]:
                first_call_per_loop["is_first"] = False
                raise RuntimeError("Event loop is closed")
            return httpx.Response(
                200,
                json=ok_response(),
                request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
            )

        client._client = type("MockClient", (), {"post": post_simulating_loop_binding})()

        for _ in range(runs):
            first_call_per_loop["is_first"] = True
            result = asyncio.run(client.chat_completion("gpt-4o", [{"role": "user", "content": "hi"}]))
            assert result.body["choices"][0]["message"]["content"] == "Hello"

        assert post_calls == runs * 2

    @pytest.mark.parametrize("loops", [2, 5, 10])
    def test_recovers_when_same_client_used_across_function_scope_loops(self, loops):
        """Same client called from N freshly-created event loops.

        Mirrors the pytest-asyncio function-scope pattern (session-scoped
        client, function-scoped loops). Each new loop's first request sees
        a stale-transport RuntimeError and recovers via the retry. Verified
        across a range of loop counts to catch any per-loop accumulation.
        """
        client = make_client()
        first_call_per_loop = {"is_first": True}
        post_calls = 0

        async def post_simulating_loop_binding(*args, **kwargs):
            nonlocal post_calls
            post_calls += 1
            if first_call_per_loop["is_first"]:
                first_call_per_loop["is_first"] = False
                raise RuntimeError("Event loop is closed")
            return httpx.Response(
                200,
                json=ok_response(),
                request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
            )

        client._client = type("MockClient", (), {"post": post_simulating_loop_binding})()

        for _ in range(loops):
            first_call_per_loop["is_first"] = True
            loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(client.chat_completion("gpt-4o", [{"role": "user", "content": "hi"}]))
                assert result.body["choices"][0]["message"]["content"] == "Hello"
            finally:
                loop.close()
                asyncio.set_event_loop(None)

        assert post_calls == loops * 2

    def test_recovers_with_real_httpx_against_refused_endpoint_across_asyncio_run(self):
        """Real httpx.AsyncClient making real network attempts across asyncio.run().

        httpx reliably fails on the second asyncio.run() with RuntimeError:
        Event loop is closed. With our retry in place, the same client should surface
        a NETWORK error on the second call (proving recovery happened and we reached the
        connection-attempt phase), not a "Stale event loop" error (which
        would mean retries exhausted on consecutive RuntimeErrors and
        recovery never happened).
        """
        client = OpenAICompatibleClient(base_url="http://127.0.0.1:1", api_key="sk-test", max_retries=2)

        with pytest.raises(LLMConnectionError) as exc1:
            asyncio.run(client.chat_completion("gpt-4o", [{"role": "user", "content": "hi"}]))
        assert "Stale event loop" not in str(exc1.value)
        assert "Connection error" in str(exc1.value)

        with pytest.raises(LLMConnectionError) as exc2:
            asyncio.run(client.chat_completion("gpt-4o", [{"role": "user", "content": "hi"}]))
        assert "Stale event loop" not in str(exc2.value)
        assert "Connection error" in str(exc2.value)


class TestCalculateRetryDelay:
    @pytest.mark.parametrize(
        "retry_after,attempt,expected",
        [
            ("20", 0, 20.0),
            ("60", 0, 60.0),
            ("0", 0, None),
            ("-10", 0, None),
            ("61", 0, None),
            ("", 0, None),
            ("abc", 0, None),
            ("NaN", 0, None),
        ],
    )
    def test_retry_after_header_parsing(self, retry_after, attempt, expected):
        delay = BaseClient._calculate_retry_delay({"retry-after": retry_after}, attempt)
        if expected is not None:
            assert delay == expected
        else:
            upper = min(INITIAL_RETRY_DELAY * (2.0**attempt), MAX_RETRY_DELAY)
            assert 0 <= delay <= upper

    @pytest.mark.parametrize(
        "attempt,expected_base",
        [
            (0, INITIAL_RETRY_DELAY),
            (1, INITIAL_RETRY_DELAY * 2),
            (2, INITIAL_RETRY_DELAY * 4),
            (3, INITIAL_RETRY_DELAY * 8),
            (4, MAX_RETRY_DELAY),
            (10, MAX_RETRY_DELAY),
        ],
    )
    def test_exponential_backoff_with_jitter(self, attempt, expected_base):
        delay = BaseClient._calculate_retry_delay({}, attempt)
        upper = min(expected_base, MAX_RETRY_DELAY)
        assert 0 <= delay <= upper

    def test_delay_cap_at_max(self):
        delay = BaseClient._calculate_retry_delay({}, 100)
        assert 0 <= delay <= MAX_RETRY_DELAY

    def test_retry_after_above_cap_falls_back(self):
        delay = BaseClient._calculate_retry_delay({"retry-after": str(MAX_RETRY_AFTER + 1)}, 0)
        upper = INITIAL_RETRY_DELAY
        assert 0 <= delay <= upper

    def test_no_headers_defaults_to_exponential(self):
        delay = BaseClient._calculate_retry_delay({}, 0)
        assert 0 <= delay <= INITIAL_RETRY_DELAY

    def test_full_jitter_distribution_spans_full_range(self):
        attempt = 3
        cap = min(INITIAL_RETRY_DELAY * (2.0**attempt), MAX_RETRY_DELAY)
        delays = [BaseClient._calculate_retry_delay({}, attempt) for _ in range(500)]
        assert all(0 <= delay <= cap for delay in delays)
        below_quarter = sum(1 for delay in delays if delay < cap * 0.25)
        below_half = sum(1 for delay in delays if delay < cap * 0.5)
        assert below_quarter > 50
        assert below_half > 150


class TestShouldRetryCaseInsensitive:
    @pytest.mark.parametrize("value", ["false", "FALSE", "False", "fAlSe"])
    def test_false_value_recognized_case_insensitively(self, value):
        assert BaseClient._should_retry(500, {"x-should-retry": value}) is False

    @pytest.mark.parametrize("value", ["true", "TRUE", "True", "tRuE"])
    def test_true_value_recognized_case_insensitively(self, value):
        assert BaseClient._should_retry(400, {"x-should-retry": value}) is True

    def test_missing_header_falls_back_to_status_code(self):
        assert BaseClient._should_retry(500, {}) is True
        assert BaseClient._should_retry(400, {}) is False


class TestNonJsonResponseRaises:
    @pytest.mark.asyncio
    async def test_non_json_200_raises_validation_error(self):
        from nemoguardrails.exceptions import LLMResponseValidationError

        client = make_client()
        mock_httpx_post(
            client,
            [(200, "<html>captive portal</html>", {"content-type": "text/html"})],
        )
        with pytest.raises(LLMResponseValidationError, match="non-JSON"):
            await client.chat_completion("gpt-4o", [])

    @pytest.mark.asyncio
    async def test_malformed_sse_chunk_raises_validation_error(self):
        from nemoguardrails.exceptions import LLMResponseValidationError

        client = stream_client(
            [
                "data: {not valid json",
                "",
            ]
        )
        with pytest.raises(LLMResponseValidationError, match="Malformed SSE"):
            async for _ in client.stream_chat_completion("gpt-4o", [{"role": "user", "content": "Hi"}]):
                pass


class TestStreamOptionsHint:
    @pytest.mark.asyncio
    async def test_stream_options_400_message_includes_kwarg_hint(self):
        client = make_client()
        mock_httpx_post(
            client,
            [
                (
                    400,
                    {"error": {"message": "Unknown parameter: stream_options"}},
                    {},
                )
            ],
        )
        with pytest.raises(LLMUnsupportedParamsError) as exc_info:
            await client.chat_completion("gpt-4o", [])
        assert "include_usage_in_stream=False" in str(exc_info.value)


class TestStreamOptionsDefaultOn:
    @pytest.mark.asyncio
    async def test_include_usage_default_on_for_unknown_provider(self):
        client = OpenAICompatibleClient(base_url="https://my-self-hosted-endpoint.example/v1", api_key="x")
        assert client.provider_name is None
        captured = {}

        @asynccontextmanager
        async def capturing_stream(*args, **kwargs):
            captured.update(kwargs)

            class FakeResponse:
                status_code = 200
                headers = {}

                async def aread(self):
                    pass

                async def aiter_lines(self):
                    yield 'data: {"id":"c","choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":"stop"}]}'
                    yield ""
                    yield "data: [DONE]"
                    yield ""

            yield FakeResponse()

        client._client = type("MockClient", (), {"stream": capturing_stream})()
        async for _ in client.stream_chat_completion("custom-model", [{"role": "user", "content": "Hi"}]):
            pass

        assert captured["json"]["stream_options"] == {"include_usage": True}

    @pytest.mark.asyncio
    async def test_include_usage_opt_out_via_kwarg(self):
        client = make_client()
        captured = {}

        @asynccontextmanager
        async def capturing_stream(*args, **kwargs):
            captured.update(kwargs)

            class FakeResponse:
                status_code = 200
                headers = {}

                async def aread(self):
                    pass

                async def aiter_lines(self):
                    yield 'data: {"id":"c","choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":"stop"}]}'
                    yield ""
                    yield "data: [DONE]"
                    yield ""

            yield FakeResponse()

        client._client = type("MockClient", (), {"stream": capturing_stream})()
        async for _ in client.stream_chat_completion(
            "gpt-4o",
            [{"role": "user", "content": "Hi"}],
            include_usage_in_stream=False,
        ):
            pass

        assert "stream_options" not in captured["json"]


class TestDoneSentinel:
    @pytest.mark.asyncio
    async def test_done_terminates_stream(self):
        client = stream_client(
            [
                'data: {"id":"a","choices":[{"index":0,"delta":{"content":"hi"}}]}',
                "",
                "data: [DONE]",
                "",
            ]
        )
        chunks = await consume(client)
        assert len(chunks) == 1
        assert chunks[0].body["choices"][0]["delta"]["content"] == "hi"

    @pytest.mark.asyncio
    async def test_done_discards_subsequent_data(self):
        client = stream_client(
            [
                'data: {"id":"a","choices":[{"index":0,"delta":{"content":"first"}}]}',
                "",
                "data: [DONE]",
                "",
                'data: {"id":"b","choices":[{"index":0,"delta":{"content":"ignored"}}]}',
                "",
            ]
        )
        chunks = await consume(client)
        assert len(chunks) == 1
        assert chunks[0].body["choices"][0]["delta"]["content"] == "first"

    @pytest.mark.asyncio
    async def test_done_with_trailing_whitespace(self):
        client = stream_client(
            [
                'data: {"id":"a","choices":[{"index":0,"delta":{"content":"hi"}}]}',
                "",
                "data: [DONE] ",
                "",
            ]
        )
        chunks = await consume(client)
        assert len(chunks) == 1

    @pytest.mark.asyncio
    async def test_stream_without_done_completes_normally(self):
        client = stream_client(
            [
                'data: {"id":"a","choices":[{"index":0,"delta":{"content":"hi"}}]}',
                "",
                'data: {"id":"b","choices":[{"index":0,"delta":{"content":" there"}}]}',
                "",
            ]
        )
        chunks = await consume(client)
        assert len(chunks) == 2


class TestNoLeakOnRetry:
    @pytest.mark.asyncio
    async def test_stream_retry_consumes_body(self):
        stream, aread_calls = tracking_mock_stream(
            [
                (503, '{"error":{"message":"unavailable"}}', {}),
                (
                    200,
                    ['data: {"id":"a","choices":[{"index":0,"delta":{"content":"hi"}}]}', "", "data: [DONE]", ""],
                    {},
                ),
            ]
        )
        client = make_client()
        client._client = type("MockClient", (), {"stream": stream})()
        chunks = []
        async for chunk in client.stream_chat_completion("gpt-4o", [{"role": "user", "content": "Hi"}]):
            chunks.append(chunk)
        assert 503 in aread_calls
        assert len(chunks) == 1

    @pytest.mark.asyncio
    async def test_stream_error_consumes_body(self):
        stream, aread_calls = tracking_mock_stream([(500, '{"error":{"message":"server error"}}', {})])
        client = make_client(max_retries=0)
        client._client = type("MockClient", (), {"stream": stream})()
        with pytest.raises(LLMServerError):
            async for _ in client.stream_chat_completion("gpt-4o", [{"role": "user", "content": "Hi"}]):
                pass
        assert 500 in aread_calls


class TestClientErrorMetadata:
    """Client-level error contract: transport raises with base_url + model_name (from payload).
    provider_name is intentionally None at this layer — it's added by the model layer above.
    """

    @pytest.mark.asyncio
    async def test_http_error_has_base_url_only(self):
        client = make_client()
        mock_httpx_post(client, [(401, {"error": {"message": "Invalid key"}}, {})])
        with pytest.raises(LLMAuthenticationError) as exc_info:
            await client.chat_completion("gpt-4o", [])
        assert exc_info.value.base_url == "https://api.openai.com/v1"
        assert exc_info.value.model_name is None
        assert exc_info.value.provider_name is None

    @pytest.mark.asyncio
    async def test_mid_stream_error_has_base_url_only(self):
        client = stream_client(
            [
                'data: {"error": {"message": "err", "type": "rate_limit_error"}}',
                "",
            ]
        )
        with pytest.raises(LLMRateLimitError) as exc_info:
            await consume(client)
        assert exc_info.value.base_url == "https://api.openai.com/v1"
        assert exc_info.value.model_name is None
        assert exc_info.value.provider_name is None

    @pytest.mark.asyncio
    async def test_http_error_different_base_url_carries_through(self):
        client = OpenAICompatibleClient(base_url="https://integrate.api.nvidia.com/v1", api_key="nvapi-x")
        mock_httpx_post(client, [(401, {"error": {"message": "Invalid key"}}, {})])
        with pytest.raises(LLMAuthenticationError) as exc_info:
            await client.chat_completion("llama", [])
        assert exc_info.value.base_url == "https://integrate.api.nvidia.com/v1"
        assert exc_info.value.model_name is None
        assert exc_info.value.provider_name is None


class TestStreamEarlyAbort:
    """Output rails break out of the stream when content is blocked. The
    underlying response's async context manager must exit so the HTTP
    connection is released back to the pool.
    """

    @pytest.mark.asyncio
    async def test_aclose_after_first_chunk_releases_response(self):
        close_called = [False]

        @asynccontextmanager
        async def mock_stream(*args, **kwargs):
            class FakeResponse:
                status_code = 200
                headers = {}

                async def aread(self):
                    pass

                async def aiter_lines(self):
                    yield 'data: {"id":"1","choices":[{"index":0,"delta":{"content":"a"}}]}'
                    yield ""
                    yield 'data: {"id":"2","choices":[{"index":0,"delta":{"content":"b"}}]}'
                    yield ""
                    yield "data: [DONE]"
                    yield ""

            try:
                yield FakeResponse()
            finally:
                close_called[0] = True

        client = make_client()
        client._client = type("MockClient", (), {"stream": mock_stream})()

        gen = client.stream_chat_completion("gpt-4o", [{"role": "user", "content": "Hi"}])
        await gen.__anext__()
        await gen.aclose()
        assert close_called[0]

    @pytest.mark.asyncio
    async def test_consumer_raises_inside_aclosing_releases_response(self):
        from contextlib import aclosing

        close_called = [False]

        @asynccontextmanager
        async def mock_stream(*args, **kwargs):
            class FakeResponse:
                status_code = 200
                headers = {}

                async def aread(self):
                    pass

                async def aiter_lines(self):
                    yield 'data: {"id":"1","choices":[{"index":0,"delta":{"content":"a"}}]}'
                    yield ""
                    yield 'data: {"id":"2","choices":[{"index":0,"delta":{"content":"b"}}]}'
                    yield ""

            try:
                yield FakeResponse()
            finally:
                close_called[0] = True

        client = make_client()
        client._client = type("MockClient", (), {"stream": mock_stream})()

        class ConsumerError(Exception):
            pass

        with pytest.raises(ConsumerError):
            async with aclosing(client.stream_chat_completion("gpt-4o", [{"role": "user", "content": "Hi"}])) as gen:
                async for _ in gen:
                    raise ConsumerError("simulated consumer failure")
        assert close_called[0]

    @pytest.mark.asyncio
    async def test_many_aborted_streams_release_each_response(self):
        close_count = [0]

        @asynccontextmanager
        async def mock_stream(*args, **kwargs):
            class FakeResponse:
                status_code = 200
                headers = {}

                async def aread(self):
                    pass

                async def aiter_lines(self):
                    for i in range(100):
                        yield f'data: {{"id":"{i}","choices":[{{"index":0,"delta":{{"content":"x"}}}}]}}'
                        yield ""
                    yield "data: [DONE]"
                    yield ""

            try:
                yield FakeResponse()
            finally:
                close_count[0] += 1

        client = make_client()
        client._client = type("MockClient", (), {"stream": mock_stream})()

        for _ in range(50):
            gen = client.stream_chat_completion("gpt-4o", [{"role": "user", "content": "Hi"}])
            await gen.__anext__()
            await gen.aclose()

        assert close_count[0] == 50


@mock.patch(
    "nemoguardrails.llm.clients.base.BaseClient._calculate_retry_delay",
    staticmethod(low_retry_delay),
)
class TestConcurrency:
    """Guardrails patterns multiple LLM calls (input rails + main + output rails)
    through one pooled client. These tests cover shared-state safety, pool
    fairness, and retry isolation under concurrency.
    """

    @pytest.mark.asyncio
    async def test_concurrent_chat_completions_independent(self):
        client = make_client()

        async def echoing_post(*args, **kwargs):
            prompt = kwargs["json"]["messages"][0]["content"]
            await asyncio.sleep(0)
            return httpx.Response(
                200,
                json={
                    "id": f"chatcmpl-{prompt}",
                    "model": "gpt-4o",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"content": f"reply-{prompt}", "role": "assistant"},
                            "finish_reason": "stop",
                        }
                    ],
                },
                request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
            )

        client._client = type("MockClient", (), {"post": echoing_post})()

        prompts = [f"msg-{i}" for i in range(10)]
        results = await asyncio.gather(
            *[client.chat_completion("gpt-4o", [{"role": "user", "content": p}]) for p in prompts]
        )

        for prompt, result in zip(prompts, results):
            assert result.body["id"] == f"chatcmpl-{prompt}"
            assert result.body["choices"][0]["message"]["content"] == f"reply-{prompt}"

    @pytest.mark.asyncio
    async def test_concurrent_streams_independent(self):
        @asynccontextmanager
        async def echoing_stream(*args, **kwargs):
            prompt = kwargs["json"]["messages"][0]["content"]

            class FakeResponse:
                status_code = 200
                headers = {}

                async def aread(self):
                    pass

                async def aiter_lines(self):
                    yield f'data: {{"id":"c-{prompt}","choices":[{{"index":0,"delta":{{"content":"{prompt}-chunk"}}}}]}}'
                    yield ""
                    await asyncio.sleep(0)
                    yield "data: [DONE]"
                    yield ""

            yield FakeResponse()

        client = make_client()
        client._client = type("MockClient", (), {"stream": echoing_stream})()

        async def consume_stream(prompt):
            chunks = []
            async for chunk in client.stream_chat_completion("gpt-4o", [{"role": "user", "content": prompt}]):
                chunks.append(chunk)
            return chunks

        prompts = [f"p{i}" for i in range(5)]
        results = await asyncio.gather(*[consume_stream(p) for p in prompts])

        for prompt, chunks in zip(prompts, results):
            assert len(chunks) == 1
            assert chunks[0].body["id"] == f"c-{prompt}"
            assert chunks[0].body["choices"][0]["delta"]["content"] == f"{prompt}-chunk"

    @pytest.mark.asyncio
    async def test_429_on_one_call_does_not_block_peers(self):
        client = make_client()
        seen_calls = {}
        lock = asyncio.Lock()

        async def sometimes_429(*args, **kwargs):
            prompt = kwargs["json"]["messages"][0]["content"]
            async with lock:
                seen_calls[prompt] = seen_calls.get(prompt, 0) + 1
                current = seen_calls[prompt]
            if prompt == "slow" and current == 1:
                return httpx.Response(
                    429,
                    json={"error": {"message": "rate limit"}},
                    headers={"retry-after": "0"},
                    request=httpx.Request("POST", "url"),
                )
            return httpx.Response(
                200,
                json={
                    "id": f"c-{prompt}",
                    "model": "gpt-4o",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"content": f"ok-{prompt}", "role": "assistant"},
                            "finish_reason": "stop",
                        }
                    ],
                },
                request=httpx.Request("POST", "url"),
            )

        client._client = type("MockClient", (), {"post": sometimes_429})()

        prompts = ["slow", "fast-1", "fast-2", "fast-3", "fast-4"]
        results = await asyncio.gather(
            *[client.chat_completion("gpt-4o", [{"role": "user", "content": p}]) for p in prompts]
        )

        for prompt, result in zip(prompts, results):
            assert result.body["id"] == f"c-{prompt}"
            assert result.body["choices"][0]["message"]["content"] == f"ok-{prompt}"

        assert seen_calls["slow"] == 2
        for p in prompts[1:]:
            assert seen_calls[p] == 1
