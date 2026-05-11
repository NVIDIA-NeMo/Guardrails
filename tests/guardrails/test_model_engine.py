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

"""Unit tests for model_engine module."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from nemoguardrails.guardrails._http import (
    DEFAULT_MAX_ATTEMPTS,
    DEFAULT_TIMEOUT_CONNECT,
    DEFAULT_TIMEOUT_TOTAL,
)
from nemoguardrails.guardrails.model_engine import (
    _CHAT_COMPLETIONS_ENDPOINT,
    _ENGINE_BASE_URLS,
    ModelEngine,
    ModelEngineError,
    _parse_chat_completion,
    _parse_chat_completion_chunk,
)
from nemoguardrails.rails.llm.config import Model
from nemoguardrails.types import LLMResponse, LLMResponseChunk, UsageInfo


def _make_model(
    model_type: str = "main",
    engine: str = "nim",
    model: str | None = "meta/llama-3.3-70b-instruct",
    api_key_env_var: str | None = None,
    parameters: dict | None = None,
) -> Model:
    """Create a Model config for testing."""
    return Model(
        type=model_type,
        engine=engine,
        model=model,
        api_key_env_var=api_key_env_var,
        parameters=parameters or {},
    )


class TestModelEngineError:
    """Test the ModelEngineError Exception type fields."""

    def test_basic_error(self):
        """Error stores message and model_name, status defaults to None."""
        err = ModelEngineError("something broke", model_name="my-model")
        assert str(err) == "something broke"
        assert err.model_name == "my-model"
        assert err.status is None

    def test_error_with_status(self):
        """Error stores the HTTP status code when provided."""
        err = ModelEngineError("bad request", model_name="my-model", status=400)
        assert err.status == 400
        assert err.model_name == "my-model"

    def test_is_exception(self):
        """ModelEngineError is a subclass of Exception."""
        assert issubclass(ModelEngineError, Exception)


class TestModelEngineBaseUrl:
    """Test base URL resolution from engine type and parameters."""

    @patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"})
    def test_nim_engine_uses_nvidia_url(self):
        """NIM engine resolves to the NVIDIA integrate URL."""
        engine = ModelEngine(_make_model(engine="nim"))
        assert engine.base_url == _ENGINE_BASE_URLS["nim"]

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    def test_openai_engine_uses_openai_url(self):
        """OpenAI engine resolves to the OpenAI API URL."""
        engine = ModelEngine(_make_model(engine="openai"))
        assert engine.base_url == _ENGINE_BASE_URLS["openai"]

    @patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"})
    def test_explicit_base_url_overrides_engine_default(self):
        """A base_url in parameters takes priority over engine default."""
        engine = ModelEngine(_make_model(engine="nim", parameters={"base_url": "https://custom.example.com"}))
        assert engine.base_url == "https://custom.example.com"

    def test_unknown_engine_without_base_url_raises(self):
        """Unknown engine with no base_url raises ValueError."""
        with pytest.raises(ValueError, match="cannot infer from engine"):
            ModelEngine(_make_model(engine="unknown"))

    @patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"})
    @pytest.mark.asyncio
    async def test_base_url_with_trailing_v1_does_not_double_v1(self):
        """A user-supplied base_url ending in /v1 must not produce /v1/v1/chat/completions."""
        engine = ModelEngine(_make_model(engine="nim", parameters={"base_url": "https://custom.example.com/v1"}))

        mock_response = AsyncMock()
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"choices": [{"message": {"content": "ok"}}]})

        mock_client = AsyncMock()
        mock_client.post = MagicMock(return_value=mock_response)
        mock_client.closed = False
        engine._client = mock_client
        engine._running = True

        await engine.call([{"role": "user", "content": "Hi"}])

        url = mock_client.post.call_args[0][0]
        assert url == "https://custom.example.com/v1/chat/completions"
        assert "/v1/v1/" not in url

    @pytest.mark.parametrize(
        "base_url_input,expected",
        [
            ("https://host.example.com", "https://host.example.com"),
            ("https://host.example.com/", "https://host.example.com"),
            ("https://host.example.com/v1", "https://host.example.com"),
            ("https://host.example.com/v1/", "https://host.example.com"),
            ("https://api-v1.example.com", "https://api-v1.example.com"),
            ("https://api-v1.example.com/v1", "https://api-v1.example.com"),
            ("https://host.example.com/api/v1", "https://host.example.com/api"),
        ],
    )
    @patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"})
    def test_resolve_base_url_normalization_https(self, base_url_input, expected):
        """_resolve_base_url strips trailing slash + trailing /v1 path segment only.

        Hostnames containing 'v1' (e.g. api-v1.example.com) must not be mangled.
        """
        engine = ModelEngine(_make_model(engine="nim", parameters={"base_url": base_url_input}))
        assert engine.base_url == expected

    @pytest.mark.parametrize(
        "base_url_input,expected",
        [
            ("http://localhost:8000", "http://localhost:8000"),
            ("http://localhost:8000/v1", "http://localhost:8000"),
            ("http://localhost:11434/v1/", "http://localhost:11434"),
            ("http://127.0.0.1:8000/v1", "http://127.0.0.1:8000"),
        ],
    )
    @patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"})
    def test_resolve_base_url_normalization_http(self, base_url_input, expected):
        """Same normalization for plain-http base_urls (common for local models: vLLM, Ollama)."""
        engine = ModelEngine(_make_model(engine="nim", parameters={"base_url": base_url_input}))
        assert engine.base_url == expected

    @pytest.mark.parametrize(
        "base_url_input,expected_url",
        [
            ("https://host.example.com", "https://host.example.com/v1/chat/completions"),
            ("https://host.example.com/", "https://host.example.com/v1/chat/completions"),
            ("https://host.example.com/v1/", "https://host.example.com/v1/chat/completions"),
            ("https://api-v1.example.com", "https://api-v1.example.com/v1/chat/completions"),
        ],
    )
    @patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"})
    @pytest.mark.asyncio
    async def test_call_url_for_accepted_base_url_shapes_https(self, base_url_input, expected_url):
        """End-to-end: call() POSTs to a canonical /v1/chat/completions URL across accepted https base_url shapes."""
        engine = ModelEngine(_make_model(engine="nim", parameters={"base_url": base_url_input}))

        mock_response = AsyncMock()
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"choices": [{"message": {"content": "ok"}}]})

        mock_client = AsyncMock()
        mock_client.post = MagicMock(return_value=mock_response)
        mock_client.closed = False
        engine._client = mock_client
        engine._running = True

        await engine.call([{"role": "user", "content": "Hi"}])

        url = mock_client.post.call_args[0][0]
        assert url == expected_url
        assert "/v1/v1/" not in url

    @pytest.mark.parametrize(
        "base_url_input,expected_url",
        [
            ("http://localhost:8000", "http://localhost:8000/v1/chat/completions"),
            ("http://localhost:8000/v1", "http://localhost:8000/v1/chat/completions"),
            ("http://127.0.0.1:11434/v1/", "http://127.0.0.1:11434/v1/chat/completions"),
        ],
    )
    @patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"})
    @pytest.mark.asyncio
    async def test_call_url_for_accepted_base_url_shapes_http(self, base_url_input, expected_url):
        """End-to-end: same URL composition for plain-http local-model base_urls."""
        engine = ModelEngine(_make_model(engine="nim", parameters={"base_url": base_url_input}))

        mock_response = AsyncMock()
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"choices": [{"message": {"content": "ok"}}]})

        mock_client = AsyncMock()
        mock_client.post = MagicMock(return_value=mock_response)
        mock_client.closed = False
        engine._client = mock_client
        engine._running = True

        await engine.call([{"role": "user", "content": "Hi"}])

        url = mock_client.post.call_args[0][0]
        assert url == expected_url
        assert "/v1/v1/" not in url


class TestModelEngineApiKey:
    """Test API key resolution from environment variables."""

    @patch.dict("os.environ", {"NVIDIA_API_KEY": "nvidia-key-123"})
    def test_nim_engine_reads_nvidia_api_key(self):
        """NIM engine reads NVIDIA_API_KEY from environment."""
        engine = ModelEngine(_make_model(engine="nim"))
        assert engine.api_key == "nvidia-key-123"

    @patch.dict("os.environ", {"OPENAI_API_KEY": "openai-key-456"})
    def test_openai_engine_reads_openai_api_key(self):
        """OpenAI engine reads OPENAI_API_KEY from environment."""
        engine = ModelEngine(_make_model(engine="openai"))
        assert engine.api_key == "openai-key-456"

    @patch.dict("os.environ", {"MY_CUSTOM_KEY": "custom-key-789"})
    def test_api_key_env_var_overrides_engine_default(self):
        """api_key_env_var in model config takes priority over engine default."""
        engine = ModelEngine(_make_model(engine="nim", api_key_env_var="MY_CUSTOM_KEY"))
        assert engine.api_key == "custom-key-789"

    @patch.dict("os.environ", {}, clear=True)
    def test_missing_env_var_raises(self):
        """Missing NVIDIA_API_KEY with nim engine stores api key as None"""
        engine = ModelEngine(_make_model(engine="nim"))
        assert engine.api_key is None

    @patch.dict("os.environ", {}, clear=True)
    def test_custom_env_var_missing_raises(self):
        """Missing custom env var raises RuntimeError naming the variable."""
        with pytest.raises(RuntimeError, match="Environment variable 'DOES_NOT_EXIST' not set"):
            ModelEngine(_make_model(engine="nim", api_key_env_var="DOES_NOT_EXIST"))

    @patch.dict("os.environ", {}, clear=True)
    def test_unknown_engine_no_base_url_raises_value_error(self):
        """Unknown engine without base_url fails at URL resolution first."""
        with pytest.raises(ValueError, match="cannot infer from engine"):
            ModelEngine(_make_model(engine="unknown", parameters={}))

    @patch.dict("os.environ", {}, clear=True)
    def test_unknown_engine_with_base_url_raises_runtime_error_for_api_key(self):
        """Unknown engine with base_url passes URL resolution but fails API key resolution."""
        model_engine = ModelEngine(_make_model(engine="custom", parameters={"base_url": "https://custom.example.com"}))
        assert model_engine.api_key is None


class TestModelEngineConfig:
    """Test default and custom timeout, retry, and model name configuration."""

    @patch.dict("os.environ", {"NVIDIA_API_KEY": "key"})
    def test_default_timeout_values(self):
        """Timeout defaults match module constants when no parameters given."""
        engine = ModelEngine(_make_model())
        assert engine._timeout.total == DEFAULT_TIMEOUT_TOTAL
        assert engine._timeout.connect == DEFAULT_TIMEOUT_CONNECT

    @patch.dict("os.environ", {"NVIDIA_API_KEY": "key"})
    def test_custom_timeout_from_parameters(self):
        """Timeout values can be overridden via model parameters."""
        engine = ModelEngine(_make_model(parameters={"timeout": 120, "timeout_connect": 30}))
        assert engine._timeout.total == 120.0
        assert engine._timeout.connect == 30.0

    @patch.dict("os.environ", {"NVIDIA_API_KEY": "key"})
    def test_custom_max_attempts_from_parameters(self):
        """Max attempts can be overridden via model parameters."""
        engine = ModelEngine(_make_model(parameters={"max_attempts": 5}))
        assert engine._retry_options.attempts == 5

    @patch.dict("os.environ", {"NVIDIA_API_KEY": "key"})
    def test_default_max_attempts(self):
        """Max attempts defaults to module constant when not specified."""
        engine = ModelEngine(_make_model())
        assert engine._retry_options.attempts == DEFAULT_MAX_ATTEMPTS

    @patch.dict("os.environ", {"NVIDIA_API_KEY": "key"})
    def test_model_name_set(self):
        """model_name is taken from the Model config's model field."""
        engine = ModelEngine(_make_model(model="my-model"))
        assert engine.model_name == "my-model"

    @patch.dict("os.environ", {"NVIDIA_API_KEY": "key"})
    def test_model_name_from_parameters(self):
        """model_name falls back to parameters.model_name when model is None."""
        engine = ModelEngine(_make_model(model=None, parameters={"model_name": "param-model"}))
        assert engine.model_name == "param-model"

    @patch.dict("os.environ", {"NVIDIA_API_KEY": "key"})
    def test_null_timeout_falls_back_to_default(self):
        """Explicit None for timeout/timeout_connect/max_attempts uses defaults."""
        engine = ModelEngine(_make_model(parameters={"timeout": None, "timeout_connect": None, "max_attempts": None}))
        assert engine._timeout.total == DEFAULT_TIMEOUT_TOTAL
        assert engine._timeout.connect == DEFAULT_TIMEOUT_CONNECT
        assert engine._retry_options.attempts == DEFAULT_MAX_ATTEMPTS

    @patch.dict("os.environ", {"NVIDIA_API_KEY": "key"})
    def test_client_initially_none(self):
        """RetryClient is not created until start() is called."""
        engine = ModelEngine(_make_model())
        assert engine._client is None


class TestModelEngineLifecycle:
    """Test the ModelEngine start() and stop() client lifecycle."""

    @patch.dict("os.environ", {"NVIDIA_API_KEY": "key"})
    @pytest.mark.asyncio
    async def test_start_stop_lifecycle(self):
        """start() creates the client, stop() tears it down to None."""
        engine = ModelEngine(_make_model())
        assert engine._client is None
        assert engine._running is False
        await engine.start()
        assert engine._client is not None
        assert engine._running is True
        await engine.stop()
        assert engine._client is None
        assert engine._running is False

    @patch.dict("os.environ", {"NVIDIA_API_KEY": "key"})
    @pytest.mark.asyncio
    async def test_start_is_idempotent(self):
        """Calling start() twice reuses the same client instance."""
        engine = ModelEngine(_make_model())
        await engine.start()
        first_client = engine._client
        await engine.start()  # should not create a new client
        assert engine._client is first_client
        await engine.stop()

    @patch.dict("os.environ", {"NVIDIA_API_KEY": "key"})
    @pytest.mark.asyncio
    async def test_stop_when_no_client_is_noop(self):
        """stop() without a prior start() does not raise."""
        engine = ModelEngine(_make_model())
        await engine.stop()  # should not raise
        assert engine._running is False

    @patch.dict("os.environ", {"NVIDIA_API_KEY": "key"})
    @pytest.mark.asyncio
    async def test_stop_is_idempotent(self):
        """Calling stop() twice does not raise."""
        engine = ModelEngine(_make_model())
        await engine.start()
        await engine.stop()
        await engine.stop()  # second stop is a no-op
        assert engine._running is False


class TestModelEngineConcurrentLifecycle:
    """Test that the asyncio.Lock in BaseEngine protects stop() from races.

    start() has no await in its critical section so it's effectively atomic
    in asyncio's cooperative model. stop() has `await client.close()` which
    creates a real interleaving window — the lock prevents double-close.
    """

    @patch.dict("os.environ", {"NVIDIA_API_KEY": "key"})
    @pytest.mark.asyncio
    async def test_concurrent_stop_closes_client_once(self):
        """Two concurrent stop() calls only close the client once."""
        engine = ModelEngine(_make_model())
        await engine.start()

        close_mock = AsyncMock()
        engine._client.close = close_mock

        await asyncio.gather(engine.stop(), engine.stop())

        assert not engine._running
        assert engine._client is None
        close_mock.assert_awaited_once()

    @patch.dict("os.environ", {"NVIDIA_API_KEY": "key"})
    @pytest.mark.asyncio
    async def test_concurrent_start_stop_does_not_leak(self):
        """Concurrent start() and stop() leave the engine in a consistent state."""
        engine = ModelEngine(_make_model())
        await engine.start()
        assert engine._running

        await asyncio.gather(engine.stop(), engine.start())

        # Engine should be in a consistent state — clean up if still running
        assert (engine._running and engine._client is not None) or (not engine._running and engine._client is None)
        if engine._running:
            await engine.stop()


class TestModelEngineContextManager:
    """Test async context manager calls start/stop correctly."""

    @patch.dict("os.environ", {"NVIDIA_API_KEY": "key"})
    @pytest.mark.asyncio
    async def test_context_manager_calls_start_and_stop(self):
        """async with calls start() on enter and stop() on exit."""
        engine = ModelEngine(_make_model())
        assert engine._running is False
        async with engine as eng:
            assert eng is engine
            assert engine._running is True
            assert engine._client is not None
        assert engine._running is False
        assert engine._client is None


class TestModelEngineCall:
    """Test ModelEngine.call() HTTP request construction and error handling."""

    @patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"})
    @pytest.mark.asyncio
    async def test_successful_call(self):
        """Successful call returns parsed JSON and posts to correct URL with headers."""
        model = _make_model()
        engine = ModelEngine(model)

        expected_response = {"choices": [{"message": {"role": "assistant", "content": "Hello!"}}]}

        mock_response = AsyncMock()
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=expected_response)

        mock_client = AsyncMock()
        mock_client.post = MagicMock(return_value=mock_response)
        mock_client.closed = False

        engine._client = mock_client
        engine._running = True

        messages = [{"role": "user", "content": "Hi"}]
        result = await engine.call(messages)
        assert result == expected_response

        # Verify correct URL
        call_args = mock_client.post.call_args
        assert _CHAT_COMPLETIONS_ENDPOINT in call_args[0][0]

        expected_url = _ENGINE_BASE_URLS[model.engine] + "/v1/chat/completions"
        expected_json = {"messages": messages, "model": model.model}
        expected_headers = {"Content-Type": "application/json", "Authorization": "Bearer test-key"}
        mock_client.post.assert_called_once_with(expected_url, json=expected_json, headers=expected_headers)

    @patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"})
    @pytest.mark.asyncio
    async def test_call_includes_model_name_and_messages_in_body(self):
        """Request body contains model name, messages, and extra kwargs."""
        engine = ModelEngine(_make_model(model="my-llm"))

        mock_response = AsyncMock()
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"choices": [{"message": {"content": "ok"}}]})

        mock_client = AsyncMock()
        mock_client.post = MagicMock(return_value=mock_response)
        mock_client.closed = False
        engine._client = mock_client
        engine._running = True

        messages = [{"role": "user", "content": "Hello"}]
        await engine.call(messages, temperature=0.7)

        call_kwargs = mock_client.post.call_args
        body = call_kwargs[1]["json"]
        assert body["model"] == "my-llm"
        assert body["messages"] == messages
        assert body["temperature"] == 0.7

    @patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"})
    @pytest.mark.asyncio
    async def test_call_without_api_key_omits_auth_header(self):
        """No Authorization header when api_key is None."""
        engine = ModelEngine(_make_model())
        engine.api_key = None  # simulate no API key

        mock_response = AsyncMock()
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"choices": [{"message": {"content": "ok"}}]})

        mock_client = AsyncMock()
        mock_client.post = MagicMock(return_value=mock_response)
        mock_client.closed = False
        engine._client = mock_client
        engine._running = True

        await engine.call([{"role": "user", "content": "Hi"}])

        call_kwargs = mock_client.post.call_args
        headers = call_kwargs[1]["headers"]
        assert "Authorization" not in headers

    @patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"})
    @pytest.mark.asyncio
    async def test_call_http_error_raises_model_engine_error(self):
        """HTTP 4xx/5xx raises ModelEngineError with status and model name."""
        engine = ModelEngine(_make_model())

        mock_response = AsyncMock()
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.status = 400
        mock_response.text = AsyncMock(return_value='{"error": "bad request"}')

        mock_client = AsyncMock()
        mock_client.post = MagicMock(return_value=mock_response)
        mock_client.closed = False
        engine._client = mock_client
        engine._running = True

        with pytest.raises(ModelEngineError) as exc_info:
            await engine.call([{"role": "user", "content": "Hi"}])

        assert exc_info.value.status == 400
        assert exc_info.value.model_name == "meta/llama-3.3-70b-instruct"

    @patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"})
    @pytest.mark.asyncio
    async def test_call_unexpected_exception_wraps_in_model_engine_error(self):
        """Non-HTTP exceptions are wrapped in ModelEngineError."""
        engine = ModelEngine(_make_model())

        mock_client = AsyncMock()
        mock_client.post = MagicMock(side_effect=RuntimeError("connection dropped"))
        mock_client.closed = False
        engine._client = mock_client
        engine._running = True

        with pytest.raises(ModelEngineError, match="connection dropped"):
            await engine.call([{"role": "user", "content": "Hi"}])

    @patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"})
    @pytest.mark.asyncio
    async def test_call_raises_if_not_started(self):
        """call() raises ModelEngineError if start() hasn't been called."""
        engine = ModelEngine(_make_model())
        assert engine._client is None

        with pytest.raises(ModelEngineError, match="has not been started"):
            await engine.call([{"role": "user", "content": "Hi"}])


class TestModelEngineStreamCall:
    """Test ModelEngine.stream_call() SSE streaming."""

    @staticmethod
    def _make_sse_content(chunks):
        """Build raw SSE byte lines from a list of content strings."""
        lines = []
        for text in chunks:
            payload = json.dumps({"choices": [{"delta": {"content": text}}]})
            lines.append(f"data: {payload}\n\n".encode())
        lines.append(b"data: [DONE]\n\n")
        return lines

    @staticmethod
    def _mock_streaming_response(raw_lines, status=200):
        """Create a mock aiohttp response with a readline()-based content mock.

        Splits each raw_line on ``\\n`` boundaries so that readline() returns
        one line at a time, matching real aiohttp StreamReader behaviour.
        """
        # Flatten raw_lines into individual \n-terminated lines
        all_lines = []
        for raw in raw_lines:
            for part in raw.split(b"\n"):
                if part:
                    all_lines.append(part + b"\n")

        line_iter = iter(all_lines)

        async def _readline():
            return next(line_iter, b"")

        mock_content = MagicMock()
        mock_content.readline = _readline

        mock_response = AsyncMock()
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.status = status
        mock_response.content = mock_content
        return mock_response

    @patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"})
    @pytest.mark.asyncio
    async def test_stream_call_yields_content_chunks(self):
        """stream_call() yields LLMResponseChunk objects with delta_content set."""
        engine = ModelEngine(_make_model())

        raw_lines = self._make_sse_content(["Hello", " world", "!"])
        mock_response = self._mock_streaming_response(raw_lines)

        mock_client = AsyncMock()
        mock_client.post = MagicMock(return_value=mock_response)
        engine._client = mock_client
        engine._running = True

        chunks = []
        async for chunk in engine.stream_call([{"role": "user", "content": "Hi"}]):
            chunks.append(chunk)

        assert all(isinstance(c, LLMResponseChunk) for c in chunks)
        assert [c.delta_content for c in chunks] == ["Hello", " world", "!"]
        assert all(c.delta_reasoning is None for c in chunks)

    @patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"})
    @pytest.mark.asyncio
    async def test_stream_call_yields_reasoning_deltas(self):
        """stream_call() surfaces delta.reasoning_content as LLMResponseChunk.delta_reasoning."""
        engine = ModelEngine(_make_model())

        raw_lines = [
            b'data: {"choices": [{"delta": {"reasoning_content": "let me think"}}]}\n\n',
            b'data: {"choices": [{"delta": {"content": "Hello"}}]}\n\n',
            b'data: {"choices": [{"delta": {"reasoning_content": " more"}}]}\n\n',
            b"data: [DONE]\n\n",
        ]
        mock_response = self._mock_streaming_response(raw_lines)

        mock_client = AsyncMock()
        mock_client.post = MagicMock(return_value=mock_response)
        engine._client = mock_client
        engine._running = True

        chunks = []
        async for chunk in engine.stream_call([{"role": "user", "content": "Hi"}]):
            chunks.append(chunk)

        assert [(c.delta_content, c.delta_reasoning) for c in chunks] == [
            (None, "let me think"),
            ("Hello", None),
            (None, " more"),
        ]

    @patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"})
    @pytest.mark.asyncio
    async def test_stream_call_yields_combined_content_and_reasoning_in_one_chunk(self):
        """A single SSE delta with both content and reasoning_content populates both fields."""
        engine = ModelEngine(_make_model())

        raw_lines = [
            b'data: {"choices": [{"delta": {"content": "answer", "reasoning_content": "thought"}}]}\n\n',
            b"data: [DONE]\n\n",
        ]
        mock_response = self._mock_streaming_response(raw_lines)

        mock_client = AsyncMock()
        mock_client.post = MagicMock(return_value=mock_response)
        engine._client = mock_client
        engine._running = True

        chunks = []
        async for chunk in engine.stream_call([{"role": "user", "content": "Hi"}]):
            chunks.append(chunk)

        assert len(chunks) == 1
        assert chunks[0].delta_content == "answer"
        assert chunks[0].delta_reasoning == "thought"

    @patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"})
    @pytest.mark.asyncio
    async def test_stream_call_sends_stream_true(self):
        """stream_call() includes stream=True in the request body."""
        engine = ModelEngine(_make_model())

        raw_lines = self._make_sse_content(["ok"])
        mock_response = self._mock_streaming_response(raw_lines)

        mock_client = AsyncMock()
        mock_client.post = MagicMock(return_value=mock_response)
        engine._client = mock_client
        engine._running = True

        async for _ in engine.stream_call([{"role": "user", "content": "Hi"}]):
            pass

        body = mock_client.post.call_args[1]["json"]
        assert body["stream"] is True

    @patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"})
    @pytest.mark.asyncio
    async def test_stream_call_forwards_kwargs(self):
        """Extra kwargs (temperature, etc.) are included in the request body."""
        engine = ModelEngine(_make_model())

        raw_lines = self._make_sse_content(["ok"])
        mock_response = self._mock_streaming_response(raw_lines)

        mock_client = AsyncMock()
        mock_client.post = MagicMock(return_value=mock_response)
        engine._client = mock_client
        engine._running = True

        async for _ in engine.stream_call([{"role": "user", "content": "Hi"}], temperature=0.5):
            pass

        body = mock_client.post.call_args[1]["json"]
        assert body["temperature"] == 0.5

    @patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"})
    @pytest.mark.asyncio
    async def test_stream_call_http_error(self):
        """stream_call() raises ModelEngineError on HTTP 4xx/5xx."""
        engine = ModelEngine(_make_model())

        mock_response = AsyncMock()
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.status = 500
        mock_response.text = AsyncMock(return_value="internal error")

        mock_client = AsyncMock()
        mock_client.post = MagicMock(return_value=mock_response)
        engine._client = mock_client
        engine._running = True

        with pytest.raises(ModelEngineError) as exc_info:
            await anext(engine.stream_call([{"role": "user", "content": "Hi"}]))

        assert exc_info.value.status == 500

    @patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"})
    @pytest.mark.asyncio
    async def test_stream_call_raises_if_not_started(self):
        """stream_call() raises ModelEngineError if start() hasn't been called."""
        engine = ModelEngine(_make_model())

        with pytest.raises(ModelEngineError, match="has not been started"):
            await anext(engine.stream_call([{"role": "user", "content": "Hi"}]))

    @patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"})
    @pytest.mark.asyncio
    async def test_stream_call_uses_streaming_timeout(self):
        """stream_call() overrides total timeout to None and sets sock_read."""
        engine = ModelEngine(_make_model())

        raw_lines = self._make_sse_content(["ok"])
        mock_response = self._mock_streaming_response(raw_lines)

        mock_client = AsyncMock()
        mock_client.post = MagicMock(return_value=mock_response)
        engine._client = mock_client
        engine._running = True

        async for _ in engine.stream_call([{"role": "user", "content": "Hi"}]):
            pass

        call_kwargs = mock_client.post.call_args[1]
        timeout = call_kwargs["timeout"]
        assert isinstance(timeout, aiohttp.ClientTimeout)
        assert timeout.total is None
        assert timeout.connect == engine._timeout.connect
        assert timeout.sock_read == engine._timeout.total

    @patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"})
    @pytest.mark.asyncio
    async def test_stream_call_skips_empty_content(self):
        """Chunks where delta has no 'content' key are skipped."""
        engine = ModelEngine(_make_model())

        # Include a chunk with role-only delta (no content) — typical for first SSE event
        raw_lines = [
            b'data: {"choices": [{"delta": {"role": "assistant"}}]}\n\n',
            b'data: {"choices": [{"delta": {"content": "Hello"}}]}\n\n',
            b"data: [DONE]\n\n",
        ]
        mock_response = self._mock_streaming_response(raw_lines)

        mock_client = AsyncMock()
        mock_client.post = MagicMock(return_value=mock_response)
        engine._client = mock_client
        engine._running = True

        chunks = []
        async for chunk in engine.stream_call([{"role": "user", "content": "Hi"}]):
            chunks.append(chunk)

        assert [c.delta_content for c in chunks] == ["Hello"]

    @patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"})
    @pytest.mark.asyncio
    async def test_stream_call_skips_empty_and_non_data_lines(self):
        """Empty lines and non-'data:' lines (e.g. comments, event types) are skipped."""
        engine = ModelEngine(_make_model())

        raw_lines = [
            b"\n",  # empty line
            b": keepalive\n",  # SSE comment
            b"event: ping\n",  # non-data event line
            b'data: {"choices": [{"delta": {"content": "ok"}}]}\n\n',
            b"data: [DONE]\n\n",
        ]
        mock_response = self._mock_streaming_response(raw_lines)

        mock_client = AsyncMock()
        mock_client.post = MagicMock(return_value=mock_response)
        engine._client = mock_client
        engine._running = True

        chunks = []
        async for chunk in engine.stream_call([{"role": "user", "content": "Hi"}]):
            chunks.append(chunk)

        assert [c.delta_content for c in chunks] == ["ok"]

    @patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"})
    @pytest.mark.asyncio
    async def test_stream_call_skips_unparseable_json(self):
        """Malformed JSON in an SSE data line is logged and skipped."""
        engine = ModelEngine(_make_model())

        raw_lines = [
            b"data: {not valid json}\n\n",
            b'data: {"choices": [{"delta": {"content": "ok"}}]}\n\n',
            b"data: [DONE]\n\n",
        ]
        mock_response = self._mock_streaming_response(raw_lines)

        mock_client = AsyncMock()
        mock_client.post = MagicMock(return_value=mock_response)
        engine._client = mock_client
        engine._running = True

        chunks = []
        async for chunk in engine.stream_call([{"role": "user", "content": "Hi"}]):
            chunks.append(chunk)

        assert [c.delta_content for c in chunks] == ["ok"]

    @patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"})
    @pytest.mark.asyncio
    async def test_stream_call_unexpected_exception_wraps_in_model_engine_error(self):
        """Non-HTTP exceptions during streaming are wrapped in ModelEngineError."""
        engine = ModelEngine(_make_model())

        mock_client = AsyncMock()
        mock_client.post = MagicMock(side_effect=RuntimeError("connection dropped"))
        engine._client = mock_client
        engine._running = True

        with pytest.raises(ModelEngineError, match="connection dropped"):
            await anext(engine.stream_call([{"role": "user", "content": "Hi"}]))

    @patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"})
    @pytest.mark.asyncio
    async def test_stream_call_skips_empty_choices(self):
        """SSE events with choices: [] (e.g. include_usage) are skipped without IndexError."""
        engine = ModelEngine(_make_model())

        raw_lines = [
            b'data: {"choices": []}\n\n',
            b'data: {"choices": [{"delta": {"content": "ok"}}]}\n\n',
            b"data: [DONE]\n\n",
        ]
        mock_response = self._mock_streaming_response(raw_lines)

        mock_client = AsyncMock()
        mock_client.post = MagicMock(return_value=mock_response)
        engine._client = mock_client
        engine._running = True

        chunks = []
        async for chunk in engine.stream_call([{"role": "user", "content": "Hi"}]):
            chunks.append(chunk)

        assert [c.delta_content for c in chunks] == ["ok"]

    @patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"})
    @pytest.mark.asyncio
    async def test_stream_call_eof_without_done(self):
        """Stream ends gracefully when readline() returns empty bytes (no [DONE] marker)."""
        engine = ModelEngine(_make_model())

        raw_lines = [
            b'data: {"choices": [{"delta": {"content": "Hi"}}]}\n\n',
            # No "data: [DONE]" — readline() will return b"" next
        ]
        mock_response = self._mock_streaming_response(raw_lines)

        mock_client = AsyncMock()
        mock_client.post = MagicMock(return_value=mock_response)
        engine._client = mock_client
        engine._running = True

        chunks = []
        async for chunk in engine.stream_call([{"role": "user", "content": "Hi"}]):
            chunks.append(chunk)

        assert [c.delta_content for c in chunks] == ["Hi"]

    @patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"})
    @pytest.mark.asyncio
    async def test_stream_call_skips_blank_lines(self):
        """Blank lines (whitespace-only) between SSE events are skipped."""
        engine = ModelEngine(_make_model())

        # Build lines manually to include real blank lines the helper would strip
        line_data = [
            b'data: {"choices": [{"delta": {"content": "Hi"}}]}\n',
            b"   \n",  # whitespace-only line — empty after strip()
            b'data: {"choices": [{"delta": {"content": "!"}}]}\n',
            b"data: [DONE]\n",
        ]
        line_iter = iter(line_data)

        async def _readline():
            return next(line_iter, b"")

        mock_content = MagicMock()
        mock_content.readline = _readline

        mock_response = AsyncMock()
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.status = 200
        mock_response.content = mock_content

        mock_client = AsyncMock()
        mock_client.post = MagicMock(return_value=mock_response)
        engine._client = mock_client
        engine._running = True

        chunks = []
        async for chunk in engine.stream_call([{"role": "user", "content": "Hi"}]):
            chunks.append(chunk)

        assert [c.delta_content for c in chunks] == ["Hi", "!"]


class TestModelEngineConstants:
    """Test values of model-engine-specific constants."""

    def test_engine_base_urls_contains_nim_and_openai(self):
        """Default URL map covers nim and openai engines."""
        assert "nim" in _ENGINE_BASE_URLS
        assert "openai" in _ENGINE_BASE_URLS

    def test_chat_completions_endpoint(self):
        """Endpoint path matches OpenAI-compatible chat completions."""
        assert _CHAT_COMPLETIONS_ENDPOINT == "/v1/chat/completions"


class TestModelEngineStreamChatCompletion:
    """Test ModelEngine.stream_chat_completion() delegates to stream_call()."""

    @patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"})
    @pytest.mark.asyncio
    async def test_yields_chunks_from_stream_call(self):
        """stream_chat_completion() yields all chunks from stream_call()."""
        engine = ModelEngine(_make_model())

        async def mock_stream_call(messages, **kwargs):
            for text in ["Hello", " world"]:
                yield LLMResponseChunk(delta_content=text)

        engine.stream_call = mock_stream_call  # type: ignore[method-assign]

        chunks = []
        async for chunk in engine.stream_chat_completion([{"role": "user", "content": "Hi"}]):
            chunks.append(chunk)

        assert [c.delta_content for c in chunks] == ["Hello", " world"]

    @patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"})
    @pytest.mark.asyncio
    async def test_forwards_kwargs_to_stream_call(self):
        """stream_chat_completion() passes kwargs through to stream_call()."""
        engine = ModelEngine(_make_model())
        captured_kwargs = {}

        async def mock_stream_call(messages, **kwargs):
            captured_kwargs.update(kwargs)
            yield LLMResponseChunk(delta_content="ok")

        engine.stream_call = mock_stream_call  # type: ignore[method-assign]

        async for _ in engine.stream_chat_completion(
            [{"role": "user", "content": "Hi"}], temperature=0.7, max_tokens=50
        ):
            pass

        assert captured_kwargs["temperature"] == 0.7
        assert captured_kwargs["max_tokens"] == 50


class TestModelEngineChatCompletion:
    """Test ModelEngine.chat_completion() parses OpenAI-format responses into LLMResponse."""

    @patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"})
    @pytest.mark.asyncio
    async def test_returns_llm_response_with_content(self):
        """chat_completion() returns an LLMResponse carrying the assistant message content."""
        engine = ModelEngine(_make_model())
        engine.call = AsyncMock(return_value={"choices": [{"message": {"role": "assistant", "content": "Hello!"}}]})

        result = await engine.chat_completion([{"role": "user", "content": "Hi"}])

        assert isinstance(result, LLMResponse)
        assert result.content == "Hello!"
        assert result.reasoning is None

    @patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"})
    @pytest.mark.asyncio
    async def test_returns_reasoning_when_present(self):
        """chat_completion() forwards message.reasoning_content to LLMResponse.reasoning."""
        engine = ModelEngine(_make_model())
        engine.call = AsyncMock(
            return_value={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "The answer is 42.",
                            "reasoning_content": "Let me think step by step...",
                        }
                    }
                ]
            }
        )

        result = await engine.chat_completion([{"role": "user", "content": "Hi"}])

        assert result.content == "The answer is 42."
        assert result.reasoning == "Let me think step by step..."

    @patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"})
    @pytest.mark.asyncio
    async def test_returns_usage_with_reasoning_tokens(self):
        """chat_completion() forwards usage incl. reasoning_tokens from completion_tokens_details."""
        engine = ModelEngine(_make_model())
        engine.call = AsyncMock(
            return_value={
                "id": "chatcmpl-abc",
                "model": "gpt-5",
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 25,
                    "total_tokens": 35,
                    "completion_tokens_details": {"reasoning_tokens": 12},
                    "prompt_tokens_details": {"cached_tokens": 4},
                },
            }
        )

        result = await engine.chat_completion([{"role": "user", "content": "Hi"}])

        assert result.model == "gpt-5"
        assert result.finish_reason == "stop"
        assert result.request_id == "chatcmpl-abc"
        assert result.usage == UsageInfo(
            input_tokens=10,
            output_tokens=25,
            total_tokens=35,
            reasoning_tokens=12,
            cached_tokens=4,
        )

    @patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"})
    @pytest.mark.asyncio
    async def test_forwards_kwargs_to_call(self):
        """chat_completion() passes kwargs through to call()."""
        engine = ModelEngine(_make_model())
        engine.call = AsyncMock(return_value={"choices": [{"message": {"content": "ok"}}]})

        await engine.chat_completion([{"role": "user", "content": "Hi"}], temperature=0.5, max_tokens=100)
        call_kwargs = engine.call.call_args[1]
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["max_tokens"] == 100

    @patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"})
    @pytest.mark.asyncio
    async def test_raises_on_missing_choices(self):
        """chat_completion() raises ModelEngineError when 'choices' key is missing."""
        engine = ModelEngine(_make_model())
        engine.call = AsyncMock(return_value={})

        with pytest.raises(ModelEngineError, match="Unexpected response format"):
            await engine.chat_completion([{"role": "user", "content": "Hi"}])

    @patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"})
    @pytest.mark.asyncio
    async def test_raises_on_empty_choices(self):
        """chat_completion() raises ModelEngineError when choices list is empty."""
        engine = ModelEngine(_make_model())
        engine.call = AsyncMock(return_value={"choices": []})

        with pytest.raises(ModelEngineError, match="Unexpected response format"):
            await engine.chat_completion([{"role": "user", "content": "Hi"}])

    @patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"})
    @pytest.mark.asyncio
    async def test_raises_on_missing_message(self):
        """chat_completion() raises ModelEngineError when 'message' key is missing from choice."""
        engine = ModelEngine(_make_model())
        engine.call = AsyncMock(return_value={"choices": [{}]})

        with pytest.raises(ModelEngineError, match="Unexpected response format"):
            await engine.chat_completion([{"role": "user", "content": "Hi"}])

    @patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"})
    @pytest.mark.asyncio
    async def test_raises_on_missing_content(self):
        """chat_completion() raises ModelEngineError when 'content' key is missing from message."""
        engine = ModelEngine(_make_model())
        engine.call = AsyncMock(return_value={"choices": [{"message": {}}]})

        with pytest.raises(ModelEngineError, match="Unexpected response format"):
            await engine.chat_completion([{"role": "user", "content": "Hi"}])

    @patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"})
    @pytest.mark.asyncio
    async def test_raises_on_null_content(self):
        """chat_completion() raises ModelEngineError for unsupported tool-calls"""
        engine = ModelEngine(_make_model())
        engine.call = AsyncMock(return_value={"choices": [{"message": {"content": None}}]})

        with pytest.raises(ModelEngineError, match="Expected string content"):
            await engine.chat_completion([{"role": "user", "content": "Hi"}])

    @patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"})
    @pytest.mark.asyncio
    async def test_raises_clearer_error_for_tool_call_only_response(self):
        """Tool-call-only responses (content=None, tool_calls set) get a scope-specific error."""
        engine = ModelEngine(_make_model())
        engine.call = AsyncMock(
            return_value={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_abc",
                                    "type": "function",
                                    "function": {"name": "calculate", "arguments": '{"expr":"2+2"}'},
                                }
                            ],
                        }
                    }
                ]
            }
        )

        with pytest.raises(ModelEngineError, match="Tool-call-only responses are not yet supported"):
            await engine.chat_completion([{"role": "user", "content": "Hi"}])


class TestParseChatCompletion:
    """Direct tests for the _parse_chat_completion helper."""

    def test_minimal_response(self):
        """Parses content; leaves optional fields as None."""
        result = _parse_chat_completion({"choices": [{"message": {"content": "hi"}}]})
        assert isinstance(result, LLMResponse)
        assert result.content == "hi"
        assert result.reasoning is None
        assert result.usage is None
        assert result.model is None
        assert result.finish_reason is None

    def test_empty_reasoning_is_normalized_to_none(self):
        """Empty-string reasoning_content is treated as no reasoning."""
        result = _parse_chat_completion({"choices": [{"message": {"content": "hi", "reasoning_content": ""}}]})
        assert result.reasoning is None

    def test_usage_without_details(self):
        """Usage without completion/prompt details still parses base counts."""
        result = _parse_chat_completion(
            {
                "choices": [{"message": {"content": "hi"}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
            }
        )
        assert result.usage == UsageInfo(input_tokens=5, output_tokens=7, total_tokens=12)

    def test_raises_on_malformed_response(self):
        """Missing choices/message/content raises ValueError."""
        with pytest.raises(ValueError, match="Unexpected /v1/chat/completions response shape"):
            _parse_chat_completion({})

    def test_raises_on_non_string_content(self):
        """Non-string content raises ValueError."""
        with pytest.raises(ValueError, match="Expected string content"):
            _parse_chat_completion({"choices": [{"message": {"content": None}}]})

    def test_raises_specific_error_for_tool_call_only_response(self):
        """When content is None and tool_calls is set, raise a scope-specific error.

        OpenAI returns this shape for tool_choice='required'. Tool-call support is out of
        scope for this PR series; the error message should make that clear rather than
        suggesting malformed data.
        """
        with pytest.raises(ValueError, match="Tool-call-only responses are not yet supported"):
            _parse_chat_completion(
                {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [{"id": "x", "type": "function", "function": {"name": "f"}}],
                            }
                        }
                    ]
                }
            )


class TestParseChatCompletionChunk:
    """Direct tests for the _parse_chat_completion_chunk helper."""

    def test_content_only_delta(self):
        """delta.content populates delta_content."""
        result = _parse_chat_completion_chunk({"choices": [{"delta": {"content": "hi"}}]})
        assert isinstance(result, LLMResponseChunk)
        assert result.delta_content == "hi"
        assert result.delta_reasoning is None

    def test_reasoning_only_delta(self):
        """delta.reasoning_content populates delta_reasoning."""
        result = _parse_chat_completion_chunk({"choices": [{"delta": {"reasoning_content": "thinking"}}]})
        assert result is not None
        assert result.delta_content is None
        assert result.delta_reasoning == "thinking"

    def test_empty_reasoning_alongside_content_normalized_to_none(self):
        """Empty-string reasoning_content is normalized to None, matching _parse_chat_completion."""
        result = _parse_chat_completion_chunk({"choices": [{"delta": {"content": "hi", "reasoning_content": ""}}]})
        assert result is not None
        assert result.delta_content == "hi"
        assert result.delta_reasoning is None

    def test_combined_content_and_reasoning_in_one_delta(self):
        """A single delta carrying both content and reasoning_content populates both fields.

        LLMResponseChunk is parallel-optional, not a discriminated union — providers
        may emit both fields on the same SSE chunk.
        """
        result = _parse_chat_completion_chunk(
            {"choices": [{"delta": {"content": "answer", "reasoning_content": "thought"}}]}
        )
        assert result is not None
        assert result.delta_content == "answer"
        assert result.delta_reasoning == "thought"

    def test_role_only_delta_returns_none(self):
        """Role-only deltas (typical first event) are skipped."""
        assert _parse_chat_completion_chunk({"choices": [{"delta": {"role": "assistant"}}]}) is None

    def test_empty_choices_with_no_usage_returns_none(self):
        """Empty-choices events with no usage info are skipped (e.g.
        provider-specific keepalive frames)."""
        assert _parse_chat_completion_chunk({"choices": []}) is None

    def test_empty_choices_with_usage_returns_chunk_with_usage(self):
        """Empty-choices events that carry a ``usage`` payload pass
        through — the OpenAI-compatible terminal chunk emitted with
        ``stream_options.include_usage=true``.  ``delta_content`` and
        ``delta_reasoning`` stay ``None``; ``usage`` is populated."""
        result = _parse_chat_completion_chunk(
            {
                "choices": [],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }
        )
        assert result is not None
        assert result.delta_content is None
        assert result.delta_reasoning is None
        assert result.usage is not None
        assert result.usage.input_tokens == 10
        assert result.usage.output_tokens == 5
        assert result.usage.total_tokens == 15

    def test_content_chunk_with_no_usage_has_usage_none(self):
        """A normal content-bearing chunk still reports ``usage=None``
        — usage flows only on the terminal frame."""
        result = _parse_chat_completion_chunk({"choices": [{"delta": {"content": "hi"}}]})
        assert result is not None
        assert result.usage is None

    def test_finish_only_delta_returns_none(self):
        """Finish-only deltas (no content/reasoning) are skipped, matching prior behavior."""
        assert _parse_chat_completion_chunk({"choices": [{"delta": {}, "finish_reason": "stop"}]}) is None

    def test_passes_through_metadata(self):
        """model, request id, and finish_reason flow into the chunk when content is present."""
        result = _parse_chat_completion_chunk(
            {
                "id": "chunk-1",
                "model": "gpt-5",
                "choices": [{"delta": {"content": "hi"}, "finish_reason": "stop"}],
            }
        )
        assert result is not None
        assert result.model == "gpt-5"
        assert result.request_id == "chunk-1"
        assert result.finish_reason == "stop"
