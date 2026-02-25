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

from unittest.mock import AsyncMock, MagicMock, patch

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
)
from nemoguardrails.rails.llm.config import Model


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


class TestModelEngineConstants:
    """Test values of model-engine-specific constants."""

    def test_engine_base_urls_contains_nim_and_openai(self):
        """Default URL map covers nim and openai engines."""
        assert "nim" in _ENGINE_BASE_URLS
        assert "openai" in _ENGINE_BASE_URLS

    def test_chat_completions_endpoint(self):
        """Endpoint path matches OpenAI-compatible chat completions."""
        assert _CHAT_COMPLETIONS_ENDPOINT == "/v1/chat/completions"
