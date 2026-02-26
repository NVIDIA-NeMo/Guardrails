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

"""Unit tests for api_engine module."""

from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest
from pydantic import SecretStr

from nemoguardrails.guardrails._http import (
    DEFAULT_MAX_ATTEMPTS,
    DEFAULT_TIMEOUT_CONNECT,
    DEFAULT_TIMEOUT_TOTAL,
)
from nemoguardrails.guardrails.api_engine import (
    APIEngine,
    APIEngineError,
)
from nemoguardrails.rails.llm.config import JailbreakDetectionConfig


class TestAPIEngineError:
    """Test the APIEngineError Exception type fields."""

    def test_basic_error(self):
        """Error stores message and endpoint, status defaults to None."""
        err = APIEngineError("something broke", endpoint="https://example.com/api")
        assert str(err) == "something broke"
        assert err.endpoint == "https://example.com/api"
        assert err.status is None

    def test_error_with_status(self):
        """Error stores the HTTP status code when provided."""
        err = APIEngineError("bad request", endpoint="https://example.com/api", status=400)
        assert str(err) == "bad request"
        assert err.status == 400
        assert err.endpoint == "https://example.com/api"

    def test_is_exception(self):
        """APIEngineError is a subclass of Exception."""
        assert issubclass(APIEngineError, Exception)


class TestAPIEngineInit:
    """Test APIEngine initialization and URL construction."""

    def test_url_property(self):
        """URL is constructed from base_url + endpoint."""
        engine = APIEngine(base_url="https://api.example.com", endpoint="/v1/classify")
        assert engine.url == "https://api.example.com/v1/classify"

    def test_url_handles_trailing_slash_on_base(self):
        """Trailing slash on base_url is handled correctly."""
        engine = APIEngine(base_url="https://api.example.com/", endpoint="/v1/classify")
        assert engine.url == "https://api.example.com/v1/classify"

    def test_url_handles_no_leading_slash_on_endpoint(self):
        """Missing leading slash on endpoint is handled correctly."""
        engine = APIEngine(base_url="https://api.example.com", endpoint="v1/classify")
        assert engine.url == "https://api.example.com/v1/classify"

    def test_api_key_stored(self):
        """API key is stored when provided."""
        engine = APIEngine(base_url="https://api.example.com", endpoint="/classify", api_key="my-key")
        assert engine.api_key == "my-key"

    def test_api_key_defaults_to_none(self):
        """API key defaults to None when not provided."""
        engine = APIEngine(base_url="https://api.example.com", endpoint="/classify")
        assert engine.api_key is None

    def test_default_timeout_values(self):
        """Timeout defaults match module constants."""
        engine = APIEngine(base_url="https://api.example.com", endpoint="/classify")
        assert engine._timeout.total == DEFAULT_TIMEOUT_TOTAL
        assert engine._timeout.connect == DEFAULT_TIMEOUT_CONNECT

    def test_custom_timeout_values(self):
        """Timeout values can be overridden."""
        engine = APIEngine(
            base_url="https://api.example.com",
            endpoint="/classify",
            timeout_total=120,
            timeout_connect=30,
        )
        assert engine._timeout.total == 120.0
        assert engine._timeout.connect == 30.0

    def test_custom_max_attempts(self):
        """Max attempts can be overridden."""
        engine = APIEngine(base_url="https://api.example.com", endpoint="/classify", max_attempts=5)
        assert engine._retry_options.attempts == 5

    def test_default_max_attempts(self):
        """Max attempts defaults to module constant."""
        engine = APIEngine(base_url="https://api.example.com", endpoint="/classify")
        assert engine._retry_options.attempts == DEFAULT_MAX_ATTEMPTS

    def test_client_initially_none(self):
        """RetryClient is not created until start() is called."""
        engine = APIEngine(base_url="https://api.example.com", endpoint="/classify")
        assert engine._client is None
        assert not engine._running


class TestAPIEngineFromJailbreakConfig:
    """Test the from_jailbreak_config factory classmethod."""

    def test_creates_engine_from_valid_config(self):
        """Factory creates an APIEngine with correct URL and endpoint."""
        config = JailbreakDetectionConfig(
            nim_base_url="https://ai.api.nvidia.com",
            nim_server_endpoint="/v1/security/nvidia/nemoguard-jailbreak-detect",
        )
        engine = APIEngine.from_jailbreak_config(config)
        assert engine.base_url == "https://ai.api.nvidia.com"
        assert engine.endpoint == "/v1/security/nvidia/nemoguard-jailbreak-detect"
        assert engine.api_key is None

    @patch.dict("os.environ", {"MY_API_KEY": "secret-123"})
    def test_resolves_api_key_from_env_var(self):
        """Factory resolves API key from environment variable."""
        config = JailbreakDetectionConfig(
            nim_base_url="https://ai.api.nvidia.com",
            nim_server_endpoint="/v1/security/nvidia/nemoguard-jailbreak-detect",
            api_key_env_var="MY_API_KEY",
        )
        engine = APIEngine.from_jailbreak_config(config)
        assert engine.api_key == "secret-123"

    def test_resolves_api_key_from_secret_str(self):
        """Factory resolves API key from SecretStr field."""
        config = JailbreakDetectionConfig(
            nim_base_url="https://ai.api.nvidia.com",
            nim_server_endpoint="/v1/security/nvidia/nemoguard-jailbreak-detect",
            api_key=SecretStr("direct-key-456"),
        )
        engine = APIEngine.from_jailbreak_config(config)
        assert engine.api_key == "direct-key-456"

    def test_missing_nim_base_url_raises(self):
        """Factory raises ValueError when nim_base_url is not set."""
        config = JailbreakDetectionConfig(
            nim_server_endpoint="/v1/security/nvidia/nemoguard-jailbreak-detect",
        )
        with pytest.raises(ValueError, match="nim_base_url is required"):
            APIEngine.from_jailbreak_config(config)

    def test_missing_nim_server_endpoint_raises(self):
        """Factory raises ValueError when nim_server_endpoint is not set."""
        config = JailbreakDetectionConfig(
            nim_base_url="https://ai.api.nvidia.com",
            nim_server_endpoint=None,
        )
        with pytest.raises(ValueError, match="nim_server_endpoint is required"):
            APIEngine.from_jailbreak_config(config)


class TestAPIEngineLifecycle:
    """Test the APIEngine start() and stop() client lifecycle."""

    @pytest.mark.asyncio
    async def test_start_stop_lifecycle(self):
        """start() creates the client, stop() tears it down to None."""
        engine = APIEngine(base_url="https://api.example.com", endpoint="/classify")
        assert engine._client is None
        assert not engine._running
        await engine.start()
        assert engine._client is not None
        assert engine._running
        await engine.stop()
        assert engine._client is None
        assert not engine._running

    @pytest.mark.asyncio
    async def test_start_is_idempotent(self):
        """Calling start() twice reuses the same client instance."""
        engine = APIEngine(base_url="https://api.example.com", endpoint="/classify")
        await engine.start()
        first_client = engine._client
        await engine.start()
        assert engine._client is first_client
        await engine.stop()

    @pytest.mark.asyncio
    async def test_stop_when_no_client_is_noop(self):
        """stop() without a prior start() does not raise."""
        engine = APIEngine(base_url="https://api.example.com", endpoint="/classify")
        await engine.stop()
        assert not engine._running

    @pytest.mark.asyncio
    async def test_stop_is_idempotent(self):
        """Calling stop() twice does not raise."""
        engine = APIEngine(base_url="https://api.example.com", endpoint="/classify")
        await engine.start()
        await engine.stop()
        await engine.stop()
        assert not engine._running


class TestAPIEngineContextManager:
    """Test async context manager calls start/stop correctly."""

    @pytest.mark.asyncio
    async def test_context_manager_calls_start_and_stop(self):
        """async with calls start() on enter and stop() on exit."""
        engine = APIEngine(base_url="https://api.example.com", endpoint="/classify")
        assert not engine._running
        async with engine as eng:
            assert eng is engine
            assert engine._running
            assert engine._client is not None
        assert not engine._running
        assert engine._client is None


class TestAPIEngineCall:
    """Test APIEngine.call() HTTP request construction and error handling."""

    @pytest.mark.asyncio
    async def test_successful_call(self):
        """Successful call returns parsed JSON and posts to correct URL with headers."""
        engine = APIEngine(base_url="https://api.example.com", endpoint="/v1/classify", api_key="test-key")

        expected_response = {"jailbreak": False, "score": -0.87}

        mock_response = AsyncMock()
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=expected_response)

        mock_client = AsyncMock()
        mock_client.post = MagicMock(return_value=mock_response)
        mock_client.closed = False

        engine._client = mock_client
        engine._running = True

        result = await engine.call({"input": "Hello"})
        assert result == expected_response

        expected_url = "https://api.example.com/v1/classify"
        expected_headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": "Bearer test-key",
        }
        mock_client.post.assert_called_once_with(expected_url, json={"input": "Hello"}, headers=expected_headers)

    @pytest.mark.asyncio
    async def test_call_without_api_key_omits_auth_header(self):
        """No Authorization header when api_key is None."""
        engine = APIEngine(base_url="https://api.example.com", endpoint="/v1/classify")

        mock_response = AsyncMock()
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"jailbreak": False, "score": -0.87})

        mock_client = AsyncMock()
        mock_client.post = MagicMock(return_value=mock_response)
        mock_client.closed = False
        engine._client = mock_client
        engine._running = True

        await engine.call({"input": "Hi"})

        call_kwargs = mock_client.post.call_args
        headers = call_kwargs[1]["headers"]
        assert "Authorization" not in headers

    @pytest.mark.asyncio
    async def test_call_http_error_raises_api_engine_error(self):
        """HTTP 4xx/5xx raises APIEngineError with status and endpoint."""
        engine = APIEngine(base_url="https://api.example.com", endpoint="/v1/classify")

        mock_response = AsyncMock()
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.status = 400
        mock_response.text = AsyncMock(return_value='{"error": "bad request"}')

        mock_client = AsyncMock()
        mock_client.post = MagicMock(return_value=mock_response)
        mock_client.closed = False
        engine._client = mock_client
        engine._running = True

        with pytest.raises(APIEngineError) as exc_info:
            await engine.call({"input": "test"})

        assert exc_info.value.status == 400
        assert "api.example.com" in exc_info.value.endpoint

    @pytest.mark.asyncio
    async def test_call_unexpected_exception_wraps_in_api_engine_error(self):
        """Non-HTTP exceptions are wrapped in APIEngineError."""
        engine = APIEngine(base_url="https://api.example.com", endpoint="/v1/classify")

        mock_client = AsyncMock()
        mock_client.post = MagicMock(side_effect=RuntimeError("connection dropped"))
        mock_client.closed = False
        engine._client = mock_client
        engine._running = True

        with pytest.raises(APIEngineError, match="connection dropped"):
            await engine.call({"input": "test"})

    @pytest.mark.asyncio
    async def test_call_content_type_error_raises_api_engine_error(self):
        """ContentTypeError from response.json() is caught and wrapped in APIEngineError."""
        engine = APIEngine(base_url="https://api.example.com", endpoint="/v1/classify")

        mock_response = AsyncMock()
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)
        mock_response.status = 200
        mock_response.json = AsyncMock(
            side_effect=aiohttp.ContentTypeError(
                MagicMock(real_url="https://api.example.com/v1/classify"),
                (),
                message="Attempt to decode JSON with unexpected mimetype: text/html",
            )
        )

        mock_client = AsyncMock()
        mock_client.post = MagicMock(return_value=mock_response)
        mock_client.closed = False
        engine._client = mock_client
        engine._running = True

        with pytest.raises(APIEngineError, match="Failed to parse response as JSON"):
            await engine.call({"input": "test"})

    @pytest.mark.asyncio
    async def test_call_raises_if_not_started(self):
        """call() raises APIEngineError if start() hasn't been called."""
        engine = APIEngine(base_url="https://api.example.com", endpoint="/v1/classify")
        assert engine._client is None

        with pytest.raises(APIEngineError, match="has not been started"):
            await engine.call({"input": "test"})
