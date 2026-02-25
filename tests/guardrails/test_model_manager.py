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

"""Unit tests for model_manager module."""

from unittest.mock import AsyncMock, patch

import pytest

from nemoguardrails.guardrails.model_manager import ModelManager
from nemoguardrails.rails.llm.config import RailsConfig
from tests.guardrails.test_data import NEMOGUARDS_CONFIG


@pytest.fixture
def rails_config():
    """Create a RailsConfig from the nemoguards_v2 test data."""
    return RailsConfig.from_content(config=NEMOGUARDS_CONFIG)


@pytest.fixture
@patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"})
def manager(rails_config):
    """Create a ModelManager from test config."""
    return ModelManager(rails_config)


class TestModelManagerInit:
    """Test ModelManager creates engines from config."""

    def test_create_engines_for_each_model_type(self, manager):
        """Creates one engine per model type in config."""
        manager_engine_types = {engine for engine, _ in manager._engines.items()}
        assert {"main", "content_safety", "topic_control"} == manager_engine_types

    @patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"})
    def test_empty_config_creates_no_engines(self):
        """Empty models list results in no engines."""
        config = RailsConfig.from_content(config={"models": []})
        mgr = ModelManager(config)
        assert len(mgr._engines) == 0


class TestModelManagerGetModelEngine:
    """Test engine lookup by model type."""

    def test_get_existing_engine(self, manager):
        """Returns the main LLM engine with correct model name."""
        engine = manager._get_model_engine("main")
        assert engine is not None
        assert engine.model_name == "meta/llama-3.3-70b-instruct"

    def test_get_content_safety_engine(self, manager):
        """Returns the content safety engine with correct model name."""
        engine = manager._get_model_engine("content_safety")
        assert engine.model_name == "nvidia/llama-3.1-nemoguard-8b-content-safety"

    def test_get_missing_engine_raises_key_error(self, manager):
        """Raises KeyError for an unconfigured model type."""
        with pytest.raises(KeyError, match="No model configured with type 'nonexistent'"):
            manager._get_model_engine("nonexistent")

    def test_key_error_message_lists_available_types(self, manager):
        """KeyError message includes available model types for debugging."""
        with pytest.raises(KeyError) as exc_info:
            manager._get_model_engine("missing")
        assert "main" in str(exc_info.value)


class TestModelManagerLifecycle:
    """Test ModelManager start/stop delegation to engines."""

    @pytest.mark.asyncio
    async def test_start_calls_start_on_all_engines(self, manager):
        """start() delegates to each engine's start() and sets _running."""
        for engine in manager._engines.values():
            engine.start = AsyncMock()

        assert not manager._running
        await manager.start()
        assert manager._running

        for engine in manager._engines.values():
            engine.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_calls_stop_on_all_engines(self, manager):
        """stop() delegates to each engine's stop() and clears _running."""
        for engine in manager._engines.values():
            engine.start = AsyncMock()
            engine.stop = AsyncMock()

        await manager.start()
        assert manager._running
        await manager.stop()
        assert not manager._running

        for engine in manager._engines.values():
            engine.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_is_idempotent(self, manager):
        """Calling start() twice only starts engines once."""
        for engine in manager._engines.values():
            engine.start = AsyncMock()

        await manager.start()
        await manager.start()  # second call is a no-op

        for engine in manager._engines.values():
            engine.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_is_idempotent(self, manager):
        """Calling stop() twice only stops engines once."""
        for engine in manager._engines.values():
            engine.start = AsyncMock()
            engine.stop = AsyncMock()

        await manager.start()
        await manager.stop()
        await manager.stop()  # second call is a no-op

        for engine in manager._engines.values():
            engine.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_without_start_is_noop(self, manager):
        """stop() without a prior start() does not raise."""
        for engine in manager._engines.values():
            engine.stop = AsyncMock()

        await manager.stop()  # should not raise
        assert not manager._running

        for engine in manager._engines.values():
            engine.stop.assert_not_called()


class TestModelManagerGenerateAsync:
    """Test generate_async routes to the correct engine and extracts content."""

    @pytest.mark.asyncio
    async def test_generate_from_correct_engine(self, manager):
        """Calls the named engine and returns choices[0].message.content."""
        messages = [{"role": "user", "content": "Hi"}]
        mock_response = {"choices": [{"message": {"role": "assistant", "content": "Hello world"}}]}
        engine = manager._get_model_engine("main")
        engine.call = AsyncMock(return_value=mock_response)

        result = await manager.generate_async("main", messages)
        assert result == "Hello world"
        engine.call.assert_called_once_with(messages)

    @pytest.mark.asyncio
    async def test_passes_kwargs_to_engine(self, manager):
        """Extra kwargs (temperature, max_tokens) are forwarded to engine.call()."""
        messages = [{"role": "user", "content": "Hi"}]
        mock_response = {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
        engine = manager._get_model_engine("main")
        engine.call = AsyncMock(return_value=mock_response)

        await manager.generate_async("main", messages, temperature=0.5, max_tokens=100)

        call_kwargs = engine.call.call_args[1]
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["max_tokens"] == 100

    @pytest.mark.asyncio
    async def test_raises_key_error_for_unknown_model_type(self, manager):
        """Raises KeyError when the model type doesn't exist."""
        with pytest.raises(KeyError):
            await manager.generate_async("nonexistent", [{"role": "user", "content": "Hi"}])


class TestModelManagerStartErrors:
    """Test ModelManager start() error handling and rollback."""

    @pytest.mark.asyncio
    async def test_start_rolls_back_on_engine_failure(self, manager):
        """When one engine fails to start, already-started engines are stopped."""
        engines = list(manager._engines.values())

        # First two engines start OK, third raises
        engines[0].start = AsyncMock()
        engines[0].stop = AsyncMock()
        engines[1].start = AsyncMock()
        engines[1].stop = AsyncMock()
        engines[2].start = AsyncMock(side_effect=RuntimeError("Error starting model"))
        engines[2].stop = AsyncMock()

        with pytest.raises(RuntimeError, match="Failed to start engines"):
            await manager.start()

        # Successfully-started engines should have been rolled back
        engines[0].stop.assert_called_once()
        engines[1].stop.assert_called_once()

        # Manager should not be running
        assert not manager._running

    @pytest.mark.asyncio
    async def test_start_error_message_includes_engine_type(self, manager):
        """Error message includes which engine types failed."""
        engine = manager._get_model_engine("main")
        engine.start = AsyncMock(side_effect=RuntimeError("connection refused"))

        # Mock other engines to succeed
        for engine_type, engine in manager._engines.items():
            if engine_type != "main":
                engine.start = AsyncMock()
                engine.stop = AsyncMock()

        with pytest.raises(RuntimeError, match="Engine main"):
            await manager.start()

    @pytest.mark.asyncio
    async def test_start_rollback_swallows_stop_errors(self, manager):
        """Rollback continues even if stopping a started engine raises."""
        engines = list(manager._engines.items())

        # First engine starts OK but stop raises during rollback
        engines[0][1].start = AsyncMock()
        engines[0][1].stop = AsyncMock(side_effect=RuntimeError("stop failed"))

        # Second engine starts OK
        engines[1][1].start = AsyncMock()
        engines[1][1].stop = AsyncMock()

        # Third engine fails to start
        engines[2][1].start = AsyncMock(side_effect=RuntimeError("start failed"))
        engines[2][1].stop = AsyncMock()

        with pytest.raises(RuntimeError, match="Failed to start engines"):
            await manager.start()

        # Both started engines should have had stop() called (even if one raises)
        engines[0][1].stop.assert_called_once()
        engines[1][1].stop.assert_called_once()


class TestModelManagerStopErrors:
    """Test ModelManager stop() error handling."""

    @pytest.mark.asyncio
    async def test_stop_raises_on_engine_error(self, manager):
        """stop() raises RuntimeError when an engine fails to stop."""
        for engine in manager._engines.values():
            engine.start = AsyncMock()

        await manager.start()

        # One engine fails to stop
        engine = manager._get_model_engine("main")
        engine.stop = AsyncMock(side_effect=RuntimeError("close failed"))
        for engine_type, engine in manager._engines.items():
            if engine_type != "main":
                engine.stop = AsyncMock()

        with pytest.raises(RuntimeError, match="Failed to stop engines"):
            await manager.stop()

    @pytest.mark.asyncio
    async def test_stop_error_includes_engine_type(self, manager):
        """Error message includes which engine type failed to stop."""
        for engine in manager._engines.values():
            engine.start = AsyncMock()

        await manager.start()

        engine = manager._get_model_engine("content_safety")
        engine.stop = AsyncMock(side_effect=RuntimeError("timeout"))
        for engine_type, engine in manager._engines.items():
            if engine_type != "content_safety":
                engine.stop = AsyncMock()

        with pytest.raises(RuntimeError, match="Engine content_safety"):
            await manager.stop()

    @pytest.mark.asyncio
    async def test_stop_attempts_all_engines_even_on_errors(self, manager):
        """stop() tries to stop all engines, not just the first one that fails."""
        for engine in manager._engines.values():
            engine.start = AsyncMock()

        await manager.start()

        for engine in manager._engines.values():
            engine.stop = AsyncMock(side_effect=RuntimeError("fail"))

        with pytest.raises(RuntimeError):
            await manager.stop()

        # All engines should have had stop() called
        for engine in manager._engines.values():
            engine.stop.assert_called_once()


class TestModelManagerContextManager:
    """Test async context manager calls start/stop correctly."""

    @pytest.mark.asyncio
    async def test_context_manager_calls_start_and_stop(self, manager):
        """async with calls start() on enter and stop() on exit."""
        for engine in manager._engines.values():
            engine.start = AsyncMock()
            engine.stop = AsyncMock()

        async with manager as mgr:
            assert mgr is manager
            for engine in manager._engines.values():
                engine.start.assert_called_once()

        for engine in manager._engines.values():
            engine.stop.assert_called_once()


class TestModelManagerGetApiEngine:
    """Test API engine lookup by name."""

    def test_get_existing_api_engine(self, manager):
        """Returns the jailbreak detection API engine."""
        api_engine = manager._get_api_engine("jailbreak_detection")
        assert api_engine is not None
        assert "jailbreak" in api_engine.url

    def test_get_missing_api_engine_raises_key_error(self, manager):
        """Raises KeyError for an unconfigured API engine name."""
        with pytest.raises(KeyError, match="No API engine configured with name 'nonexistent'"):
            manager._get_api_engine("nonexistent")

    def test_key_error_message_lists_available_engines(self, manager):
        """KeyError message includes available API engine names."""
        with pytest.raises(KeyError) as exc_info:
            manager._get_api_engine("missing")
        assert "jailbreak_detection" in str(exc_info.value)


class TestModelManagerApiCall:
    """Test api_call routes to the correct API engine."""

    @pytest.mark.asyncio
    async def test_calls_correct_api_engine(self, manager):
        """api_call delegates to the named API engine and returns its response."""
        api_engine = manager._get_api_engine("jailbreak_detection")
        mock_response = {"jailbreak": False, "score": -0.95}
        api_engine.call = AsyncMock(return_value=mock_response)

        result = await manager.api_call("jailbreak_detection", {"input": "hello"})

        assert result == mock_response
        api_engine.call.assert_called_once_with({"input": "hello"})

    @pytest.mark.asyncio
    async def test_passes_kwargs_to_api_engine(self, manager):
        """Extra kwargs are forwarded to the API engine's call()."""
        api_engine = manager._get_api_engine("jailbreak_detection")
        api_engine.call = AsyncMock(return_value={"jailbreak": False, "score": -0.80})

        await manager.api_call("jailbreak_detection", {"input": "test"}, extra_param="value")

        api_engine.call.assert_called_once_with({"input": "test"}, extra_param="value")

    @pytest.mark.asyncio
    async def test_raises_key_error_for_unknown_api_name(self, manager):
        """Raises KeyError when the API engine name doesn't exist."""
        with pytest.raises(KeyError):
            await manager.api_call("nonexistent", {"input": "test"})


class TestModelManagerApiEngineStartErrors:
    """Test start() error handling for API engines."""

    @pytest.mark.asyncio
    async def test_start_rolls_back_on_api_engine_failure(self, manager):
        """When an API engine fails to start, all started engines are rolled back."""
        # Mock model engines to succeed
        for engine in manager._engines.values():
            engine.start = AsyncMock()
            engine.stop = AsyncMock()

        # Mock API engine to fail
        api_engine = manager._get_api_engine("jailbreak_detection")
        api_engine.start = AsyncMock(side_effect=RuntimeError("API unreachable"))
        api_engine.stop = AsyncMock()

        with pytest.raises(RuntimeError, match="Failed to start engines"):
            await manager.start()

        # Model engines that started should have been rolled back
        for engine in manager._engines.values():
            engine.stop.assert_called_once()

        assert not manager._running

    @pytest.mark.asyncio
    async def test_start_error_message_includes_api_engine_name(self, manager):
        """Error message includes which API engine failed to start."""
        for engine in manager._engines.values():
            engine.start = AsyncMock()
            engine.stop = AsyncMock()

        api_engine = manager._get_api_engine("jailbreak_detection")
        api_engine.start = AsyncMock(side_effect=RuntimeError("timeout"))
        api_engine.stop = AsyncMock()

        with pytest.raises(RuntimeError, match="Engine jailbreak_detection"):
            await manager.start()


class TestModelManagerApiEngineStopErrors:
    """Test stop() error handling for API engines."""

    @pytest.mark.asyncio
    async def test_stop_raises_on_api_engine_error(self, manager):
        """stop() raises RuntimeError when an API engine fails to stop."""
        for engine in manager._engines.values():
            engine.start = AsyncMock()
            engine.stop = AsyncMock()

        api_engine = manager._get_api_engine("jailbreak_detection")
        api_engine.start = AsyncMock()
        api_engine.stop = AsyncMock(side_effect=RuntimeError("close failed"))

        await manager.start()

        with pytest.raises(RuntimeError, match="Failed to stop engines"):
            await manager.stop()

    @pytest.mark.asyncio
    async def test_stop_error_includes_api_engine_name(self, manager):
        """Error message includes which API engine failed to stop."""
        for engine in manager._engines.values():
            engine.start = AsyncMock()
            engine.stop = AsyncMock()

        api_engine = manager._get_api_engine("jailbreak_detection")
        api_engine.start = AsyncMock()
        api_engine.stop = AsyncMock(side_effect=RuntimeError("timeout"))

        await manager.start()

        with pytest.raises(RuntimeError, match="Engine jailbreak_detection"):
            await manager.stop()
