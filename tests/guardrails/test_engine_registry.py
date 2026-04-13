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

"""Unit tests for engine_registry module."""

from unittest.mock import AsyncMock, patch

import pytest

from nemoguardrails.guardrails.api_engine import APIEngine
from nemoguardrails.guardrails.engine_registry import EngineRegistry
from nemoguardrails.guardrails.model_engine import ModelEngine
from nemoguardrails.rails.llm.config import RailsConfig
from tests.guardrails.test_data import NEMOGUARDS_CONFIG


@pytest.fixture
def rails_config():
    """Create a RailsConfig from the nemoguards_v2 test data."""
    return RailsConfig.from_content(config=NEMOGUARDS_CONFIG)


@pytest.fixture
@patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"})
def manager(rails_config):
    """Create a EngineRegistry from test config."""
    return EngineRegistry(rails_config.models, rails_config.rails.config)


class TestEngineRegistryInit:
    """Test EngineRegistry creates engines from config."""

    def test_create_engines_for_each_model_type(self, manager):
        """Creates one engine per model type in config, plus API engines."""
        engine_names = set(manager._engines.keys())
        assert {"main", "content_safety", "topic_control", "jailbreak_detection"} == engine_names

    @patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"})
    def test_empty_config_creates_no_engines(self):
        """Empty models list results in no engines."""
        config = RailsConfig.from_content(config={"models": []})
        mgr = EngineRegistry(config.models, config.rails.config)
        assert len(mgr._engines) == 0

    @patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"})
    def test_model_type_collision_with_api_engine_raises(self):
        """Raises ValueError when a model type collides with an API engine name."""
        config = RailsConfig.from_content(
            config={
                "models": [
                    {"type": "main", "engine": "nim", "model": "meta/llama-3.3-70b-instruct"},
                    {"type": "jailbreak_detection", "engine": "nim", "model": "some/model"},
                ],
                "rails": {
                    "config": {
                        "jailbreak_detection": {
                            "nim_base_url": "https://ai.api.nvidia.com",
                            "nim_server_endpoint": "/v1/security/nvidia/nemoguard-jailbreak-detect",
                            "api_key_env_var": "NVIDIA_API_KEY",
                        }
                    }
                },
            }
        )
        with pytest.raises(ValueError, match="already registered"):
            EngineRegistry(config.models, config.rails.config)


class TestEngineRegistryGetModelEngine:
    """Test engine lookup by model type."""

    def test_get_existing_engine(self, manager):
        """Returns the main LLM engine with correct model name."""
        engine = manager._get_engine("main", ModelEngine)
        assert engine is not None
        assert engine.model_name == "meta/llama-3.3-70b-instruct"

    def test_get_content_safety_engine(self, manager):
        """Returns the content safety engine with correct model name."""
        engine = manager._get_engine("content_safety", ModelEngine)
        assert engine.model_name == "nvidia/llama-3.1-nemoguard-8b-content-safety"

    def test_get_missing_engine_raises_key_error(self, manager):
        """Raises KeyError for an unconfigured model type."""
        with pytest.raises(KeyError, match="No engine configured with name 'nonexistent'"):
            manager._get_engine("nonexistent", ModelEngine)

    def test_key_error_message_lists_available_types(self, manager):
        """KeyError message includes available model types for debugging."""
        with pytest.raises(KeyError) as exc_info:
            manager._get_engine("missing", ModelEngine)
        assert "main" in str(exc_info.value)

    def test_wrong_type_raises_type_error(self, manager):
        """Raises TypeError when engine exists but is the wrong type."""
        with pytest.raises(TypeError, match="Engine 'jailbreak_detection' is APIEngine, expected ModelEngine"):
            manager._get_engine("jailbreak_detection", ModelEngine)


class TestEngineRegistryLifecycle:
    """Test EngineRegistry start/stop delegation to engines."""

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


class TestEngineRegistryGenerateAsync:
    """Test model_call routes to the correct engine."""

    @pytest.mark.asyncio
    async def test_generate_from_correct_engine(self, manager):
        """Calls the named engine's chat_completion() and returns its result."""
        messages = [{"role": "user", "content": "Hi"}]
        engine = manager._get_engine("main", ModelEngine)
        engine.chat_completion = AsyncMock(return_value="Hello world")

        result = await manager.model_call("main", messages)
        assert result == "Hello world"
        engine.chat_completion.assert_called_once_with(messages)

    @pytest.mark.asyncio
    async def test_passes_kwargs_to_engine(self, manager):
        """Extra kwargs (temperature, max_tokens) are forwarded to engine.chat_completion()."""
        messages = [{"role": "user", "content": "Hi"}]
        engine = manager._get_engine("main", ModelEngine)
        engine.chat_completion = AsyncMock(return_value="ok")

        await manager.model_call("main", messages, temperature=0.5, max_tokens=100)

        call_kwargs = engine.chat_completion.call_args[1]
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["max_tokens"] == 100

    @pytest.mark.asyncio
    async def test_raises_key_error_for_unknown_model_type(self, manager):
        """Raises KeyError when the model type doesn't exist."""
        with pytest.raises(KeyError):
            await manager.model_call("nonexistent", [{"role": "user", "content": "Hi"}])


class TestEngineRegistryStartErrors:
    """Test EngineRegistry start() error handling and rollback."""

    @pytest.mark.asyncio
    async def test_start_rolls_back_on_engine_failure(self, manager):
        """When one engine fails to start, already-started engines are stopped."""
        failing_engine = "jailbreak_detection"

        for name, engine in manager._engines.items():
            if name == failing_engine:
                engine.start = AsyncMock(side_effect=RuntimeError("Error starting model"))
            else:
                engine.start = AsyncMock()
                engine.stop = AsyncMock()

        with pytest.raises(RuntimeError, match="Failed to start engine"):
            await manager.start()

        # Engines before the failing one should have been rolled back
        engine_names = list(manager._engines.keys())
        failed_idx = engine_names.index(failing_engine)
        for i, name in enumerate(engine_names):
            if i < failed_idx:
                manager._engines[name].stop.assert_called_once()

        assert not manager._running

    @pytest.mark.asyncio
    async def test_start_error_message_includes_engine_type(self, manager):
        """Error message includes which engine types failed."""
        engine = manager._get_engine("main", ModelEngine)
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
        failing_engine = "jailbreak_detection"
        stop_error_engine = "main"

        for name, engine in manager._engines.items():
            if name == failing_engine:
                engine.start = AsyncMock(side_effect=RuntimeError("start failed"))
            elif name == stop_error_engine:
                engine.start = AsyncMock()
                engine.stop = AsyncMock(side_effect=RuntimeError("stop failed"))
            else:
                engine.start = AsyncMock()
                engine.stop = AsyncMock()

        with pytest.raises(RuntimeError, match="Failed to start engine"):
            await manager.start()

        # All started engines should have had stop() called (even if one raises)
        engine_names = list(manager._engines.keys())
        failed_idx = engine_names.index(failing_engine)
        for i, name in enumerate(engine_names):
            if i < failed_idx:
                manager._engines[name].stop.assert_called_once()


class TestEngineRegistryStopErrors:
    """Test EngineRegistry stop() error handling."""

    @pytest.mark.asyncio
    async def test_stop_raises_on_engine_error(self, manager):
        """stop() raises RuntimeError when an engine fails to stop."""
        for engine in manager._engines.values():
            engine.start = AsyncMock()

        await manager.start()

        # One engine fails to stop
        engine = manager._get_engine("main", ModelEngine)
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

        engine = manager._get_engine("content_safety", ModelEngine)
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


class TestEngineRegistryContextManager:
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


class TestEngineRegistryGetApiEngine:
    """Test API engine lookup by name."""

    def test_get_existing_api_engine(self, manager):
        """Returns the jailbreak detection API engine."""
        api_engine = manager._get_engine("jailbreak_detection", APIEngine)
        assert api_engine is not None
        assert "jailbreak" in api_engine.url

    def test_get_missing_api_engine_raises_key_error(self, manager):
        """Raises KeyError for an unconfigured API engine name."""
        with pytest.raises(KeyError, match="No engine configured with name 'nonexistent'"):
            manager._get_engine("nonexistent", APIEngine)

    def test_key_error_message_lists_available_engines(self, manager):
        """KeyError message includes available API engine names."""
        with pytest.raises(KeyError) as exc_info:
            manager._get_engine("missing", APIEngine)
        assert "jailbreak_detection" in str(exc_info.value)


class TestEngineRegistryApiCall:
    """Test api_call routes to the correct API engine."""

    @pytest.mark.asyncio
    async def test_calls_correct_api_engine(self, manager):
        """api_call delegates to the named API engine and returns its response."""
        api_engine = manager._get_engine("jailbreak_detection", APIEngine)
        mock_response = {"jailbreak": False, "score": -0.95}
        api_engine.call = AsyncMock(return_value=mock_response)

        result = await manager.api_call("jailbreak_detection", {"input": "hello"})

        assert result == mock_response
        api_engine.call.assert_called_once_with({"input": "hello"})

    @pytest.mark.asyncio
    async def test_passes_kwargs_to_api_engine(self, manager):
        """Extra kwargs are forwarded to the API engine's call()."""
        api_engine = manager._get_engine("jailbreak_detection", APIEngine)
        api_engine.call = AsyncMock(return_value={"jailbreak": False, "score": -0.80})

        await manager.api_call("jailbreak_detection", {"input": "test"}, extra_param="value")

        api_engine.call.assert_called_once_with({"input": "test"}, extra_param="value")

    @pytest.mark.asyncio
    async def test_raises_key_error_for_unknown_api_name(self, manager):
        """Raises KeyError when the API engine name doesn't exist."""
        with pytest.raises(KeyError):
            await manager.api_call("nonexistent", {"input": "test"})


class TestEngineRegistryApiEngineStartErrors:
    """Test start() error handling for API engines."""

    @pytest.mark.asyncio
    async def test_start_rolls_back_on_api_engine_failure(self, manager):
        """When an API engine fails to start, all started engines are rolled back."""
        # Mock all engines to succeed
        for engine in manager._engines.values():
            engine.start = AsyncMock()
            engine.stop = AsyncMock()

        # Override jailbreak API engine to fail
        api_engine = manager._get_engine("jailbreak_detection", APIEngine)
        api_engine.start = AsyncMock(side_effect=RuntimeError("API unreachable"))
        api_engine.stop = AsyncMock()

        with pytest.raises(RuntimeError, match="Failed to start engine"):
            await manager.start()

        # Engines that started successfully should have been rolled back
        for name, engine in manager._engines.items():
            if name != "jailbreak_detection":
                engine.stop.assert_called_once()

        assert not manager._running

    @pytest.mark.asyncio
    async def test_start_error_message_includes_api_engine_name(self, manager):
        """Error message includes which API engine failed to start."""
        for engine in manager._engines.values():
            engine.start = AsyncMock()
            engine.stop = AsyncMock()

        api_engine = manager._get_engine("jailbreak_detection", APIEngine)
        api_engine.start = AsyncMock(side_effect=RuntimeError("timeout"))
        api_engine.stop = AsyncMock()

        with pytest.raises(RuntimeError, match="Engine jailbreak_detection"):
            await manager.start()


class TestEngineRegistryApiEngineStopErrors:
    """Test stop() error handling for API engines."""

    @pytest.mark.asyncio
    async def test_stop_raises_on_api_engine_error(self, manager):
        """stop() raises RuntimeError when an API engine fails to stop."""
        for engine in manager._engines.values():
            engine.start = AsyncMock()
            engine.stop = AsyncMock()

        api_engine = manager._get_engine("jailbreak_detection", APIEngine)
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

        api_engine = manager._get_engine("jailbreak_detection", APIEngine)
        api_engine.start = AsyncMock()
        api_engine.stop = AsyncMock(side_effect=RuntimeError("timeout"))

        await manager.start()

        with pytest.raises(RuntimeError, match="Engine jailbreak_detection"):
            await manager.stop()


class TestEngineRegistryStreamModelCall:
    """Test stream_model_call routes to the correct engine and yields chunks."""

    @pytest.mark.asyncio
    async def test_streams_chunks_from_correct_engine(self, manager):
        """Calls the named engine's stream_chat_completion and yields all chunks."""
        messages = [{"role": "user", "content": "Hi"}]

        async def mock_stream_chat_completion(msgs, **kwargs):
            """Mock stream yielding two chunks."""
            for chunk in ["Hello", " world"]:
                yield chunk

        engine = manager._get_engine("main", ModelEngine)
        engine.stream_chat_completion = mock_stream_chat_completion

        chunks = []
        async for chunk in manager.stream_model_call("main", messages):
            chunks.append(chunk)

        assert chunks == ["Hello", " world"]

    @pytest.mark.asyncio
    async def test_forwards_kwargs_to_engine(self, manager):
        """Extra kwargs are forwarded to engine.stream_chat_completion()."""
        messages = [{"role": "user", "content": "Hi"}]
        captured_kwargs = {}

        async def mock_stream_chat_completion(msgs, **kwargs):
            """Mock stream that records kwargs."""
            captured_kwargs.update(kwargs)
            yield "ok"

        engine = manager._get_engine("main", ModelEngine)
        engine.stream_chat_completion = mock_stream_chat_completion

        async for _ in manager.stream_model_call("main", messages, temperature=0.7):
            pass

        assert captured_kwargs["temperature"] == 0.7

    @pytest.mark.asyncio
    async def test_raises_key_error_for_unknown_model_type(self, manager):
        """Raises KeyError when the model type doesn't exist."""
        with pytest.raises(KeyError):
            await anext(manager.stream_model_call("nonexistent", [{"role": "user", "content": "Hi"}]))
