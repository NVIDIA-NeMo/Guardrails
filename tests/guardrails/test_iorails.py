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

"""Unit tests for iorails module."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from nemoguardrails.guardrails.guardrails_types import RailResult
from nemoguardrails.guardrails.iorails import REFUSAL_MESSAGE, IORails
from nemoguardrails.rails.llm.config import RailsConfig
from nemoguardrails.rails.llm.options import GenerationOptions
from tests.guardrails.test_data import NEMOGUARDS_CONFIG


@pytest.fixture
@patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"})
def rails_config():
    return RailsConfig.from_content(config=NEMOGUARDS_CONFIG)


@pytest.fixture
@patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"})
def iorails(rails_config):
    return IORails(rails_config)


class TestIORailsInit:
    """Test IORails wires up ModelManager and RailsManager from config."""

    def test_creates_model_manager(self, iorails):
        """ModelManager is created during init."""
        assert iorails.model_manager is not None

    def test_creates_rails_manager(self, iorails):
        """RailsManager is created during init."""
        assert iorails.rails_manager is not None

    def test_rails_manager_uses_model_manager(self, iorails):
        """RailsManager receives the same ModelManager instance."""
        assert iorails.rails_manager.model_manager is iorails.model_manager


class TestGenerateAsync:
    """Test the generate_async input-check → LLM → output-check pipeline."""

    @pytest.mark.asyncio
    async def test_safe_input_and_output(self, iorails):
        """Returns LLM response when both input and output rails pass."""
        messages = [{"role": "user", "content": "hi"}]
        llm_response = "Hello from LLM"

        iorails.rails_manager.is_input_safe = AsyncMock(return_value=RailResult(is_safe=True))
        iorails.model_manager.generate_async = AsyncMock(return_value=llm_response)
        iorails.rails_manager.is_output_safe = AsyncMock(return_value=RailResult(is_safe=True))

        result = await iorails.generate_async(messages)

        assert result == {"role": "assistant", "content": llm_response}
        iorails.rails_manager.is_input_safe.assert_called_once_with(messages)
        iorails.model_manager.generate_async.assert_called_once_with("main", messages)
        iorails.rails_manager.is_output_safe.assert_called_once_with(messages, llm_response)

    @pytest.mark.asyncio
    async def test_safe_input_and_output_with_generation_options(self, iorails):
        """Returns LLM response when both input and output rails pass."""
        messages = [{"role": "user", "content": "hi"}]
        llm_response = "Hello from LLM"

        llm_params = {"temperature": 0.01, "max_completion_tokens": 1000}
        options = GenerationOptions(llm_params=llm_params)

        iorails.rails_manager.is_input_safe = AsyncMock(return_value=RailResult(is_safe=True))
        iorails.model_manager.generate_async = AsyncMock(return_value=llm_response)
        iorails.rails_manager.is_output_safe = AsyncMock(return_value=RailResult(is_safe=True))

        result = await iorails.generate_async(messages, options=options)

        assert result == {"role": "assistant", "content": llm_response}
        iorails.rails_manager.is_input_safe.assert_called_once_with(messages)
        iorails.model_manager.generate_async.assert_called_once_with("main", messages, **llm_params)
        iorails.rails_manager.is_output_safe.assert_called_once_with(messages, llm_response)

    @pytest.mark.asyncio
    async def test_safe_input_and_output_call_sequence(self, iorails):
        """Pipeline executes in order: input check → generate → output check."""
        call_order = []

        async def mock_input_safe(messages):
            call_order.append("input")
            return RailResult(is_safe=True)

        async def mock_generate(model_type, messages):
            call_order.append("generate")
            return "response"

        async def mock_output_safe(messages, response):
            call_order.append("output")
            return RailResult(is_safe=True)

        iorails.rails_manager.is_input_safe = mock_input_safe
        iorails.model_manager.generate_async = mock_generate
        iorails.rails_manager.is_output_safe = mock_output_safe

        await iorails.generate_async([{"role": "user", "content": "hi"}])
        assert call_order == ["input", "generate", "output"]

    @pytest.mark.asyncio
    async def test_unsafe_input(self, iorails):
        """Returns refusal and skips LLM + output check when input is unsafe."""
        iorails.rails_manager.is_input_safe = AsyncMock(return_value=RailResult(is_safe=False, reason="blocked"))
        iorails.model_manager.generate_async = AsyncMock()
        iorails.rails_manager.is_output_safe = AsyncMock()

        messages = [{"role": "user", "content": "bad input"}]
        result = await iorails.generate_async(messages)

        assert result == {"role": "assistant", "content": REFUSAL_MESSAGE}
        iorails.rails_manager.is_input_safe.assert_called_once_with(messages)
        iorails.model_manager.generate_async.assert_not_called()
        iorails.rails_manager.is_output_safe.assert_not_called()

    @pytest.mark.asyncio
    async def test_unsafe_output(self, iorails):
        """Returns refusal when output check fails, even though LLM was called."""
        messages = [{"role": "user", "content": "hi"}]
        llm_response = "Unsafe response from the LLM!"

        iorails.rails_manager.is_input_safe = AsyncMock(return_value=RailResult(is_safe=True))
        iorails.model_manager.generate_async = AsyncMock(return_value=llm_response)
        iorails.rails_manager.is_output_safe = AsyncMock(return_value=RailResult(is_safe=False, reason="blocked"))

        result = await iorails.generate_async(messages)

        assert result == {"role": "assistant", "content": REFUSAL_MESSAGE}
        iorails.rails_manager.is_input_safe.assert_called_once_with(messages)
        iorails.model_manager.generate_async.assert_called_once_with("main", messages)
        iorails.rails_manager.is_output_safe.assert_called_once_with(messages, llm_response)


class TestIORailsLifecycle:
    """Test IORails start/stop lifecycle management."""

    @pytest.mark.asyncio
    async def test_start_stop_lifecycle(self, iorails):
        """start() delegates to model_manager.start(), stop() to model_manager.stop()."""
        iorails.model_manager.start = AsyncMock()
        iorails.model_manager.stop = AsyncMock()

        assert not iorails._running
        await iorails.start()
        assert iorails._running
        iorails.model_manager.start.assert_called_once()

        await iorails.stop()
        assert not iorails._running
        iorails.model_manager.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_is_idempotent(self, iorails):
        """Calling start() twice only starts model_manager once."""
        iorails.model_manager.start = AsyncMock()

        await iorails.start()
        await iorails.start()

        iorails.model_manager.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_is_idempotent(self, iorails):
        """Calling stop() twice only stops model_manager once."""
        iorails.model_manager.start = AsyncMock()
        iorails.model_manager.stop = AsyncMock()

        await iorails.start()
        await iorails.stop()
        await iorails.stop()

        iorails.model_manager.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_without_start_is_noop(self, iorails):
        """stop() without a prior start() does not raise."""
        iorails.model_manager.stop = AsyncMock()

        await iorails.stop()
        assert not iorails._running
        iorails.model_manager.stop.assert_not_called()


class TestIORailsStartErrors:
    """Test IORails start() error propagation."""

    @pytest.mark.asyncio
    async def test_start_propagates_model_manager_error(self, iorails):
        """start() re-raises when model_manager.start() fails."""
        iorails.model_manager.start = AsyncMock(side_effect=RuntimeError("engine failed"))

        with pytest.raises(RuntimeError, match="engine failed"):
            await iorails.start()

        assert iorails._running

    @pytest.mark.asyncio
    async def test_start_propagates_model_engine_error(self, iorails):
        """A ModelEngine failure propagates through ModelManager up to IORails.start()."""
        # Inject the failure at the ModelEngine level, leaving ModelManager real
        engine = iorails.model_manager._get_model_engine("content_safety")
        engine.start = AsyncMock(side_effect=RuntimeError("NIM endpoint unreachable"))

        # Mock the other engines so they don't make real HTTP connections
        for engine_type, engine in iorails.model_manager._engines.items():
            if engine_type != "content_safety":
                engine.start = AsyncMock()
                engine.stop = AsyncMock()

        with pytest.raises(RuntimeError, match="Failed to start engines"):
            await iorails.start()

        assert iorails._running
        assert not iorails.model_manager._running

        # The engines that started successfully should have been rolled back
        for engine_type, engine in iorails.model_manager._engines.items():
            if engine_type != "content_safety":
                engine.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_failure_allows_retry(self, iorails):
        """After a failed start(), a subsequent start() is not blocked by the idempotency guard."""
        iorails.model_manager.start = AsyncMock(side_effect=RuntimeError("first fail"))

        with pytest.raises(RuntimeError):
            await iorails.start()
        assert iorails._running

        # Stop IORails (and underlying ModelManager)
        await iorails.stop()

        # Call start() without an exception being thrown, now this will work
        iorails.model_manager.start = AsyncMock()
        await iorails.start()
        assert iorails._running
        iorails.model_manager.start.assert_called_once()


class TestIORailsStopErrors:
    """Test IORails stop() error propagation."""

    @pytest.mark.asyncio
    async def test_stop_propagates_model_manager_error(self, iorails):
        """stop() re-raises when model_manager.stop() fails, but sets _running=False."""
        iorails.model_manager.start = AsyncMock()
        iorails.model_manager.stop = AsyncMock(side_effect=RuntimeError("stop failed"))

        await iorails.start()
        assert iorails._running

        with pytest.raises(RuntimeError, match="stop failed"):
            await iorails.stop()

        # _running should be False due to the finally clause
        assert not iorails._running


class TestIORailsContextManager:
    """Test IORails async context manager."""

    @pytest.mark.asyncio
    async def test_context_manager_calls_start_and_stop(self, iorails):
        """async with calls start() on enter and stop() on exit."""
        iorails.model_manager.start = AsyncMock()
        iorails.model_manager.stop = AsyncMock()

        async with iorails as engine:
            assert engine is iorails
            assert iorails._running
            iorails.model_manager.start.assert_called_once()

        assert not iorails._running
        iorails.model_manager.stop.assert_called_once()


class TestGenerate:
    """Test the synchronous generate() method."""

    def test_generate_delegates_to_generate_async(self, iorails):
        """generate() creates a temp IORails, starts it, calls generate_async, and stops it."""
        messages = [{"role": "user", "content": "hi"}]
        expected = {"role": "assistant", "content": "Hello from LLM"}

        iorails.rails_manager.is_input_safe = AsyncMock(return_value=RailResult(is_safe=True))
        iorails.model_manager.generate_async = AsyncMock(return_value="Hello from LLM")
        iorails.rails_manager.is_output_safe = AsyncMock(return_value=RailResult(is_safe=True))

        # Patch IORails so the temp instance inside generate() uses our mocked iorails
        with patch("nemoguardrails.guardrails.iorails.IORails", return_value=iorails):
            result = iorails.generate(messages)

        assert result == expected

    def test_generate_passes_kwargs(self, iorails):
        """generate() forwards kwargs to generate_async."""
        messages = [{"role": "user", "content": "hi"}]
        options = GenerationOptions(llm_params={"temperature": 0.5})

        iorails.rails_manager.is_input_safe = AsyncMock(return_value=RailResult(is_safe=True))
        iorails.model_manager.generate_async = AsyncMock(return_value="response")
        iorails.rails_manager.is_output_safe = AsyncMock(return_value=RailResult(is_safe=True))

        with patch("nemoguardrails.guardrails.iorails.IORails", return_value=iorails):
            iorails.generate(messages, options=options)

        iorails.model_manager.generate_async.assert_called_once_with("main", messages, temperature=0.5)

    def test_generate_raises_when_called_from_async_loop(self, iorails):
        """generate() raises RuntimeError when called inside a running event loop."""

        async def call_generate():
            iorails.generate([{"role": "user", "content": "hi"}])

        with pytest.raises(RuntimeError):
            asyncio.get_event_loop().run_until_complete(call_generate())


class TestRefusalMessage:
    """Test the REFUSAL_MESSAGE module constant."""

    def test_refusal_message_is_string(self):
        """REFUSAL_MESSAGE is a non-empty string."""
        assert isinstance(REFUSAL_MESSAGE, str)
        assert len(REFUSAL_MESSAGE) > 0
