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

"""Tests for speculative generation (M2): input rails race LLM generation."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from nemoguardrails.guardrails.guardrails_types import RailResult
from nemoguardrails.guardrails.iorails import REFUSAL_MESSAGE, IORails
from nemoguardrails.rails.llm.config import RailsConfig
from tests.guardrails.test_data import NEMOGUARDS_CONFIG, NEMOGUARDS_SPECULATIVE_CONFIG

MESSAGES = [{"role": "user", "content": "hi"}]


@pytest.fixture
@patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"})
def iorails():
    return IORails(RailsConfig.from_content(config=NEMOGUARDS_SPECULATIVE_CONFIG))


@pytest.fixture
@patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"})
def iorails_sequential():
    return IORails(RailsConfig.from_content(config=NEMOGUARDS_CONFIG))


class TestSpeculativeGeneration:
    """Speculative generation races input rails against LLM generation."""

    @pytest.mark.asyncio
    async def test_rails_first_pass(self, iorails):
        """Rails finish first and pass — generation is awaited, output rails run."""

        async def fast_rails(messages):
            return RailResult(is_safe=True)

        async def slow_llm(model_type, messages):
            await asyncio.sleep(0.05)
            return "Hello from LLM"

        iorails.rails_manager.is_input_safe = fast_rails
        iorails.engine_registry.model_call = slow_llm
        iorails.rails_manager.is_output_safe = AsyncMock(return_value=RailResult(is_safe=True))

        result = await iorails.generate_async(MESSAGES)

        assert result == {"role": "assistant", "content": "Hello from LLM"}

    @pytest.mark.asyncio
    async def test_rails_first_reject(self, iorails):
        """Rails finish first and reject — generation is cancelled, refusal returned."""

        async def fast_reject(messages):
            return RailResult(is_safe=False, reason="unsafe")

        async def slow_llm(model_type, messages):
            await asyncio.sleep(0.5)
            return "Should not be used"

        iorails.rails_manager.is_input_safe = fast_reject
        iorails.engine_registry.model_call = slow_llm
        iorails.rails_manager.is_output_safe = AsyncMock()

        result = await iorails.generate_async(MESSAGES)

        assert result == {"role": "assistant", "content": REFUSAL_MESSAGE}
        iorails.rails_manager.is_output_safe.assert_not_called()

    @pytest.mark.asyncio
    async def test_gen_first_pass(self, iorails):
        """Generation finishes first — rails verdict awaited, response served on pass."""

        async def slow_rails(messages):
            await asyncio.sleep(0.05)
            return RailResult(is_safe=True)

        async def fast_llm(model_type, messages):
            return "Fast LLM response"

        iorails.rails_manager.is_input_safe = slow_rails
        iorails.engine_registry.model_call = fast_llm
        iorails.rails_manager.is_output_safe = AsyncMock(return_value=RailResult(is_safe=True))

        result = await iorails.generate_async(MESSAGES)

        assert result == {"role": "assistant", "content": "Fast LLM response"}

    @pytest.mark.asyncio
    async def test_gen_first_reject(self, iorails):
        """Generation finishes first, then rails reject — response discarded."""

        async def slow_reject(messages):
            await asyncio.sleep(0.05)
            return RailResult(is_safe=False, reason="unsafe")

        async def fast_llm(model_type, messages):
            return "Should be discarded"

        iorails.rails_manager.is_input_safe = slow_reject
        iorails.engine_registry.model_call = fast_llm
        iorails.rails_manager.is_output_safe = AsyncMock()

        result = await iorails.generate_async(MESSAGES)

        assert result == {"role": "assistant", "content": REFUSAL_MESSAGE}
        iorails.rails_manager.is_output_safe.assert_not_called()

    @pytest.mark.asyncio
    async def test_llm_error_cancels_rails(self, iorails):
        """LLM errors while rails still running — rails cancelled, error propagated."""

        async def slow_rails(messages):
            await asyncio.sleep(0.5)
            return RailResult(is_safe=True)

        iorails.rails_manager.is_input_safe = slow_rails
        iorails.engine_registry.model_call = AsyncMock(side_effect=RuntimeError("LLM crashed"))

        with pytest.raises(RuntimeError, match="LLM crashed"):
            await iorails.generate_async(MESSAGES)

    @pytest.mark.asyncio
    async def test_rails_error_cancels_generation(self, iorails):
        """Rails error while LLM still running — generation cancelled, error propagated."""

        async def slow_llm(model_type, messages):
            await asyncio.sleep(0.5)
            return "Should not be used"

        iorails.rails_manager.is_input_safe = AsyncMock(side_effect=RuntimeError("Rails crashed"))
        iorails.engine_registry.model_call = slow_llm

        with pytest.raises(RuntimeError, match="Rails crashed"):
            await iorails.generate_async(MESSAGES)

    @pytest.mark.asyncio
    async def test_flag_disabled_runs_sequentially(self, iorails_sequential):
        """When speculative_generation is false, pipeline runs sequentially."""
        call_order = []

        async def mock_input(messages):
            call_order.append("input")
            return RailResult(is_safe=True)

        async def mock_generate(model_type, messages):
            call_order.append("generate")
            return "response"

        async def mock_output(messages, response):
            call_order.append("output")
            return RailResult(is_safe=True)

        iorails_sequential.rails_manager.is_input_safe = mock_input
        iorails_sequential.engine_registry.model_call = mock_generate
        iorails_sequential.rails_manager.is_output_safe = mock_output

        await iorails_sequential.generate_async(MESSAGES)
        assert call_order == ["input", "generate", "output"]
