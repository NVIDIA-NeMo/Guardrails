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

from unittest.mock import AsyncMock, patch

import pytest

from nemoguardrails.guardrails.guardrails_types import RailResult
from nemoguardrails.guardrails.iorails import REFUSAL_MESSAGE, IORails
from nemoguardrails.rails.llm.config import RailsConfig
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


class TestRefusalMessage:
    """Test the REFUSAL_MESSAGE module constant."""

    def test_refusal_message_is_string(self):
        """REFUSAL_MESSAGE is a non-empty string."""
        assert isinstance(REFUSAL_MESSAGE, str)
        assert len(REFUSAL_MESSAGE) > 0
