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

"""Reasoning-model support for IORails (non-streaming).

Mirrors LLMRails behavior (``actions/llm/utils.py:_store_reasoning_traces``
and ``rails/llm/llmrails.py:1173-1175``):
- Prefer the provider-native ``LLMResponse.reasoning`` field.
- Fall back to inline ``<think>...</think>`` tags in content (the helper
  strips the tags from ``response.content`` in place so output rails never
  see reasoning).
- Surface reasoning to the caller via a ``<think>{reasoning}</think>\\n``
  prefix on the returned content (LLMRails legacy delivery shape).
"""

from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from nemoguardrails.guardrails.guardrails_types import RailResult
from nemoguardrails.guardrails.iorails import REFUSAL_MESSAGE, IORails
from nemoguardrails.types import LLMResponse
from tests.guardrails.async_helpers import started_iorails
from tests.guardrails.test_data import NEMOGUARDS_CONFIG


@pytest_asyncio.fixture
async def iorails():
    """Started IORails instance with worker-queue teardown after each test."""
    async with started_iorails(NEMOGUARDS_CONFIG) as iorails:
        yield iorails


def _stub_safe_rails(iorails: IORails) -> None:
    """Default-safe input + output rails so each test focuses on the LLM call."""
    iorails.rails_manager.is_input_safe = AsyncMock(return_value=RailResult(is_safe=True))
    iorails.rails_manager.is_output_safe = AsyncMock(return_value=RailResult(is_safe=True))


class TestReasoningContent:
    """Reasoning extraction + delivery on the non-streaming path."""

    @pytest.mark.asyncio
    async def test_native_reasoning_field_prefixes_content(self, iorails):
        """``response.reasoning`` is wrapped in <think> tags and prefixed to content."""
        messages = [{"role": "user", "content": "hi"}]
        _stub_safe_rails(iorails)
        iorails.engine_registry.model_call = AsyncMock(
            return_value=LLMResponse(content="Hello", reasoning="thinking step")
        )

        result = await iorails.generate_async(messages)

        assert result == {"role": "assistant", "content": "<think>thinking step</think>\nHello"}
        # Output rails see the original content unchanged when reasoning came from
        # the native field (no <think> tags to strip).
        iorails.rails_manager.is_output_safe.assert_called_once_with(messages, "Hello")

    @pytest.mark.asyncio
    async def test_inline_think_tags_extracted_and_stripped(self, iorails):
        """<think>...</think> in content is extracted, stripped before output rails, and re-prefixed on return."""
        messages = [{"role": "user", "content": "hi"}]
        _stub_safe_rails(iorails)
        iorails.engine_registry.model_call = AsyncMock(
            return_value=LLMResponse(content="<think>thinking step</think>Hello")
        )

        result = await iorails.generate_async(messages)

        assert result == {"role": "assistant", "content": "<think>thinking step</think>\nHello"}
        # Output rails MUST receive content with <think> tags stripped — this is
        # the central guarantee that reasoning bypasses output rails.
        iorails.rails_manager.is_output_safe.assert_called_once_with(messages, "Hello")

    @pytest.mark.asyncio
    async def test_native_reasoning_wins_over_inline_tags(self, iorails):
        """When both are present, ``response.reasoning`` is used and inline tags are NOT stripped.

        Mirrors the LLMRails ``_store_reasoning_traces`` order (utils.py:339-342):
        the inline-tag fallback is only consulted if the native field is empty.
        Inline tags remain in content verbatim — output rails see them unstripped.
        """
        messages = [{"role": "user", "content": "hi"}]
        _stub_safe_rails(iorails)
        iorails.engine_registry.model_call = AsyncMock(
            return_value=LLMResponse(
                content="<think>tag reasoning</think>Hello",
                reasoning="native reasoning",
            )
        )

        result = await iorails.generate_async(messages)

        assert result == {
            "role": "assistant",
            "content": "<think>native reasoning</think>\n<think>tag reasoning</think>Hello",
        }
        iorails.rails_manager.is_output_safe.assert_called_once_with(messages, "<think>tag reasoning</think>Hello")

    @pytest.mark.asyncio
    async def test_malformed_think_tag_opener_only(self, iorails):
        """Opening tag without closing → no extraction, content preserved verbatim."""
        messages = [{"role": "user", "content": "hi"}]
        _stub_safe_rails(iorails)
        iorails.engine_registry.model_call = AsyncMock(return_value=LLMResponse(content="<think>incomplete reasoning"))

        result = await iorails.generate_async(messages)

        assert result == {"role": "assistant", "content": "<think>incomplete reasoning"}
        iorails.rails_manager.is_output_safe.assert_called_once_with(messages, "<think>incomplete reasoning")

    @pytest.mark.asyncio
    async def test_malformed_think_tag_closer_only(self, iorails):
        """Closing tag without opening → no extraction, content preserved verbatim."""
        messages = [{"role": "user", "content": "hi"}]
        _stub_safe_rails(iorails)
        iorails.engine_registry.model_call = AsyncMock(return_value=LLMResponse(content="orphan</think> reply"))

        result = await iorails.generate_async(messages)

        assert result == {"role": "assistant", "content": "orphan</think> reply"}
        iorails.rails_manager.is_output_safe.assert_called_once_with(messages, "orphan</think> reply")

    @pytest.mark.asyncio
    async def test_output_rail_block_with_native_reasoning(self, iorails):
        """When output rails block, reasoning is dropped — refusal carries no <think> prefix."""
        messages = [{"role": "user", "content": "hi"}]
        iorails.rails_manager.is_input_safe = AsyncMock(return_value=RailResult(is_safe=True))
        iorails.rails_manager.is_output_safe = AsyncMock(return_value=RailResult(is_safe=False, reason="unsafe"))
        iorails.engine_registry.model_call = AsyncMock(
            return_value=LLMResponse(content="bad answer", reasoning="reasoning step")
        )

        result = await iorails.generate_async(messages)

        assert result == {"role": "assistant", "content": REFUSAL_MESSAGE}

    @pytest.mark.asyncio
    async def test_output_rail_block_with_inline_tags(self, iorails):
        """Block path strips inline <think> tags from output-rail input even though the rail blocks."""
        messages = [{"role": "user", "content": "hi"}]
        iorails.rails_manager.is_input_safe = AsyncMock(return_value=RailResult(is_safe=True))
        iorails.rails_manager.is_output_safe = AsyncMock(return_value=RailResult(is_safe=False, reason="unsafe"))
        iorails.engine_registry.model_call = AsyncMock(
            return_value=LLMResponse(content="<think>thinking</think>bad answer")
        )

        result = await iorails.generate_async(messages)

        assert result == {"role": "assistant", "content": REFUSAL_MESSAGE}
        # Output rails see only the stripped content — they're judging the model
        # output, not the reasoning trace.
        iorails.rails_manager.is_output_safe.assert_called_once_with(messages, "bad answer")

    @pytest.mark.asyncio
    async def test_empty_string_reasoning_falls_through_to_inline_extraction(self, iorails):
        """``response.reasoning = ''`` is falsy → fallback to inline-tag extraction runs."""
        messages = [{"role": "user", "content": "hi"}]
        _stub_safe_rails(iorails)
        iorails.engine_registry.model_call = AsyncMock(
            return_value=LLMResponse(content="<think>fallback reasoning</think>Hi", reasoning="")
        )

        result = await iorails.generate_async(messages)

        assert result == {"role": "assistant", "content": "<think>fallback reasoning</think>\nHi"}
        iorails.rails_manager.is_output_safe.assert_called_once_with(messages, "Hi")

    @pytest.mark.asyncio
    async def test_no_reasoning_returns_unchanged_content(self, iorails):
        """Baseline: no ``response.reasoning`` and no inline tags → content unmodified, no prefix."""
        messages = [{"role": "user", "content": "hi"}]
        _stub_safe_rails(iorails)
        iorails.engine_registry.model_call = AsyncMock(return_value=LLMResponse(content="plain answer"))

        result = await iorails.generate_async(messages)

        assert result == {"role": "assistant", "content": "plain answer"}
        iorails.rails_manager.is_output_safe.assert_called_once_with(messages, "plain answer")
