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

"""Structured GenerationResponse return from IORails, matching LLMRails.

When ``options`` is supplied, ``generate_async``/``generate`` return a
``GenerationResponse`` instead of a bare ``LLMMessage`` dict, mirroring LLMRails'
``if gen_options:`` branch. When ``options`` is absent the bare dict is returned
unchanged. The structured path populates ``response``, ``reasoning_content``,
``tool_calls`` (as ``ToolCall.to_dict()`` with dict arguments), and
``llm_metadata`` (main-call ``provider_metadata`` plus a ``usage`` sub-key);
``llm_output`` is always ``None`` (parity with LLMRails' unwired ``raw_response``);
``log`` is deferred; and ``output_vars``/``state``/``log`` request options raise.
"""

from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio

from nemoguardrails.guardrails.guardrails_types import RailResult
from nemoguardrails.guardrails.iorails import REFUSAL_MESSAGE, IORails
from nemoguardrails.rails.llm.config import RailsConfig
from nemoguardrails.rails.llm.options import GenerationOptions, GenerationResponse
from nemoguardrails.types import LLMResponse, ToolCall, ToolCallFunction, UsageInfo
from tests.guardrails.async_helpers import started_iorails
from tests.guardrails.test_data import NEMOGUARDS_CONFIG


@pytest_asyncio.fixture
async def iorails():
    """Started IORails instance with worker-queue teardown after each test."""
    async with started_iorails(NEMOGUARDS_CONFIG) as iorails:
        yield iorails


@pytest.fixture
def iorails_sync():
    """Unstarted IORails instance for driving the synchronous ``generate`` path."""
    return IORails(RailsConfig.from_content(config=NEMOGUARDS_CONFIG))


def _stub_safe_rails(iorails: IORails) -> None:
    """Default-safe input, output, and tool-call rails so tests focus on the LLM response."""
    iorails.rails_manager.is_input_safe = AsyncMock(return_value=RailResult(is_safe=True))
    iorails.rails_manager.is_output_safe = AsyncMock(return_value=RailResult(is_safe=True))
    iorails.rails_manager.are_tool_calls_safe = AsyncMock(return_value=RailResult(is_safe=True))


def _stub_model(iorails: IORails, response: LLMResponse) -> None:
    """Make the main-model call return a fixed structured LLMResponse."""
    iorails.engine_registry.model_call = AsyncMock(return_value=response)


_USER = [{"role": "user", "content": "hi"}]


class TestStructuredResponseTrigger:
    """``options`` presence decides GenerationResponse vs. bare dict."""

    @pytest.mark.asyncio
    async def test_options_dict_returns_generation_response(self, iorails):
        """A dict ``options`` argument switches the return type to GenerationResponse."""
        _stub_safe_rails(iorails)
        _stub_model(iorails, LLMResponse(content="Hello"))

        result = await iorails.generate_async(_USER, options={"llm_params": {"temperature": 0.5}})

        assert isinstance(result, GenerationResponse)

    @pytest.mark.asyncio
    async def test_generation_options_returns_generation_response(self, iorails):
        """A GenerationOptions instance also switches the return type."""
        _stub_safe_rails(iorails)
        _stub_model(iorails, LLMResponse(content="Hello"))

        result = await iorails.generate_async(_USER, options=GenerationOptions())

        assert isinstance(result, GenerationResponse)

    @pytest.mark.asyncio
    async def test_no_generation_options_returns_messages(self, iorails):
        """Without ``options`` the return stays the bare assistant-message dict."""
        _stub_safe_rails(iorails)
        _stub_model(iorails, LLMResponse(content="Hello"))

        result = await iorails.generate_async(_USER)

        assert not isinstance(result, GenerationResponse)
        assert result == {"role": "assistant", "content": "Hello"}


class TestResponseField:
    """The ``response`` field wraps the assistant message in a one-element list."""

    @pytest.mark.asyncio
    async def test_plain_content_wrapped_in_list(self, iorails):
        """``response`` is ``[{"role":"assistant","content": text}]``."""
        _stub_safe_rails(iorails)
        _stub_model(iorails, LLMResponse(content="Hello there"))

        result = await iorails.generate_async(_USER, options={})

        assert result.response == [{"role": "assistant", "content": "Hello there"}]


class TestReasoningContent:
    """Reasoning goes to ``reasoning_content`` with clean content (no inline <think>)."""

    @pytest.mark.asyncio
    async def test_native_reasoning_in_field_content_clean(self, iorails):
        """Provider ``reasoning`` populates ``reasoning_content``; ``response`` content has no <think> prefix."""
        _stub_safe_rails(iorails)
        _stub_model(iorails, LLMResponse(content="Hello", reasoning="thinking step"))

        result = await iorails.generate_async(_USER, options={})

        assert isinstance(result, GenerationResponse)
        assert result.reasoning_content == "thinking step"
        assert result.response == [{"role": "assistant", "content": "Hello"}]
        iorails.rails_manager.is_output_safe.assert_called_once_with(_USER, "Hello", enabled=True)

    @pytest.mark.asyncio
    async def test_inline_think_tags_extracted_to_field(self, iorails):
        """Inline <think> tags are stripped into ``reasoning_content`` and never reach output rails."""
        _stub_safe_rails(iorails)
        _stub_model(iorails, LLMResponse(content="<think>thinking step</think>Hello"))

        result = await iorails.generate_async(_USER, options={})

        assert isinstance(result, GenerationResponse)
        assert result.reasoning_content == "thinking step"
        assert result.response == [{"role": "assistant", "content": "Hello"}]
        iorails.rails_manager.is_output_safe.assert_called_once_with(_USER, "Hello", enabled=True)

    @pytest.mark.asyncio
    async def test_no_reasoning_field_is_none(self, iorails):
        """Absent reasoning leaves ``reasoning_content`` as None."""
        _stub_safe_rails(iorails)
        _stub_model(iorails, LLMResponse(content="plain answer"))

        result = await iorails.generate_async(_USER, options={})

        assert isinstance(result, GenerationResponse)
        assert result.reasoning_content is None
        assert result.response == [{"role": "assistant", "content": "plain answer"}]


class TestToolCalls:
    """``tool_calls`` use the LLMRails ``ToolCall.to_dict()`` shape (dict arguments)."""

    @pytest.mark.asyncio
    async def test_tool_calls_serialized_with_dict_arguments(self, iorails):
        """``tool_calls`` is a list of ``to_dict()`` entries whose ``arguments`` stay a dict, not a JSON string."""
        _stub_safe_rails(iorails)
        tool_call = ToolCall(
            id="call_1",
            type="function",
            function=ToolCallFunction(name="get_weather", arguments={"city": "SF"}),
        )
        _stub_model(iorails, LLMResponse(content="", tool_calls=[tool_call]))

        result = await iorails.generate_async(_USER, options={})

        assert isinstance(result, GenerationResponse)
        assert result.tool_calls == [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": {"city": "SF"}},
            }
        ]

    @pytest.mark.asyncio
    async def test_response_message_has_no_tool_calls_key(self, iorails):
        """In the structured path tool calls live only in the top-level field, not on the message."""
        _stub_safe_rails(iorails)
        tool_call = ToolCall(
            id="call_1",
            type="function",
            function=ToolCallFunction(name="get_weather", arguments={"city": "SF"}),
        )
        _stub_model(iorails, LLMResponse(content="", tool_calls=[tool_call]))

        result = await iorails.generate_async(_USER, options={})

        assert isinstance(result, GenerationResponse)
        assert result.response == [{"role": "assistant", "content": ""}]
        assert "tool_calls" not in result.response[0]

    @pytest.mark.asyncio
    async def test_no_tool_calls_field_is_none(self, iorails):
        """A text-only response leaves ``tool_calls`` as None."""
        _stub_safe_rails(iorails)
        _stub_model(iorails, LLMResponse(content="Hello"))

        result = await iorails.generate_async(_USER, options={})

        assert isinstance(result, GenerationResponse)
        assert result.tool_calls is None


class TestLLMMetadata:
    """``llm_metadata`` carries main-call ``provider_metadata`` plus a ``usage`` sub-key."""

    @pytest.mark.asyncio
    async def test_provider_metadata_and_usage_merged(self, iorails):
        """provider_metadata is surfaced and structured usage is added under ``usage``."""
        _stub_safe_rails(iorails)
        _stub_model(
            iorails,
            LLMResponse(
                content="Hello",
                provider_metadata={"response_headers": {"nvcf-status": "fulfilled"}},
                usage=UsageInfo(input_tokens=10, output_tokens=5, total_tokens=15),
            ),
        )

        result = await iorails.generate_async(_USER, options={})

        assert isinstance(result, GenerationResponse)
        assert result.llm_metadata == {
            "response_headers": {"nvcf-status": "fulfilled"},
            "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        }

    @pytest.mark.asyncio
    async def test_usage_only_when_no_provider_metadata(self, iorails):
        """With usage but no provider_metadata, ``llm_metadata`` holds just the ``usage`` sub-key."""
        _stub_safe_rails(iorails)
        _stub_model(
            iorails,
            LLMResponse(content="Hello", usage=UsageInfo(input_tokens=3, output_tokens=4, total_tokens=7)),
        )

        result = await iorails.generate_async(_USER, options={})

        assert isinstance(result, GenerationResponse)
        assert result.llm_metadata == {"usage": {"input_tokens": 3, "output_tokens": 4, "total_tokens": 7}}

    @pytest.mark.asyncio
    async def test_usage_includes_reasoning_and_cached_tokens_when_present(self, iorails):
        """Non-None ``reasoning_tokens``/``cached_tokens`` appear in the ``usage`` sub-key."""
        _stub_safe_rails(iorails)
        _stub_model(
            iorails,
            LLMResponse(
                content="Hello",
                usage=UsageInfo(input_tokens=10, output_tokens=5, total_tokens=15, reasoning_tokens=3, cached_tokens=2),
            ),
        )

        result = await iorails.generate_async(_USER, options={})

        assert isinstance(result, GenerationResponse)
        assert result.llm_metadata == {
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
                "reasoning_tokens": 3,
                "cached_tokens": 2,
            }
        }

    @pytest.mark.asyncio
    async def test_no_metadata_or_usage_is_none(self, iorails):
        """No provider_metadata and no usage leaves ``llm_metadata`` as None."""
        _stub_safe_rails(iorails)
        _stub_model(iorails, LLMResponse(content="Hello"))

        result = await iorails.generate_async(_USER, options={})

        assert isinstance(result, GenerationResponse)
        assert result.llm_metadata is None


class TestLLMOutput:
    """``llm_output`` is always None, matching LLMRails' unwired ``raw_response``."""

    @pytest.mark.asyncio
    async def test_llm_output_none_even_when_requested(self, iorails):
        """``options={"llm_output": True}`` is accepted but the field stays None."""
        _stub_safe_rails(iorails)
        _stub_model(
            iorails,
            LLMResponse(content="Hello", provider_metadata={"response_headers": {"nvcf-status": "fulfilled"}}),
        )

        result = await iorails.generate_async(_USER, options={"llm_output": True})

        assert isinstance(result, GenerationResponse)
        assert result.llm_output is None


class TestUnsupportedOptionGuards:
    """Colang-coupled options IORails cannot honor raise ValueError."""

    @pytest.mark.asyncio
    async def test_output_vars_true_raises(self, iorails):
        """``output_vars=True`` requests Colang context IORails has no access to."""
        _stub_safe_rails(iorails)
        _stub_model(iorails, LLMResponse(content="Hello"))

        with pytest.raises(ValueError):
            await iorails.generate_async(_USER, options={"output_vars": True})

    @pytest.mark.asyncio
    async def test_output_vars_list_raises(self, iorails):
        """A list of ``output_vars`` also raises."""
        _stub_safe_rails(iorails)
        _stub_model(iorails, LLMResponse(content="Hello"))

        with pytest.raises(ValueError):
            await iorails.generate_async(_USER, options={"output_vars": ["relevant_chunks"]})

    @pytest.mark.asyncio
    async def test_log_request_flag_raises(self, iorails):
        """Requesting any ``log`` detail raises NotImplementedError while ``log`` is deferred."""
        _stub_safe_rails(iorails)
        _stub_model(iorails, LLMResponse(content="Hello"))

        with pytest.raises(NotImplementedError):
            await iorails.generate_async(_USER, options={"log": {"llm_calls": True}})

    @pytest.mark.asyncio
    async def test_state_argument_raises(self, iorails):
        """A ``state`` argument is unsupported for the stateless IORails engine."""
        _stub_safe_rails(iorails)
        _stub_model(iorails, LLMResponse(content="Hello"))

        with pytest.raises(ValueError):
            await iorails.generate_async(_USER, options={}, state={"conversation": []})


class TestBlockedStructuredResponse:
    """A blocked request returns the refusal in ``response`` with the other fields empty."""

    @pytest.mark.asyncio
    async def test_input_block_returns_refusal_response(self, iorails):
        """Input-rail block yields a GenerationResponse whose response is the refusal message."""
        iorails.rails_manager.is_input_safe = AsyncMock(return_value=RailResult(is_safe=False, reason="unsafe"))
        iorails.rails_manager.is_output_safe = AsyncMock(return_value=RailResult(is_safe=True))
        _stub_model(iorails, LLMResponse(content="unused"))

        result = await iorails.generate_async(_USER, options={})

        assert isinstance(result, GenerationResponse)
        assert result.response == [{"role": "assistant", "content": REFUSAL_MESSAGE}]
        assert result.tool_calls is None
        assert result.reasoning_content is None
        assert result.llm_metadata is None

    @pytest.mark.asyncio
    async def test_output_block_returns_refusal_response(self, iorails):
        """Output-rail block yields a GenerationResponse whose response is the refusal message."""
        iorails.rails_manager.is_input_safe = AsyncMock(return_value=RailResult(is_safe=True))
        iorails.rails_manager.is_output_safe = AsyncMock(return_value=RailResult(is_safe=False, reason="unsafe"))
        _stub_model(iorails, LLMResponse(content="bad answer"))

        result = await iorails.generate_async(_USER, options={})

        assert isinstance(result, GenerationResponse)
        assert result.response == [{"role": "assistant", "content": REFUSAL_MESSAGE}]


class TestBarePathUnchanged:
    """The optionless bare-dict path keeps its existing behavior."""

    @pytest.mark.asyncio
    async def test_bare_path_inlines_reasoning_prefix(self, iorails):
        """Without ``options`` reasoning is still delivered inline as a <think> prefix."""
        _stub_safe_rails(iorails)
        _stub_model(iorails, LLMResponse(content="Hello", reasoning="thinking step"))

        result = await iorails.generate_async(_USER)

        assert result == {"role": "assistant", "content": "<think>thinking step</think>\nHello"}


class TestSyncGenerateStructured:
    """The synchronous ``generate`` mirrors the async structured return."""

    def test_sync_generate_with_options_returns_generation_response(self, iorails_sync):
        """``generate(options=...)`` returns a GenerationResponse from the ephemeral engine."""
        iorails_sync.rails_manager.is_input_safe = AsyncMock(return_value=RailResult(is_safe=True))
        iorails_sync.rails_manager.is_output_safe = AsyncMock(return_value=RailResult(is_safe=True))
        iorails_sync.engine_registry.model_call = AsyncMock(return_value=LLMResponse(content="Hello"))

        with patch("nemoguardrails.guardrails.iorails.IORails", return_value=iorails_sync):
            result = iorails_sync.generate(_USER, options={})

        assert isinstance(result, GenerationResponse)
