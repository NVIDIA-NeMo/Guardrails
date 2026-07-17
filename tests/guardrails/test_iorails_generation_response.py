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
``tool_calls`` (as ``ToolCall.to_dict()`` with dict arguments), ``log`` (from
per-rail records), and ``llm_metadata`` (main-call ``provider_metadata`` only —
token usage lives in ``log``); ``llm_output`` is always ``None`` (parity with
LLMRails' unwired ``raw_response``). ``output_vars``/``state`` raise ``ValueError``
and ``log.internal_events``/``log.colang_history`` raise ``NotImplementedError``.
"""

from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio

from nemoguardrails.guardrails.guardrails_types import RailResult
from nemoguardrails.guardrails.iorails import REFUSAL_MESSAGE, IORails, _response_content_for_capture
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

_WEATHER_TOOL_CALL = ToolCall(
    id="call_1",
    type="function",
    function=ToolCallFunction(name="get_weather", arguments={"city": "SF"}),
)


async def _generate_structured(iorails: IORails, response: LLMResponse, *, options=None, **kwargs):
    """Stub safe rails + a fixed model response, then run the structured (``options``) path."""
    _stub_safe_rails(iorails)
    _stub_model(iorails, response)
    return await iorails.generate_async(_USER, options={} if options is None else options, **kwargs)


class TestStructuredResponseTrigger:
    """``options`` presence decides GenerationResponse vs. bare dict."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "options",
        [{"llm_params": {"temperature": 0.5}}, GenerationOptions()],
        ids=["dict", "GenerationOptions"],
    )
    async def test_options_returns_generation_response(self, iorails, options):
        """Any ``options`` value (dict or GenerationOptions) switches the return type to GenerationResponse."""
        result = await _generate_structured(iorails, LLMResponse(content="Hello"), options=options)

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
        result = await _generate_structured(iorails, LLMResponse(content="Hello there"))

        assert isinstance(result, GenerationResponse)
        assert result.response == [{"role": "assistant", "content": "Hello there"}]


class TestReasoningContent:
    """Reasoning goes to ``reasoning_content`` with clean content (no inline <think>)."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "response",
        [
            LLMResponse(content="Hello", reasoning="thinking step"),
            LLMResponse(content="<think>thinking step</think>Hello"),
        ],
        ids=["native-reasoning-field", "inline-think-tags"],
    )
    async def test_reasoning_extracted_content_clean(self, iorails, response):
        """Reasoning (native field or inline <think>) goes to ``reasoning_content``; response content stays clean and output rails see only the clean text."""
        result = await _generate_structured(iorails, response)

        assert isinstance(result, GenerationResponse)
        assert result.reasoning_content == "thinking step"
        assert result.response == [{"role": "assistant", "content": "Hello"}]
        iorails.rails_manager.is_output_safe.assert_called_once_with(_USER, "Hello", enabled=True)

    @pytest.mark.asyncio
    async def test_no_reasoning_field_is_none(self, iorails):
        """Absent reasoning leaves ``reasoning_content`` as None."""
        result = await _generate_structured(iorails, LLMResponse(content="plain answer"))

        assert isinstance(result, GenerationResponse)
        assert result.reasoning_content is None
        assert result.response == [{"role": "assistant", "content": "plain answer"}]


class TestToolCalls:
    """``tool_calls`` use the LLMRails ``ToolCall.to_dict()`` shape (dict arguments)."""

    @pytest.mark.asyncio
    async def test_tool_calls_serialized_with_dict_arguments(self, iorails):
        """``tool_calls`` is a list of ``to_dict()`` entries whose ``arguments`` stay a dict, not a JSON string."""
        result = await _generate_structured(iorails, LLMResponse(content="", tool_calls=[_WEATHER_TOOL_CALL]))

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
        result = await _generate_structured(iorails, LLMResponse(content="", tool_calls=[_WEATHER_TOOL_CALL]))

        assert isinstance(result, GenerationResponse)
        assert result.response == [{"role": "assistant", "content": ""}]
        assert "tool_calls" not in result.response[0]

    @pytest.mark.asyncio
    async def test_no_tool_calls_field_is_none(self, iorails):
        """A text-only response leaves ``tool_calls`` as None."""
        result = await _generate_structured(iorails, LLMResponse(content="Hello"))

        assert isinstance(result, GenerationResponse)
        assert result.tool_calls is None


class TestLLMMetadata:
    """``llm_metadata`` is the main-call ``provider_metadata`` verbatim; usage lives in ``log``."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "response, expected",
        [
            (
                LLMResponse(
                    content="Hello",
                    provider_metadata={"response_headers": {"nvcf-status": "fulfilled"}},
                    usage=UsageInfo(input_tokens=10, output_tokens=5, total_tokens=15),
                ),
                {"response_headers": {"nvcf-status": "fulfilled"}},
            ),
            (LLMResponse(content="Hello", usage=UsageInfo(input_tokens=3, output_tokens=4, total_tokens=7)), None),
            (LLMResponse(content="Hello"), None),
        ],
        ids=["provider-metadata-passthrough", "usage-only-none", "nothing-none"],
    )
    async def test_llm_metadata_is_provider_metadata_only(self, iorails, response, expected):
        """llm_metadata is the main-call provider_metadata verbatim (else None); token usage is never grafted under ``usage``."""
        result = await _generate_structured(iorails, response)

        assert isinstance(result, GenerationResponse)
        assert result.llm_metadata == expected


class TestLLMOutput:
    """``llm_output`` is always None, matching LLMRails' unwired ``raw_response``."""

    @pytest.mark.asyncio
    async def test_llm_output_none_even_when_requested(self, iorails):
        """``options={"llm_output": True}`` is accepted but the field stays None."""
        response = LLMResponse(content="Hello", provider_metadata={"response_headers": {"nvcf-status": "fulfilled"}})
        result = await _generate_structured(iorails, response, options={"llm_output": True})

        assert isinstance(result, GenerationResponse)
        assert result.llm_output is None


class TestUnsupportedOptionGuards:
    """Colang-coupled options IORails cannot honor raise."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "gen_kwargs",
        [
            {"options": {"output_vars": True}},
            {"options": {"output_vars": ["relevant_chunks"]}},
            {"options": {}, "state": {"conversation": []}},
        ],
        ids=["output_vars-true", "output_vars-list", "state-arg"],
    )
    async def test_colang_state_option_raises_value_error(self, iorails, gen_kwargs):
        """``output_vars`` (any form) and ``state`` need Colang runtime context IORails lacks — ValueError."""
        _stub_safe_rails(iorails)
        _stub_model(iorails, LLMResponse(content="Hello"))

        with pytest.raises(ValueError):
            await iorails.generate_async(_USER, **gen_kwargs)

    @pytest.mark.asyncio
    async def test_colang_only_log_flag_raises(self, iorails):
        """Colang-runtime-only log details raise NotImplementedError.

        ``activated_rails``/``llm_calls`` are supported and produce a GenerationLog;
        only ``internal_events`` and ``colang_history`` (which need the Colang runtime)
        are rejected.
        """
        _stub_safe_rails(iorails)
        _stub_model(iorails, LLMResponse(content="Hello"))

        with pytest.raises(NotImplementedError):
            await iorails.generate_async(_USER, options={"log": {"internal_events": True}})


class TestBlockedStructuredResponse:
    """A blocked request returns the refusal in ``response`` with the other fields empty."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "input_safe, output_safe", [(False, True), (True, False)], ids=["input-block", "output-block"]
    )
    async def test_block_returns_refusal_response(self, iorails, input_safe, output_safe):
        """A block at either the input or output rail yields a GenerationResponse whose response is the refusal and whose other fields are empty."""
        iorails.rails_manager.is_input_safe = AsyncMock(
            return_value=RailResult(is_safe=input_safe, reason=None if input_safe else "unsafe")
        )
        iorails.rails_manager.is_output_safe = AsyncMock(
            return_value=RailResult(is_safe=output_safe, reason=None if output_safe else "unsafe")
        )
        _stub_model(iorails, LLMResponse(content="bad answer"))

        result = await iorails.generate_async(_USER, options={})

        assert isinstance(result, GenerationResponse)
        assert result.response == [{"role": "assistant", "content": REFUSAL_MESSAGE}]
        assert result.tool_calls is None
        assert result.reasoning_content is None
        assert result.llm_metadata is None


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


class TestResponseContentForCapture:
    """`_response_content_for_capture` extracts assistant text from either return shape."""

    @pytest.mark.parametrize(
        "result, expected",
        [
            (GenerationResponse(response=[{"role": "assistant", "content": "hi there"}]), "hi there"),
            (GenerationResponse(response="hi there"), "hi there"),
            (GenerationResponse(response=[]), None),
            (GenerationResponse(response=[{"role": "assistant", "content": None}]), None),
            ({"role": "assistant", "content": "hi there"}, "hi there"),
        ],
        ids=["structured-list", "structured-str", "empty-list", "non-str-content", "bare-dict"],
    )
    def test_capture_content(self, result, expected):
        """Assistant content is pulled from structured list/str responses and from the bare-dict path; non-str/absent content yields None."""
        assert _response_content_for_capture(result) == expected
