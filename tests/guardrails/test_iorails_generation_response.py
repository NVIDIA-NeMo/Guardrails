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

from nemoguardrails.actions.rail_outcome import RailOutcome, TransformTarget
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
    iorails.rails_manager.is_input_safe = AsyncMock(return_value=RailResult.allow())
    iorails.rails_manager.is_output_safe = AsyncMock(return_value=RailResult.allow())
    iorails.rails_manager.are_tool_calls_safe = AsyncMock(return_value=RailResult.allow())


def _blocked_unless(safe: bool) -> RailResult:
    """An allow when *safe*, otherwise a block naming the standard test reason."""
    if safe:
        return RailResult.allow()
    return RailResult.block(reason="unsafe")


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
    return await iorails.generate_async(messages=_USER, options={} if options is None else options, **kwargs)


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

        result = await iorails.generate_async(messages=_USER)

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
            await iorails.generate_async(messages=_USER, **gen_kwargs)

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
            await iorails.generate_async(messages=_USER, options={"log": {"internal_events": True}})


class TestBlockedStructuredResponse:
    """A blocked request returns the refusal in ``response`` with the other fields empty."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "input_safe, output_safe", [(False, True), (True, False)], ids=["input-block", "output-block"]
    )
    async def test_block_returns_refusal_response(self, iorails, input_safe, output_safe):
        """A block at either the input or output rail yields a GenerationResponse whose response is the refusal and whose other fields are empty."""
        iorails.rails_manager.is_input_safe = AsyncMock(return_value=_blocked_unless(input_safe))
        iorails.rails_manager.is_output_safe = AsyncMock(return_value=_blocked_unless(output_safe))
        _stub_model(iorails, LLMResponse(content="bad answer"))

        result = await iorails.generate_async(messages=_USER, options={})

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

        result = await iorails.generate_async(messages=_USER)

        assert result == {"role": "assistant", "content": "<think>thinking step</think>\nHello"}


class TestSyncGenerateStructured:
    """The synchronous ``generate`` mirrors the async structured return."""

    def test_sync_generate_with_options_returns_generation_response(self, iorails_sync):
        """``generate(options=...)`` returns a GenerationResponse from the ephemeral engine."""
        iorails_sync.rails_manager.is_input_safe = AsyncMock(return_value=RailResult.allow())
        iorails_sync.rails_manager.is_output_safe = AsyncMock(return_value=RailResult.allow())
        iorails_sync.engine_registry.model_call = AsyncMock(return_value=LLMResponse(content="Hello"))

        with patch("nemoguardrails.guardrails.iorails.IORails", return_value=iorails_sync):
            result = iorails_sync.generate(messages=_USER, options={})

        assert isinstance(result, GenerationResponse)


class TestPromptAndMessages:
    """IORails accepts a keyword-only ``prompt`` (converted to messages) or a ``messages`` list."""

    @pytest.mark.asyncio
    async def test_prompt_string_generates(self, iorails):
        """A keyword prompt string is converted to a user message and generates normally."""
        _stub_safe_rails(iorails)
        _stub_model(iorails, LLMResponse(content="Hi"))

        result = await iorails.generate_async(prompt="Hello", options={})

        assert isinstance(result, GenerationResponse)
        assert result.response == [{"role": "assistant", "content": "Hi"}]

    @pytest.mark.asyncio
    async def test_neither_prompt_nor_messages_raises(self, iorails):
        """generate_async with neither prompt nor messages raises ValueError."""
        with pytest.raises(ValueError):
            await iorails.generate_async()

    def test_sync_generate_neither_raises(self, iorails_sync):
        """The sync generate with neither prompt nor messages raises ValueError."""
        with pytest.raises(ValueError):
            iorails_sync.generate()

    @pytest.mark.asyncio
    async def test_list_passed_positionally_as_prompt_raises_typeerror(self, iorails):
        """A message list in the positional prompt slot raises TypeError, not a silent misparse."""
        with pytest.raises(TypeError):
            await iorails.generate_async([{"role": "user", "content": "hi"}])


class TestConvertToMessages:
    """``IORails._convert_to_messages`` normalizes prompt/messages (moved from the facade)."""

    def test_messages_passthrough(self):
        """A multi-turn message list (system + user + assistant) is returned unchanged."""
        msgs = [
            {"role": "system", "content": "be nice"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        assert IORails._convert_to_messages(messages=msgs) is msgs

    def test_prompt_wrapped_as_user_turn(self):
        """A prompt string (content preserved verbatim) becomes a single user message."""
        prompt = 'Line 1\nLine 2 with "quotes" & symbols'
        assert IORails._convert_to_messages(prompt=prompt) == [{"role": "user", "content": prompt}]

    def test_messages_win_when_both_supplied(self):
        """When both are given, messages takes priority and prompt is ignored."""
        msgs = [{"role": "user", "content": "hi"}]
        assert IORails._convert_to_messages(prompt="ignored", messages=msgs) is msgs

    def test_string_messages_raises_typeerror(self):
        """A string in the messages slot raises TypeError pointing at prompt=."""
        with pytest.raises(TypeError):
            IORails._convert_to_messages(messages="hi")

    def test_list_prompt_raises_typeerror(self):
        """A message list in the prompt slot raises TypeError pointing at messages=."""
        with pytest.raises(TypeError):
            IORails._convert_to_messages(prompt=[{"role": "user", "content": "hi"}])

    def test_empty_string_prompt_raises_valueerror(self):
        """An empty prompt string is falsy, so it counts as 'neither provided'."""
        with pytest.raises(ValueError):
            IORails._convert_to_messages(prompt="")

    def test_empty_messages_list_raises_valueerror(self):
        """An empty messages list is falsy, so it counts as 'neither provided'."""
        with pytest.raises(ValueError):
            IORails._convert_to_messages(messages=[])

    def test_neither_raises_valueerror(self):
        """Neither prompt nor messages raises ValueError."""
        with pytest.raises(ValueError):
            IORails._convert_to_messages()


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


class TestReasoningDroppedWhenOutputRailsChangeTheAnswer:
    """Reasoning ships only when the returned content is still the model's own answer.

    Reasoning bypasses output rails by design, so once a rail blocks or rewrites the
    response the trace no longer describes what the caller receives -- and on a redaction
    rail it still contains the very text the rail removed.
    """

    @pytest.mark.asyncio
    async def test_rewritten_response_drops_reasoning(self, iorails):
        """An output rail that rewrites the answer clears reasoning_content."""
        _stub_safe_rails(iorails)
        iorails.rails_manager.is_output_safe = AsyncMock(
            return_value=RailResult(RailOutcome.transform([(TransformTarget.BOT_MESSAGE, "Call me at [PHONE]")]))
        )
        _stub_model(iorails, LLMResponse(content="Call me at 555-0100", reasoning="the number is 555-0100"))

        result = await iorails.generate_async(messages=_USER, options=GenerationOptions())

        assert result.response == [{"role": "assistant", "content": "Call me at [PHONE]"}]
        assert result.reasoning_content is None

    @pytest.mark.asyncio
    async def test_blocked_response_drops_reasoning(self, iorails):
        """An output rail that blocks returns the refusal with no reasoning attached."""
        _stub_safe_rails(iorails)
        iorails.rails_manager.is_output_safe = AsyncMock(return_value=RailResult.block(reason="unsafe"))
        _stub_model(iorails, LLMResponse(content="unsafe answer", reasoning="the unsafe thinking"))

        result = await iorails.generate_async(messages=_USER, options=GenerationOptions())

        assert result.response == [{"role": "assistant", "content": REFUSAL_MESSAGE}]
        assert result.reasoning_content is None

    @pytest.mark.asyncio
    async def test_untouched_response_keeps_reasoning(self, iorails):
        """Output rails that run and leave the answer alone keep reasoning_content."""
        _stub_safe_rails(iorails)
        _stub_model(iorails, LLMResponse(content="Hello", reasoning="thinking step"))

        result = await iorails.generate_async(messages=_USER, options=GenerationOptions())

        assert result.response == [{"role": "assistant", "content": "Hello"}]
        assert result.reasoning_content == "thinking step"
