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

"""Reasoning-model support for IORails.

Non-streaming path mirrors LLMRails behavior
- Prefer the provider-native ``LLMResponse.reasoning`` field.
- Fall back to inline ``<think>...</think>`` tags in content (the helper
  strips the tags from ``response.content`` in place so output rails never
  see reasoning).
- Surface reasoning to the caller via a ``<think>{reasoning}</think>\\n``
  prefix on the returned content (LLMRails legacy delivery shape).

Streaming path holds strict LLMRails parity:
- ``delta_reasoning`` is dropped on the floor (LLMRails captures it server-side
  but never delivers it to streaming callers, so neither do we).
- Inline ``<think>`` tags arriving via ``delta_content`` (rather than
  ``delta_reasoning``) pass through verbatim — output rails see them
  unstripped, matching LLMRails' silent leak. We log a single heads-up
  warning per request so operators can spot the leak.
"""

import logging
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from nemoguardrails.guardrails.guardrails_types import RailResult
from nemoguardrails.guardrails.iorails import REFUSAL_MESSAGE, IORails
from nemoguardrails.types import LLMResponse, LLMResponseChunk
from tests.guardrails.async_helpers import started_iorails
from tests.guardrails.test_data import NEMOGUARDS_CONFIG

# stream_async() raises StreamingNotSupportedError only when output rails are
# configured AND rails.output.streaming.enabled is False. Dropping the output
# flows is the simplest way to satisfy the validator here; the warning under
# test fires before END_OF_STREAM regardless of output-rail wiring.
_INPUT_ONLY_CONFIG = {
    **NEMOGUARDS_CONFIG,
    "rails": {**NEMOGUARDS_CONFIG["rails"], "output": {"flows": []}},
}


# Substring identifying the streaming inline-<think> leak warning emitted by
# IORails._generation_task. Kept as a constant so assertions stay in sync if
# the wording shifts.
_INLINE_THINK_WARNING_SUBSTR = "Streamed content contains <think>"


@pytest_asyncio.fixture
async def iorails():
    """Started IORails instance with worker-queue teardown after each test."""
    async with started_iorails(NEMOGUARDS_CONFIG) as iorails:
        yield iorails


@pytest_asyncio.fixture
async def iorails_input_only():
    """Started IORails with input rails only — required for stream_async()."""
    async with started_iorails(_INPUT_ONLY_CONFIG) as iorails:
        yield iorails


@pytest.fixture
def caplog_iorails(caplog):
    """Capture records from ``nemoguardrails.guardrails.iorails`` reliably.

    Attach ``caplog.handler`` directly to the iorails logger and set
    ``propagate=False`` for the duration of the test. The direct attach
    captures regardless of upstream propagation state; the local
    ``propagate=False`` prevents double-capture when propagation IS still
    intact (otherwise we'd record the same warning twice — once via the
    direct handler, once via the propagated chain).
    """
    iorails_logger = logging.getLogger("nemoguardrails.guardrails.iorails")
    original_propagate = iorails_logger.propagate
    iorails_logger.addHandler(caplog.handler)
    iorails_logger.propagate = False
    try:
        yield caplog
    finally:
        iorails_logger.removeHandler(caplog.handler)
        iorails_logger.propagate = original_propagate


def _stub_safe_rails(iorails: IORails) -> None:
    """Default-safe input + output rails so each test focuses on the LLM call."""
    iorails.rails_manager.is_input_safe = AsyncMock(return_value=RailResult(is_safe=True))
    iorails.rails_manager.is_output_safe = AsyncMock(return_value=RailResult(is_safe=True))


def _stub_safe_input(iorails: IORails) -> None:
    """Stub only the input rail (streaming tests on input-only config)."""
    iorails.rails_manager.is_input_safe = AsyncMock(return_value=RailResult(is_safe=True))


def _make_stream(*chunks: LLMResponseChunk):
    """Drop-in for engine_registry.stream_model_call yielding fixed chunks."""

    async def _stream(_model_type, _messages, **_kwargs):
        for c in chunks:
            yield c

    return _stream


async def _collect_stream(async_iter):
    return [c async for c in async_iter]


def _count_inline_think_warnings(caplog) -> int:
    """Number of IORails streaming inline-<think> warnings in caplog."""
    return sum(1 for r in caplog.records if r.levelname == "WARNING" and _INLINE_THINK_WARNING_SUBSTR in r.getMessage())


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


class TestStreamingReasoningWarning:
    """One-shot warning when streamed delta_content leaks <think> reasoning tokens.

    Strict LLMRails parity: reasoning is dropped on the streaming path (LLMRails
    captures it server-side but discards it before reaching the streaming caller),
    and inline ``<think>`` tags arriving via ``delta_content`` pass through
    verbatim. The warning is the only deliberate divergence from LLMRails — it
    gives operators visibility into the leak path without changing chunk content.
    """

    @pytest.mark.asyncio
    async def test_inline_think_warning_fires_once_per_request(self, iorails_input_only, caplog_iorails):
        """Two consecutive streams, both leaking <think> via delta_content → one warning per request.

        A single-inference test can't distinguish the intended "once per request"
        behavior from a buggy "once ever" behavior (e.g. a module-level flag
        suppressing subsequent warnings). Running two leaky requests and asserting
        the warning count climbs 0 → 1 → 2 proves the warning is correctly
        scoped to each request and yielded chunks stay unchanged each time.
        """
        _stub_safe_input(iorails_input_only)
        iorails_input_only.engine_registry.stream_model_call = _make_stream(
            LLMResponseChunk(delta_content="<think>"),
            LLMResponseChunk(delta_content="reasoning step"),
            LLMResponseChunk(delta_content="</think>"),
            LLMResponseChunk(delta_content="Hello"),
        )

        # First request — warning fires.
        chunks_first = await _collect_stream(
            iorails_input_only.stream_async(messages=[{"role": "user", "content": "hi"}])
        )
        assert "".join(chunks_first) == "<think>reasoning step</think>Hello"
        assert _count_inline_think_warnings(caplog_iorails) == 1

        # Second request through the same leaky stream — warning fires again,
        # not deduplicated. Yielded content is unchanged on both requests.
        chunks_second = await _collect_stream(
            iorails_input_only.stream_async(messages=[{"role": "user", "content": "again"}])
        )
        assert "".join(chunks_second) == "<think>reasoning step</think>Hello"
        assert _count_inline_think_warnings(caplog_iorails) == 2

    @pytest.mark.asyncio
    async def test_structured_delta_reasoning_no_warning(self, iorails_input_only, caplog_iorails):
        """Structured delta_reasoning channel → no warning, no <think> tokens leaked into chunks."""
        _stub_safe_input(iorails_input_only)
        iorails_input_only.engine_registry.stream_model_call = _make_stream(
            LLMResponseChunk(delta_reasoning="thinking step"),
            LLMResponseChunk(delta_content="Hello"),
            LLMResponseChunk(delta_content=" world"),
        )

        chunks = await _collect_stream(iorails_input_only.stream_async(messages=[{"role": "user", "content": "hi"}]))

        assert "".join(chunks) == "Hello world"
        assert _count_inline_think_warnings(caplog_iorails) == 0

    @pytest.mark.asyncio
    async def test_clean_content_no_warning(self, iorails_input_only, caplog_iorails):
        """Plain content with no reasoning anywhere → no warning."""
        _stub_safe_input(iorails_input_only)
        iorails_input_only.engine_registry.stream_model_call = _make_stream(
            LLMResponseChunk(delta_content="Hello"),
            LLMResponseChunk(delta_content=" world"),
        )

        chunks = await _collect_stream(iorails_input_only.stream_async(messages=[{"role": "user", "content": "hi"}]))

        assert "".join(chunks) == "Hello world"
        assert _count_inline_think_warnings(caplog_iorails) == 0

    @pytest.mark.asyncio
    async def test_malformed_opener_only_still_warns(self, iorails_input_only, caplog_iorails):
        """Only an opening <think> tag (no closer) → still warns; the leak is the same."""
        _stub_safe_input(iorails_input_only)
        iorails_input_only.engine_registry.stream_model_call = _make_stream(
            LLMResponseChunk(delta_content="<think>incomplete reasoning"),
        )

        chunks = await _collect_stream(iorails_input_only.stream_async(messages=[{"role": "user", "content": "hi"}]))

        assert "".join(chunks) == "<think>incomplete reasoning"
        assert _count_inline_think_warnings(caplog_iorails) == 1

    @pytest.mark.asyncio
    async def test_malformed_closer_only_still_warns(self, iorails_input_only, caplog_iorails):
        """Only a closing </think> tag (no opener) → still warns; the leak is the same."""
        _stub_safe_input(iorails_input_only)
        iorails_input_only.engine_registry.stream_model_call = _make_stream(
            LLMResponseChunk(delta_content="orphan</think> reply"),
        )

        chunks = await _collect_stream(iorails_input_only.stream_async(messages=[{"role": "user", "content": "hi"}]))

        assert "".join(chunks) == "orphan</think> reply"
        assert _count_inline_think_warnings(caplog_iorails) == 1
