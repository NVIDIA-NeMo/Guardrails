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

"""Unit tests for IORails streaming support."""

import asyncio
import json
import warnings
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from nemoguardrails.exceptions import StreamingNotSupportedError
from nemoguardrails.guardrails.guardrails_types import RailResult
from nemoguardrails.guardrails.iorails import (
    REFUSAL_MESSAGE,
    STREAM_MAX_CONCURRENCY,
    IORails,
    _is_stream_error_chunk,
)
from nemoguardrails.guardrails.model_engine import ModelEngine
from nemoguardrails.rails.llm.config import RailsConfig
from nemoguardrails.rails.llm.options import GenerationOptions
from nemoguardrails.types import LLMResponseChunk, ToolCall, ToolCallFunction
from tests.guardrails.async_helpers import started_iorails
from tests.guardrails.test_data import NEMOGUARDS_CONFIG


def _make_streaming_config(*, enabled: bool = True, stream_first: bool = True) -> dict:
    """Build a NEMOGUARDS_CONFIG variant with output-rail streaming settings."""
    return {
        **NEMOGUARDS_CONFIG,
        "rails": {
            **NEMOGUARDS_CONFIG["rails"],
            "output": {
                **NEMOGUARDS_CONFIG["rails"]["output"],
                "streaming": {
                    "enabled": enabled,
                    "chunk_size": 5,
                    "context_size": 2,
                    "stream_first": stream_first,
                },
            },
        },
    }


_INPUT_ONLY_CONFIG = {
    **NEMOGUARDS_CONFIG,
    "rails": {
        **NEMOGUARDS_CONFIG["rails"],
        "output": {"flows": []},
    },
}

_INPUT_ONLY_SPECULATIVE_CONFIG = {
    **_INPUT_ONLY_CONFIG,
    "rails": {
        **_INPUT_ONLY_CONFIG["rails"],
        "input": {
            **_INPUT_ONLY_CONFIG["rails"]["input"],
            "speculative_generation": True,
        },
    },
}

_SPECULATIVE_STREAM_WARNING = "speculative_generation is not supported for streaming; falling back to sequential"


async def _mock_stream(model_type, messages, **kwargs):
    """Async generator simulating streaming chunks from the main LLM."""
    for text in ["Hello", " from", " the", " streaming", " LLM", "!", " Have", " a", " nice", " day"]:
        yield LLMResponseChunk(delta_content=text)


async def _collect(async_iter):
    """Collect all chunks from an async iterator into a list."""
    return [chunk async for chunk in async_iter]


async def _failing_stream(model_type, messages, **kwargs):
    """Mock stream that raises immediately."""
    raise RuntimeError("LLM exploded")
    yield  # noqa: unreachable -- makes this an async generator


async def _mid_stream_failure(model_type, messages, **kwargs):
    """Mock stream that yields some chunks then raises."""
    yield LLMResponseChunk(delta_content="Hello")
    yield LLMResponseChunk(delta_content=" world")
    raise RuntimeError("connection lost")


def _assert_error_chunk(chunks, *, code, message_contains):
    """Assert that chunks contain exactly one error JSON with the given code and message substring."""
    error_chunks = [c for c in chunks if isinstance(c, str) and c.startswith("{")]
    assert len(error_chunks) >= 1, f"Expected error chunk, got none in {chunks}"
    error_data = json.loads(error_chunks[0])
    assert error_data["error"]["code"] == code
    assert message_contains in error_data["error"]["message"]


def _wire_mocks(iorails, *, input_safe=True, output_safe=True, stream=_mock_stream):
    """Attach standard mocks for input rails, output rails, and LLM streaming."""
    iorails.rails_manager.is_input_safe = AsyncMock(
        return_value=RailResult(is_safe=input_safe, reason=None if input_safe else "blocked")
    )
    iorails.rails_manager.is_output_safe = AsyncMock(
        return_value=RailResult(is_safe=output_safe, reason=None if output_safe else "blocked")
    )
    iorails.engine_registry.stream_model_call = stream


@pytest_asyncio.fixture
async def iorails():
    """IORails with output rails but streaming NOT enabled."""
    async with started_iorails(NEMOGUARDS_CONFIG) as iorails:
        yield iorails


@pytest_asyncio.fixture
async def iorails_stream_first():
    """IORails with output rails and streaming enabled (stream_first=True)."""
    async with started_iorails(_make_streaming_config(stream_first=True)) as iorails:
        yield iorails


@pytest_asyncio.fixture
async def iorails_stream_check_first():
    """IORails with output rails and streaming enabled (stream_first=False)."""
    async with started_iorails(_make_streaming_config(stream_first=False)) as iorails:
        yield iorails


@pytest_asyncio.fixture
async def iorails_input_only():
    """IORails with input rails only, no output rails."""
    async with started_iorails(_INPUT_ONLY_CONFIG) as iorails:
        yield iorails


class TestStreamAsyncValidation:
    """Test that stream_async raises when output rails exist but streaming is disabled."""

    @pytest.mark.asyncio
    async def test_raises_when_output_rails_without_streaming(self, iorails):
        """Raises StreamingNotSupportedError when output rails exist but streaming is disabled."""
        with pytest.raises(StreamingNotSupportedError):
            iorails.stream_async(messages=[{"role": "user", "content": "hi"}])

    @pytest.mark.asyncio
    async def test_no_error_when_no_output_rails(self, iorails_input_only):
        """Succeeds when there are no output rails at all."""
        _wire_mocks(iorails_input_only)
        chunks = await _collect(iorails_input_only.stream_async(messages=[{"role": "user", "content": "hi"}]))
        assert len(chunks) > 0

    @pytest.mark.asyncio
    async def test_no_error_when_streaming_enabled(self, iorails_stream_first):
        """Succeeds when output rails have streaming enabled."""
        _wire_mocks(iorails_stream_first)
        chunks = await _collect(iorails_stream_first.stream_async(messages=[{"role": "user", "content": "hi"}]))
        assert len(chunks) > 0

    @pytest.mark.asyncio
    async def test_raises_when_include_metadata_with_output_rails_streaming(self, iorails_stream_first):
        """include_metadata=True is rejected when output rails streaming is enabled."""
        with pytest.raises(ValueError, match="include_metadata=True is not supported"):
            iorails_stream_first.stream_async(
                messages=[{"role": "user", "content": "hi"}],
                include_metadata=True,
            )

    @pytest.mark.asyncio
    async def test_include_metadata_allowed_without_output_rails(self, iorails_input_only):
        """include_metadata=True is fine when there are no output rails."""
        _wire_mocks(iorails_input_only)
        chunks = await _collect(
            iorails_input_only.stream_async(messages=[{"role": "user", "content": "hi"}], include_metadata=True)
        )
        assert len(chunks) > 0

    @pytest.mark.asyncio
    async def test_speculative_generation_streaming_warning_recorded_once(self):
        """Default warning filtering records the speculative streaming warning once per call site."""
        with patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"}):
            iorails = IORails(RailsConfig.from_content(config=_INPUT_ONLY_SPECULATIVE_CONFIG))

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("default")
            for _ in range(2):
                iorails.stream_async(messages=[{"role": "user", "content": "hi"}])

        matching_warnings = [warning for warning in caught if str(warning.message) == _SPECULATIVE_STREAM_WARNING]
        assert len(matching_warnings) == 1

    @pytest.mark.asyncio
    async def test_tools_in_llm_params_forwarded_on_stream_async(self, iorails_input_only):
        """Tool definitions in llm_params are forwarded to the streaming LLM call unchanged.

        Streaming tool-call delta parsing is deferred to a later PR; the request-side
        forwarding works today because llm_params passes through untouched.
        """
        captured_kwargs = {}

        async def capturing_stream(model_type, messages, **kwargs):
            """Mock stream that records kwargs."""
            captured_kwargs.update(kwargs)
            yield LLMResponseChunk(delta_content="ok")

        _wire_mocks(iorails_input_only, stream=capturing_stream)
        tool = {"type": "function", "function": {"name": "get_weather"}}
        options = GenerationOptions(llm_params={"tools": [tool], "tool_choice": "auto"})

        chunks = await _collect(
            iorails_input_only.stream_async(messages=[{"role": "user", "content": "hi"}], options=options)
        )

        assert "".join(chunks) == "ok"
        assert captured_kwargs.get("tools") == [tool]
        assert captured_kwargs.get("tool_choice") == "auto"


class TestStreamAsyncNoOutputRails:
    """Test streaming when there are no output rails -- chunks flow straight through."""

    @pytest.mark.asyncio
    async def test_streams_all_chunks(self, iorails_input_only):
        """All LLM chunks are yielded to the caller."""
        _wire_mocks(iorails_input_only)
        chunks = await _collect(iorails_input_only.stream_async(messages=[{"role": "user", "content": "hi"}]))
        assert "".join(chunks) == "Hello from the streaming LLM! Have a nice day"

    @pytest.mark.asyncio
    async def test_input_rails_block(self, iorails_input_only):
        """Yields the refusal message when input rails block."""
        _wire_mocks(iorails_input_only, input_safe=False)
        chunks = await _collect(iorails_input_only.stream_async(messages=[{"role": "user", "content": "bad"}]))
        assert "".join(chunks) == REFUSAL_MESSAGE

    @pytest.mark.asyncio
    async def test_generation_options_forwarded(self, iorails_input_only):
        """llm_params from GenerationOptions are forwarded to the LLM call."""
        captured_kwargs = {}

        async def capturing_stream(model_type, messages, **kwargs):
            """Mock stream that records kwargs."""
            captured_kwargs.update(kwargs)
            yield LLMResponseChunk(delta_content="ok")

        _wire_mocks(iorails_input_only, stream=capturing_stream)
        options = GenerationOptions(llm_params={"temperature": 0.42})
        chunks = await _collect(
            iorails_input_only.stream_async(messages=[{"role": "user", "content": "hi"}], options=options)
        )
        assert "".join(chunks) == "ok"
        assert captured_kwargs.get("temperature") == 0.42

    @pytest.mark.asyncio
    async def test_dict_options_forwarded(self, iorails_input_only):
        """Dict options are converted to GenerationOptions and forwarded."""
        captured_kwargs = {}

        async def capturing_stream(model_type, messages, **kwargs):
            """Mock stream that records kwargs."""
            captured_kwargs.update(kwargs)
            yield LLMResponseChunk(delta_content="ok")

        _wire_mocks(iorails_input_only, stream=capturing_stream)
        chunks = await _collect(
            iorails_input_only.stream_async(
                messages=[{"role": "user", "content": "hi"}],
                options={"llm_params": {"temperature": 0.42}},
            )
        )
        assert "".join(chunks) == "ok"
        assert captured_kwargs.get("temperature") == 0.42


class TestStreamAsyncOutputRailsStreamFirst:
    """Test streaming with output rails in stream_first=True mode (optimistic)."""

    @pytest.mark.asyncio
    async def test_safe_output_streams_all(self, iorails_stream_first):
        """All chunks are streamed when output rails pass."""
        _wire_mocks(iorails_stream_first)
        chunks = await _collect(iorails_stream_first.stream_async(messages=[{"role": "user", "content": "hi"}]))
        text = "".join(c for c in chunks if not c.startswith("{"))
        assert "Hello from the streaming LLM" in text

    @pytest.mark.asyncio
    async def test_unsafe_output_injects_error(self, iorails_stream_first):
        """Error JSON is injected into the stream when output rails block."""
        _wire_mocks(iorails_stream_first, output_safe=False)
        chunks = await _collect(iorails_stream_first.stream_async(messages=[{"role": "user", "content": "hi"}]))
        _assert_error_chunk(chunks, code="content_blocked", message_contains="Blocked by output rails")

        error_chunks = [c for c in chunks if isinstance(c, str) and c.startswith("{")]
        assert len(error_chunks) >= 1
        error_data = json.loads(error_chunks[0])
        assert error_data["error"]["type"] == "guardrails_violation"
        assert error_data["error"]["code"] == "content_blocked"

    @pytest.mark.asyncio
    async def test_stream_first_yields_before_rail_check(self, iorails_stream_first):
        """Chunks appear before the output rail check in stream_first mode."""
        yield_order = []

        async def tracking_rail(messages, response):
            """Mock output rail that records call order."""
            yield_order.append("rail_check")
            return RailResult(is_safe=True)

        _wire_mocks(iorails_stream_first)
        iorails_stream_first.rails_manager.is_output_safe = tracking_rail

        async for chunk in iorails_stream_first.stream_async(messages=[{"role": "user", "content": "hi"}]):
            if not chunk.startswith("{"):
                yield_order.append(f"chunk:{chunk}")

        first_chunk_idx = next(i for i, v in enumerate(yield_order) if v.startswith("chunk:"))
        first_rail_idx = next(i for i, v in enumerate(yield_order) if v == "rail_check")
        assert first_chunk_idx < first_rail_idx


class TestStreamAsyncOutputRailsGated:
    """Test streaming with output rails in stream_first=False mode (gated)."""

    @pytest.mark.asyncio
    async def test_safe_output_streams_all(self, iorails_stream_check_first):
        """All chunks are eventually yielded when output rails pass."""
        _wire_mocks(iorails_stream_check_first)
        chunks = await _collect(iorails_stream_check_first.stream_async(messages=[{"role": "user", "content": "hi"}]))
        text = "".join(c for c in chunks if not c.startswith("{"))
        assert "Hello from the streaming LLM" in text

    @pytest.mark.asyncio
    async def test_unsafe_output_yields_nothing_then_error(self, iorails_stream_check_first):
        """No content chunks appear before the error in gated mode."""
        _wire_mocks(iorails_stream_check_first, output_safe=False)
        chunks = await _collect(iorails_stream_check_first.stream_async(messages=[{"role": "user", "content": "hi"}]))

        content_chunks = [c for c in chunks if isinstance(c, str) and not c.startswith("{")]
        error_chunks = [c for c in chunks if isinstance(c, str) and c.startswith("{")]
        assert len(content_chunks) == 0
        _assert_error_chunk(chunks, code="content_blocked", message_contains="Blocked by output rails")

    @pytest.mark.asyncio
    async def test_gated_yields_after_rail_check(self, iorails_stream_check_first):
        """Each chunk batch only appears after its rail check passes."""
        yield_order = []

        async def tracking_rail(messages, response):
            """Mock output rail that records call order."""
            yield_order.append("rail_check")
            return RailResult(is_safe=True)

        _wire_mocks(iorails_stream_check_first)
        iorails_stream_check_first.rails_manager.is_output_safe = tracking_rail

        async for chunk in iorails_stream_check_first.stream_async(messages=[{"role": "user", "content": "hi"}]):
            if not chunk.startswith("{"):
                yield_order.append(f"chunk:{chunk}")

        assert len([v for v in yield_order if v == "rail_check"]) > 0
        assert len([v for v in yield_order if v.startswith("chunk:")]) > 0
        assert yield_order[0] == "rail_check"


class TestStreamAsyncErrors:
    """Test error propagation during streaming."""

    @pytest.mark.asyncio
    async def test_generation_error_yields_error_json(self, iorails_input_only):
        """LLM exceptions are surfaced as error JSON chunks."""
        _wire_mocks(iorails_input_only, stream=_failing_stream)
        chunks = await _collect(iorails_input_only.stream_async(messages=[{"role": "user", "content": "hi"}]))
        _assert_error_chunk(chunks, code="generation_failed", message_contains="LLM exploded")

    @pytest.mark.asyncio
    async def test_generation_error_with_output_rails(self, iorails_stream_first):
        """LLM exceptions propagate as error JSON even with output rails active."""
        _wire_mocks(iorails_stream_first, stream=_failing_stream)
        chunks = await _collect(iorails_stream_first.stream_async(messages=[{"role": "user", "content": "hi"}]))
        _assert_error_chunk(chunks, code="generation_failed", message_contains="LLM exploded")

    @pytest.mark.asyncio
    async def test_output_rail_exception_propagates(self, iorails_stream_first):
        """Exception in is_output_safe propagates out of the stream."""
        _wire_mocks(iorails_stream_first)
        iorails_stream_first.rails_manager.is_output_safe = AsyncMock(side_effect=RuntimeError("rail crashed"))

        with pytest.raises(RuntimeError, match="rail crashed"):
            await _collect(iorails_stream_first.stream_async(messages=[{"role": "user", "content": "hi"}]))

    @pytest.mark.asyncio
    async def test_mid_stream_failure_yields_partial_then_error(self, iorails_input_only):
        """A failure after some successful chunks yields partial output then error JSON."""
        _wire_mocks(iorails_input_only, stream=_mid_stream_failure)
        chunks = await _collect(iorails_input_only.stream_async(messages=[{"role": "user", "content": "hi"}]))

        content_chunks = [c for c in chunks if not c.startswith("{")]
        assert "".join(content_chunks) == "Hello world"
        _assert_error_chunk(chunks, code="generation_failed", message_contains="connection lost")

    @pytest.mark.asyncio
    async def test_mid_stream_failure_with_output_rails(self, iorails_stream_first):
        """Mid-stream failure with output rails active still surfaces the error (stream_first)."""
        _wire_mocks(iorails_stream_first, stream=_mid_stream_failure)
        chunks = await _collect(iorails_stream_first.stream_async(messages=[{"role": "user", "content": "hi"}]))
        _assert_error_chunk(chunks, code="generation_failed", message_contains="connection lost")

    @pytest.mark.asyncio
    async def test_generation_error_bypasses_output_rails_gated(self, iorails_stream_check_first):
        """In stream_first=False, generation errors bypass output rails instead of being checked."""
        _wire_mocks(iorails_stream_check_first, stream=_failing_stream)
        # Output rail would block if the error JSON were fed through it
        iorails_stream_check_first.rails_manager.is_output_safe = AsyncMock(
            return_value=RailResult(is_safe=False, reason="blocked")
        )

        chunks = await _collect(iorails_stream_check_first.stream_async(messages=[{"role": "user", "content": "hi"}]))

        # Should see generation_failed, NOT content_blocked
        _assert_error_chunk(chunks, code="generation_failed", message_contains="LLM exploded")

    @pytest.mark.asyncio
    async def test_mid_stream_error_bypasses_output_rails_gated(self, iorails_stream_check_first):
        """In stream_first=False, mid-stream errors bypass output rails."""
        _wire_mocks(iorails_stream_check_first, stream=_mid_stream_failure)

        chunks = await _collect(iorails_stream_check_first.stream_async(messages=[{"role": "user", "content": "hi"}]))

        # The error chunk should come through as generation_failed
        _assert_error_chunk(chunks, code="generation_failed", message_contains="connection lost")


class TestStreamAsyncConcurrency:
    """Test the streaming semaphore for concurrency control."""

    @pytest.mark.asyncio
    async def test_semaphore_exhaustion_raises(self, iorails_input_only):
        """Raises QueueFull when all streaming slots are taken."""
        _wire_mocks(iorails_input_only)
        iorails_input_only._stream_semaphore = asyncio.Semaphore(0)

        with pytest.raises(asyncio.QueueFull, match="Streaming concurrency limit reached"):
            await anext(iorails_input_only.stream_async(messages=[{"role": "user", "content": "hi"}]))

    @pytest.mark.asyncio
    async def test_semaphore_released_after_stream(self, iorails_input_only):
        """Semaphore slot is released after the stream is fully consumed."""
        _wire_mocks(iorails_input_only)
        iorails_input_only._stream_semaphore = asyncio.Semaphore(1)

        await _collect(iorails_input_only.stream_async(messages=[{"role": "user", "content": "hi"}]))
        assert iorails_input_only._stream_semaphore._value == 1

    @pytest.mark.asyncio
    async def test_background_task_cancelled_on_early_exit(self, iorails_input_only):
        """When the consumer breaks early, the background generation task is cancelled."""
        task_started = asyncio.Event()

        async def slow_stream(model_type, messages, **kwargs):
            """Mock stream that yields many chunks to allow early exit testing."""
            task_started.set()
            for i in range(1000):
                yield LLMResponseChunk(delta_content=f"chunk{i}")
                await asyncio.sleep(0)  # yield control so cancellation can propagate

        _wire_mocks(iorails_input_only, stream=slow_stream)

        async for _ in iorails_input_only.stream_async(messages=[{"role": "user", "content": "hi"}]):
            if task_started.is_set():
                break  # consumer exits early

        # Give the event loop a few ticks to process cancellation and cleanup
        for _ in range(5):
            await asyncio.sleep(0)

        # Semaphore should be released even after early exit
        assert iorails_input_only._stream_semaphore._value == STREAM_MAX_CONCURRENCY


def _build_sse_streaming_mock(deltas):
    """Build an aiohttp-like response mock that yields SSE lines for the given deltas.

    Each delta is a dict like ``{"content": "Hi"}`` or ``{"reasoning_content": "thinking"}``
    that becomes a ``data: {"choices":[{"delta": <delta>}]}\\n`` line. Always terminates
    with ``data: [DONE]``.
    """
    lines = []
    for delta in deltas:
        payload = json.dumps({"choices": [{"delta": delta}]})
        lines.append(f"data: {payload}\n".encode())
    lines.append(b"data: [DONE]\n")

    line_iter = iter(lines)

    async def _readline():
        return next(line_iter, b"")

    mock_content = MagicMock()
    mock_content.readline = _readline

    mock_response = AsyncMock()
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.status = 200
    mock_response.content = mock_content

    mock_client = AsyncMock()
    mock_client.post = MagicMock(return_value=mock_response)
    return mock_client


class TestStreamAsyncEndToEnd:
    """Full chain: raw SSE bytes -> ModelEngine.stream_call._parse_chat_completion_chunk
    -> LLMResponseChunk -> EngineRegistry.stream_model_call -> IORails._generation_task pushes
    chunk.delta_content -> caller sees text chunks. Mocks at the HTTP boundary so the
    SSE-parsing path is exercised on the way through.
    """

    @pytest.mark.asyncio
    async def test_content_chunks_full_chain(self, iorails_input_only):
        iorails_input_only.rails_manager.is_input_safe = AsyncMock(return_value=RailResult(is_safe=True))

        main_engine = iorails_input_only.engine_registry._get_engine("main", ModelEngine)
        main_engine._client = _build_sse_streaming_mock([{"content": "Hello"}, {"content": " "}, {"content": "world"}])
        main_engine._running = True

        chunks = await _collect(iorails_input_only.stream_async(messages=[{"role": "user", "content": "hi"}]))

        assert "".join(chunks) == "Hello world"

    @pytest.mark.asyncio
    async def test_reasoning_deltas_dropped_from_caller_output(self, iorails_input_only):
        """Reasoning deltas pass through ModelEngine as LLMResponseChunk.delta_reasoning,
        but IORails drops them — the caller only sees content text.
        """
        iorails_input_only.rails_manager.is_input_safe = AsyncMock(return_value=RailResult(is_safe=True))

        main_engine = iorails_input_only.engine_registry._get_engine("main", ModelEngine)
        main_engine._client = _build_sse_streaming_mock(
            [
                {"reasoning_content": "let me think"},
                {"content": "The"},
                {"reasoning_content": " more thinking"},
                {"content": " answer"},
                {"content": " is 42"},
            ]
        )
        main_engine._running = True

        chunks = await _collect(iorails_input_only.stream_async(messages=[{"role": "user", "content": "hi"}]))

        assert "".join(chunks) == "The answer is 42"
        assert "think" not in "".join(chunks)


def _build_tool_call_sse_mock(*, content_deltas=None, tool_call_chunks, finish_chunk=None):
    """Build an aiohttp mock that streams optional content then tool-call SSE chunks.

    ``tool_call_chunks`` is a list of raw SSE body dicts (already shaped as
    ``{"choices": [{"delta": {"tool_calls": [...]}, "finish_reason": ...}]}``)
    appended after any ``content_deltas``.  ``finish_chunk`` is an optional
    explicit terminal chunk dict; defaults to a bare ``finish_reason`` chunk
    when omitted (OpenAI style).
    """
    lines = []
    for text in content_deltas or []:
        payload = json.dumps({"choices": [{"delta": {"content": text}}]})
        lines.append(f"data: {payload}\n".encode())
    for chunk_body in tool_call_chunks:
        lines.append(f"data: {json.dumps(chunk_body)}\n".encode())
    if finish_chunk is not None:
        lines.append(f"data: {json.dumps(finish_chunk)}\n".encode())
    lines.append(b"data: [DONE]\n")

    line_iter = iter(lines)

    async def _readline():
        return next(line_iter, b"")

    mock_content = MagicMock()
    mock_content.readline = _readline
    mock_response = AsyncMock()
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.status = 200
    mock_response.content = mock_content
    mock_client = AsyncMock()
    mock_client.post = MagicMock(return_value=mock_response)
    return mock_client


# Single NIM-style chunk: complete args on the finish_reason chunk.
_NIM_TOOL_CALL_CHUNK = {
    "choices": [
        {
            "delta": {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_nim",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                    }
                ]
            },
            "finish_reason": "tool_calls",
        }
    ]
}


class TestStreamAsyncToolCalling:
    """Tool calling in the IORails streaming path."""

    @pytest.mark.asyncio
    async def test_tools_in_llm_params_forwarded_to_stream_model_call(self, iorails_input_only):
        """Tools provided via llm_params are forwarded unchanged to stream_model_call."""
        tool = {"type": "function", "function": {"name": "get_weather", "parameters": {}}}
        options = GenerationOptions(llm_params={"tools": [tool], "tool_choice": "auto"})

        captured_kwargs: dict = {}

        async def _capturing_stream(model_type, messages, **kwargs):
            captured_kwargs.update(kwargs)
            yield LLMResponseChunk(delta_content="ok")

        iorails_input_only.engine_registry.stream_model_call = _capturing_stream
        iorails_input_only.rails_manager.is_input_safe = AsyncMock(return_value=RailResult(is_safe=True, reason=None))

        await _collect(
            iorails_input_only.stream_async(
                messages=[{"role": "user", "content": "hi"}],
                options=options,
            )
        )

        assert captured_kwargs.get("tools") == [tool]
        assert captured_kwargs.get("tool_choice") == "auto"

    @pytest.mark.asyncio
    async def test_tool_call_only_yields_terminal_json(self, iorails_input_only):
        """A tool-call-only response yields a single terminal JSON chunk containing tool_calls."""
        iorails_input_only.rails_manager.is_input_safe = AsyncMock(return_value=RailResult(is_safe=True))
        main_engine = iorails_input_only.engine_registry._get_engine("main", ModelEngine)
        main_engine._client = _build_tool_call_sse_mock(tool_call_chunks=[_NIM_TOOL_CALL_CHUNK])
        main_engine._running = True

        chunks = await _collect(iorails_input_only.stream_async(messages=[{"role": "user", "content": "hi"}]))

        assert len(chunks) == 1
        parsed = json.loads(chunks[0])
        assert "tool_calls" in parsed
        assert parsed["tool_calls"][0]["function"]["name"] == "get_weather"
        assert parsed["tool_calls"][0]["function"]["arguments"] == '{"city": "Paris"}'

    @pytest.mark.asyncio
    async def test_text_and_tool_calls_text_first_then_terminal_json(self, iorails_input_only):
        """Content chunks come first; terminal tool-call JSON follows after all text."""
        iorails_input_only.rails_manager.is_input_safe = AsyncMock(return_value=RailResult(is_safe=True))
        main_engine = iorails_input_only.engine_registry._get_engine("main", ModelEngine)
        main_engine._client = _build_tool_call_sse_mock(
            content_deltas=["Checking", " weather"],
            tool_call_chunks=[_NIM_TOOL_CALL_CHUNK],
        )
        main_engine._running = True

        chunks = await _collect(iorails_input_only.stream_async(messages=[{"role": "user", "content": "hi"}]))

        tool_json_chunks = [c for c in chunks if isinstance(c, str) and '"tool_calls"' in c]
        text_chunks = [c for c in chunks if c not in tool_json_chunks]
        assert "".join(text_chunks) == "Checking weather"
        assert len(tool_json_chunks) == 1
        assert json.loads(tool_json_chunks[0])["tool_calls"][0]["function"]["name"] == "get_weather"

    @pytest.mark.asyncio
    async def test_fragmented_args_assembled_into_complete_json_string(self, iorails_input_only):
        """OpenAI-style argument fragments are concatenated and parsed into a complete JSON string."""
        iorails_input_only.rails_manager.is_input_safe = AsyncMock(return_value=RailResult(is_safe=True))
        main_engine = iorails_input_only.engine_registry._get_engine("main", ModelEngine)

        frag1 = {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "c1",
                                "type": "function",
                                "function": {"name": "get_weather", "arguments": ""},
                            }
                        ]
                    }
                }
            ]
        }
        frag2 = {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"city"'}}]}}]}
        frag3 = {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": ': "Paris"}'}}]}}]}
        finish = {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}

        main_engine._client = _build_tool_call_sse_mock(
            tool_call_chunks=[frag1, frag2, frag3],
            finish_chunk=finish,
        )
        main_engine._running = True

        chunks = await _collect(iorails_input_only.stream_async(messages=[{"role": "user", "content": "hi"}]))

        assert len(chunks) == 1
        parsed = json.loads(chunks[0])
        assert parsed["tool_calls"][0]["function"]["arguments"] == '{"city": "Paris"}'

    @pytest.mark.asyncio
    async def test_tool_call_only_does_not_invoke_output_rails(self, iorails_stream_first):
        """Tool-call-only stream skips is_output_safe (no text content to check)."""
        iorails_stream_first.rails_manager.is_input_safe = AsyncMock(return_value=RailResult(is_safe=True))
        iorails_stream_first.rails_manager.is_output_safe = AsyncMock(return_value=RailResult(is_safe=True))
        main_engine = iorails_stream_first.engine_registry._get_engine("main", ModelEngine)
        main_engine._client = _build_tool_call_sse_mock(tool_call_chunks=[_NIM_TOOL_CALL_CHUNK])
        main_engine._running = True

        chunks = await _collect(iorails_stream_first.stream_async(messages=[{"role": "user", "content": "hi"}]))

        iorails_stream_first.rails_manager.is_output_safe.assert_not_called()
        assert any('"tool_calls"' in c for c in chunks if isinstance(c, str))

    @pytest.mark.asyncio
    async def test_forced_tool_choice_finish_reason_stop_surfaces_terminal_json(self, iorails_input_only):
        """Forced tool_choice yields finish_reason='stop'; the tool call must still surface.
        If tool_choice is `auto`, the finish_reason is `tool_calls`
        """
        forced_chunk = {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_forced",
                                "type": "function",
                                "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                            }
                        ]
                    },
                    "finish_reason": "stop",
                }
            ]
        }
        iorails_input_only.rails_manager.is_input_safe = AsyncMock(return_value=RailResult(is_safe=True))
        main_engine = iorails_input_only.engine_registry._get_engine("main", ModelEngine)
        main_engine._client = _build_tool_call_sse_mock(tool_call_chunks=[forced_chunk])
        main_engine._running = True

        chunks = await _collect(iorails_input_only.stream_async(messages=[{"role": "user", "content": "hi"}]))

        assert len(chunks) == 1
        parsed = json.loads(chunks[0])
        assert parsed["tool_calls"][0]["function"]["name"] == "get_weather"
        assert parsed["tool_calls"][0]["function"]["arguments"] == '{"city": "Paris"}'

    @pytest.mark.asyncio
    async def test_model_engine_accumulates_tool_calls(self, iorails_input_only):
        """Ensure ModelEngine.stream_call() accumulates all delta_tool_calls before
        emitting once after the stream ends.
        """
        call_a = ToolCall(id="a", type="function", function=ToolCallFunction(name="fn_a", arguments={"x": 1}))
        call_b = ToolCall(id="b", type="function", function=ToolCallFunction(name="fn_b", arguments={"y": 2}))

        async def _two_emission_stream(model_type, messages, **kwargs):
            # First emission: partial set; second: complete cumulative set.
            yield LLMResponseChunk(delta_tool_calls=[call_a])
            yield LLMResponseChunk(finish_reason="tool_calls", delta_tool_calls=[call_a, call_b])

        iorails_input_only.engine_registry.stream_model_call = _two_emission_stream
        iorails_input_only.rails_manager.is_input_safe = AsyncMock(return_value=RailResult(is_safe=True))

        chunks = await _collect(iorails_input_only.stream_async(messages=[{"role": "user", "content": "hi"}]))

        assert len(chunks) == 1
        parsed = json.loads(chunks[0])
        names = [tc["function"]["name"] for tc in parsed["tool_calls"]]
        assert names == ["fn_a", "fn_b"]

    @pytest.mark.asyncio
    async def test_tool_calls_suppressed_after_output_rails_block(self, iorails_stream_first):
        """A blocked output rail suppresses the terminal tool-call chunk."""
        iorails_stream_first.rails_manager.is_input_safe = AsyncMock(return_value=RailResult(is_safe=True))
        iorails_stream_first.rails_manager.is_output_safe = AsyncMock(
            return_value=RailResult(is_safe=False, reason="blocked")
        )
        main_engine = iorails_stream_first.engine_registry._get_engine("main", ModelEngine)
        main_engine._client = _build_tool_call_sse_mock(
            content_deltas=["Some text"],
            tool_call_chunks=[_NIM_TOOL_CALL_CHUNK],
        )
        main_engine._running = True

        chunks = await _collect(iorails_stream_first.stream_async(messages=[{"role": "user", "content": "hi"}]))

        # The block surfaces a guardrails_violation error chunk ...
        assert any(isinstance(c, str) and "guardrails_violation" in c for c in chunks)
        # ... and the terminal tool-call chunk must NOT follow it.
        assert not any(isinstance(c, str) and '"tool_calls"' in c for c in chunks)

    @pytest.mark.asyncio
    async def test_include_metadata_tool_call_only_yields_dict_frame(self, iorails_input_only):
        """With include_metadata=True the terminal tool-call chunk is a dict frame."""
        iorails_input_only.rails_manager.is_input_safe = AsyncMock(return_value=RailResult(is_safe=True))
        main_engine = iorails_input_only.engine_registry._get_engine("main", ModelEngine)
        main_engine._client = _build_tool_call_sse_mock(tool_call_chunks=[_NIM_TOOL_CALL_CHUNK])
        main_engine._running = True

        chunks = await _collect(
            iorails_input_only.stream_async(messages=[{"role": "user", "content": "hi"}], include_metadata=True)
        )

        # No raw-string tool-call chunk leaked into a metadata stream.
        assert not any(isinstance(c, str) and '"tool_calls"' in c for c in chunks)
        # The tool calls arrive as a dict frame with the payload under "text".
        tool_frames = [c for c in chunks if isinstance(c, dict) and '"tool_calls"' in (c.get("text") or "")]
        assert len(tool_frames) == 1
        parsed = json.loads(tool_frames[0]["text"])
        assert parsed["tool_calls"][0]["function"]["name"] == "get_weather"

    @pytest.mark.asyncio
    async def test_tool_calls_recorded_in_captured_content(self, iorails_input_only):
        """When content capture is on, the terminal tool-call payload is recorded on the span."""
        iorails_input_only._content_capture_enabled = True
        iorails_input_only.rails_manager.is_input_safe = AsyncMock(return_value=RailResult(is_safe=True))
        main_engine = iorails_input_only.engine_registry._get_engine("main", ModelEngine)
        main_engine._client = _build_tool_call_sse_mock(tool_call_chunks=[_NIM_TOOL_CALL_CHUNK])
        main_engine._running = True

        with patch("nemoguardrails.guardrails.iorails.set_request_content") as mock_set:
            await _collect(iorails_input_only.stream_async(messages=[{"role": "user", "content": "hi"}]))

        mock_set.assert_called_once()
        # set_request_content(request_span, messages, output_text)
        output_text = mock_set.call_args.args[2]
        assert output_text is not None and "tool_calls" in output_text

    @pytest.mark.asyncio
    async def test_output_rails_streaming_skips_empty_content_batch(self, iorails_stream_check_first):
        """An empty-content batch skips the is_output_safe check and passes the chunk through.

        Covers the empty-content guard in _run_output_rails_in_streaming on the
        stream_first=False path (e.g. a batch that formats to an empty string).
        """
        iorails_stream_check_first.rails_manager.is_output_safe = AsyncMock(return_value=RailResult(is_safe=True))

        async def _empty_content_handler():
            yield ""  # formats to an empty bot_response_chunk -> guard fires

        out = [
            chunk
            async for chunk in iorails_stream_check_first._run_output_rails_in_streaming(
                streaming_handler=_empty_content_handler(),
                messages=[{"role": "user", "content": "hi"}],
            )
        ]

        iorails_stream_check_first.rails_manager.is_output_safe.assert_not_called()
        assert out == [""]


class TestIsStreamErrorChunk:
    """Unit tests for _is_stream_error_chunk (terminal-chunk error/violation detection)."""

    def test_error_json_string(self):
        assert _is_stream_error_chunk('{"error": {"type": "guardrails_violation"}}') is True

    def test_metadata_frame_with_error(self):
        assert _is_stream_error_chunk({"text": '{"error": {"type": "generation_error"}}'}) is True

    def test_non_error_json_string(self):
        assert _is_stream_error_chunk('{"tool_calls": []}') is False

    def test_plain_text(self):
        assert _is_stream_error_chunk("just some text") is False

    def test_malformed_json_with_error_substring(self):
        # Has the "error" marker but is not valid JSON -> JSONDecodeError branch.
        assert _is_stream_error_chunk('{"error": ') is False

    def test_non_string_text(self):
        assert _is_stream_error_chunk({"text": None}) is False
