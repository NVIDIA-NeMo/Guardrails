# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from nemoguardrails import LLMRails
from nemoguardrails.actions import action
from nemoguardrails.exceptions import StreamingNotSupportedError
from nemoguardrails.types import LLMResponseChunk, ToolCall, ToolCallFunction
from tests.recorded.assertions import (
    assert_blocked_stream_error,
    assert_llm_call_usage,
    assert_no_stream_error,
    assert_runtime_model_matches,
    assert_stream_contract,
)
from tests.recorded.cassette import recorded_chat_response
from tests.recorded.normalization import normalize_stream_chunks
from tests.recorded.rails.public_api.configs import (
    INPUT_RAIL_STREAMING_CONFIG,
    NIM_BASELINE_CONFIG,
    OPENAI_BASELINE_CONFIG,
    OPENAI_MODEL,
    STREAMING_DISABLED_CONFIG,
    STREAMING_OUTPUT_RAILS_CONFIG,
    STREAMING_PASSTHROUGH_CONFIG,
)
from tests.recorded.rails_config import load_config
from tests.recorded.snapshots import snapshot
from tests.utils import FakeLLMModel

pytestmark = [pytest.mark.recorded, pytest.mark.asyncio]


async def _chunks(values: list[str]) -> AsyncIterator[str]:
    for value in values:
        yield value


class _ToolCallStreamingModel(FakeLLMModel):
    async def stream_async(self, prompt, *, stop=None, **kwargs):
        yield LLMResponseChunk(
            delta_tool_calls=[
                ToolCall(
                    id="call_weather",
                    function=ToolCallFunction(name="get_weather", arguments={"city": "Paris"}),
                )
            ],
            finish_reason="tool_calls",
        )


async def test_include_generation_metadata_matches_include_metadata():
    """A3.3: the deprecated metadata flag remains an exact alias for ``include_metadata``."""
    messages = [{"role": "user", "content": "hi"}]
    current = LLMRails(
        load_config(STREAMING_PASSTHROUGH_CONFIG),
        llm=FakeLLMModel(responses=["Hello world"]),
        verbose=False,
    )
    deprecated = LLMRails(
        load_config(STREAMING_PASSTHROUGH_CONFIG),
        llm=FakeLLMModel(responses=["Hello world"]),
        verbose=False,
    )

    current_chunks = [chunk async for chunk in current.stream_async(messages=messages, include_metadata=True)]
    with pytest.warns(DeprecationWarning, match="include_generation_metadata is deprecated"):
        deprecated_chunks = [
            chunk async for chunk in deprecated.stream_async(messages=messages, include_generation_metadata=True)
        ]

    assert deprecated_chunks == current_chunks
    assert current_chunks
    assert all(isinstance(chunk, dict) for chunk in current_chunks)


async def test_streaming_output_rail_allows_then_blocks_at_buffer_boundary():
    """A3.7: a later blocked buffer stops the stream after the already allowed prefix."""

    @action(is_system_action=True, output_mapping=lambda result: not result)
    def block_marker(context=None, **params):
        return "BLOCK" not in (context or {}).get("bot_message", "")

    rails = LLMRails(load_config(STREAMING_OUTPUT_RAILS_CONFIG), verbose=False)
    rails.register_action(block_marker, "self_check_streaming_output")

    chunks = [
        chunk
        async for chunk in rails.stream_async(
            messages=[{"role": "user", "content": "stream"}],
            generator=_chunks(["safe ", "BLOCK", "tail"]),
        )
    ]

    assert chunks[0] == "safe "
    assert len(chunks) == 2
    assert_blocked_stream_error(chunks)


@pytest.mark.parametrize(
    ("chunk_size", "context_size", "stream_first", "expected_checks"),
    [
        (1, 1, False, ["a", "ab", "bc", "c"]),
        (2, 1, False, ["ab", "bc", "c"]),
        (2, 2, False, ["ab", "abc", "bc"]),
        (2, 1, True, ["ab", "bc", "c"]),
    ],
)
async def test_streaming_output_rail_buffer_configuration(
    chunk_size,
    context_size,
    stream_first,
    expected_checks,
):
    """A3.8: chunk, context, and stream-first settings control observable buffering."""
    checks = []
    trace = []

    @action(is_system_action=True, output_mapping=lambda result: not result)
    def capture_buffer(context=None, **params):
        checks.append((context or {}).get("bot_message", ""))
        trace.append("check")
        return True

    async def source():
        for value in ["a", "b", "c"]:
            trace.append(f"source:{value}")
            yield value

    config = load_config(STREAMING_OUTPUT_RAILS_CONFIG)
    config.rails.output.streaming.chunk_size = chunk_size
    config.rails.output.streaming.context_size = context_size
    config.rails.output.streaming.stream_first = stream_first
    rails = LLMRails(config, verbose=False)
    rails.register_action(capture_buffer, "self_check_streaming_output")

    output = []
    async for chunk in rails.stream_async(
        messages=[{"role": "user", "content": "stream"}],
        generator=source(),
    ):
        output.append(chunk)
        trace.append(f"yield:{chunk}")

    assert output == ["a", "b", "c"]
    assert checks == expected_checks
    if stream_first:
        assert trace.index("yield:a") < trace.index("check")
    else:
        assert trace.index("check") < trace.index("yield:a")


async def test_stream_async_tool_call_deltas_are_not_surfaced():
    """A3.11: legacy ``LLMRails.stream_async`` currently drops accumulated tool-call deltas."""
    rails = LLMRails(
        load_config(STREAMING_PASSTHROUGH_CONFIG),
        llm=_ToolCallStreamingModel(responses=[""]),
        verbose=False,
    )

    chunks = [chunk async for chunk in rails.stream_async(messages=[{"role": "user", "content": "weather"}])]

    assert chunks
    assert all(chunk == "" for chunk in chunks)


async def test_input_rail_blocks_before_stream_generation():
    """A3.12: a blocking input rail prevents the streaming model call from starting."""
    model = FakeLLMModel(responses=["must not run"])
    rails = LLMRails(load_config(INPUT_RAIL_STREAMING_CONFIG), llm=model, verbose=False)

    chunks = [chunk async for chunk in rails.stream_async(messages=[{"role": "user", "content": "block input"}])]

    assert chunks == ["I'm sorry, I can't respond to that."]
    assert model.inference_count == 0


@pytest.mark.vcr
async def test_openai_stream_async_public_contract(openai_api_key):
    rails = LLMRails(load_config(OPENAI_BASELINE_CONFIG), verbose=False)

    chunks = []
    async for chunk in rails.stream_async(prompt="Say hello in a few words."):
        chunks.append(chunk)

    assert_stream_contract(chunks, expect_multiple=False)
    assert_no_stream_error(chunks)
    assert normalize_stream_chunks(chunks) == snapshot(
        {"content": "Hello there! 👋", "chunks": ["", "Hello", " there", "!", " 👋", "", ""], "errors": []}
    )


@pytest.mark.vcr
async def test_nim_stream_async_public_contract(nvidia_api_key):
    rails = LLMRails(load_config(NIM_BASELINE_CONFIG), verbose=False)

    chunks = []
    async for chunk in rails.stream_async(messages=[{"role": "user", "content": "Say hello in a few words."}]):
        chunks.append(chunk)

    assert_stream_contract(chunks, expect_multiple=False)
    assert_no_stream_error(chunks)
    assert normalize_stream_chunks(chunks) == snapshot(
        {
            "content": "Hello! 😊",
            "chunks": [
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "Hello! 😊",
                "",
            ],
            "errors": [],
        }
    )


@pytest.mark.vcr
async def test_stream_async_matches_recorded_chat_completion_metadata(
    openai_api_key, record_mode, recorded_cassette_path
):
    rails = LLMRails(load_config(OPENAI_BASELINE_CONFIG), verbose=False)

    chunks = []
    async for chunk in rails.stream_async(prompt="Say hello in a few words.", include_metadata=True):
        assert isinstance(chunk, dict)
        chunks.append(chunk)

    assert chunks
    content = "".join(chunk["text"] for chunk in chunks if isinstance(chunk.get("text"), str))
    assert content.strip()

    if record_mode == "none":
        expected = recorded_chat_response(
            recorded_cassette_path,
            request_model=OPENAI_MODEL,
            stream=True,
        )
        assert expected.raw_usage is not None
        assert expected.finish_reason == "stop"
        assert expected.request_id
        assert content == expected.content

        usage_chunks = [chunk for chunk in chunks if chunk.get("metadata", {}).get("usage")]
        assert len(usage_chunks) == 1
        assert usage_chunks[0]["metadata"]["usage"] == {
            "input_tokens": expected.usage["input_tokens"],
            "output_tokens": expected.usage["output_tokens"],
            "total_tokens": expected.usage["total_tokens"],
        }

        llm_calls = rails.explain().llm_calls
        assert len(llm_calls) == 1
        llm_call = llm_calls[0]
        assert llm_call.completion == expected.content
        assert llm_call.llm_provider_name == "openai"
        assert_llm_call_usage(llm_call, expected)
        assert_runtime_model_matches(llm_call, configured_model=OPENAI_MODEL, recorded_model=expected.model)

    assert normalize_stream_chunks(chunks) == snapshot(
        {
            "content": "Hello there! 👋",
            "chunks": [
                {"text": ""},
                {"text": "Hello"},
                {"text": " there"},
                {"text": "!"},
                {"text": " 👋"},
                {"text": ""},
                {"text": "", "usage": {"input_tokens": 13, "output_tokens": 8, "total_tokens": 21}},
                {"text": ""},
            ],
            "errors": [],
        }
    )


async def test_streaming_output_rails_allowed():
    rails = LLMRails(load_config(STREAMING_OUTPUT_RAILS_CONFIG), verbose=False)

    chunks = []
    async for chunk in rails.stream_async(
        messages=[{"role": "user", "content": "stream"}],
        generator=_chunks(["Hello", " ", "there"]),
    ):
        chunks.append(chunk)

    assert_stream_contract(chunks)
    assert_no_stream_error(chunks)
    assert normalize_stream_chunks(chunks) == snapshot(
        {"content": "Hello there", "chunks": ["Hello", " ", "there"], "errors": []}
    )


async def test_streaming_output_rails_blocked():
    rails = LLMRails(load_config(STREAMING_OUTPUT_RAILS_CONFIG), verbose=False)

    chunks = []
    async for chunk in rails.stream_async(
        messages=[{"role": "user", "content": "stream"}],
        generator=_chunks(["BLOCK"]),
    ):
        chunks.append(chunk)

    assert_blocked_stream_error(chunks)
    assert normalize_stream_chunks(chunks) == snapshot(
        {
            "content": "",
            "chunks": [
                '{"error": {"message": "Blocked by streaming output rail rails.", "type": "guardrails_violation", "param": "streaming output rail", "code": "content_blocked"}}'
            ],
            "errors": [
                {
                    "error": {
                        "message": "Blocked by streaming output rail rails.",
                        "type": "guardrails_violation",
                        "param": "streaming output rail",
                        "code": "content_blocked",
                    }
                }
            ],
        }
    )


async def test_streaming_output_rails_disabled_validation():
    rails = LLMRails(load_config(STREAMING_DISABLED_CONFIG), verbose=False)

    with pytest.raises(StreamingNotSupportedError):
        async for _ in rails.stream_async(
            messages=[{"role": "user", "content": "stream"}],
            generator=_chunks(["Hello"]),
        ):
            pass


@pytest.mark.vcr
async def test_nim_stream_async_reasoning_not_inlined_in_streamed_text(
    nvidia_api_key, record_mode, recorded_cassette_path
):
    """A3.10: while streaming, the NIM provider sends ``reasoning_content`` deltas, but the
    user-facing streamed text is the answer only, the reasoning is NOT inlined as a ``<think>``
    block the way it is in non-streaming ``generate`` (see ``test_nim_generate_async_public_contract``).

    KNOWN LIMITATION flagged for follow-up: a streaming consumer cannot access the model's
    reasoning at all, ``reasoning_content`` arrives in the deltas (parsed into
    ``LLMResponseChunk.delta_reasoning``) but is dropped rather than surfaced (IORails drops it
    explicitly; the standard StreamingHandler has no reasoning channel). Non-streaming ``generate``
    does expose it, so this is an asymmetry. This test pins the *current* behavior so the
    decomposition is held to it; surfacing reasoning while streaming is a separate fix.
    """
    rails = LLMRails(load_config(NIM_BASELINE_CONFIG), verbose=False)

    chunks = []
    async for chunk in rails.stream_async(messages=[{"role": "user", "content": "Say hello in one short sentence."}]):
        chunks.append(chunk)

    assert_no_stream_error(chunks)
    content = normalize_stream_chunks(chunks)["content"]
    assert "<think>" not in content
    if record_mode == "none":
        assert "reasoning_content" in recorded_cassette_path.read_text(encoding="utf-8")
    assert normalize_stream_chunks(chunks) == snapshot(
        {
            "content": "Hello!",
            "chunks": [
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "Hello",
                "!",
                "",
                "",
            ],
            "errors": [],
        }
    )
