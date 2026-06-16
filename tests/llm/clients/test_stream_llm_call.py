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

import pytest

from nemoguardrails.actions.llm.utils import _stream_llm_call
from nemoguardrails.context import llm_response_metadata_var, reasoning_trace_var, tool_calls_var
from nemoguardrails.streaming import StreamingHandler
from nemoguardrails.types import LLMResponse, LLMResponseChunk, ToolCall, ToolCallFunction, UsageInfo


@pytest.fixture(autouse=True)
def reset_context_vars():
    reasoning_token = reasoning_trace_var.set(None)
    tool_calls_token = tool_calls_var.set(None)
    metadata_token = llm_response_metadata_var.set(None)
    yield
    reasoning_trace_var.reset(reasoning_token)
    tool_calls_var.reset(tool_calls_token)
    llm_response_metadata_var.reset(metadata_token)


def _make_chunk_model(chunks):
    class _Model:
        model_name = "test-model"
        provider_name = "test"
        provider_url = None

        async def generate_async(self, prompt, *, stop=None, **kwargs):
            return LLMResponse(content="")

        async def stream_async(self, prompt, *, stop=None, **kwargs):
            for c in chunks:
                yield c

    return _Model()


class TestStreamLlmCallAccumulation:
    @pytest.mark.asyncio
    async def test_accumulates_tool_calls(self):
        tc = [ToolCall(id="call_1", function=ToolCallFunction(name="get_weather", arguments={"city": "Paris"}))]
        model = _make_chunk_model(
            [
                LLMResponseChunk(model="gpt-4o"),
                LLMResponseChunk(delta_tool_calls=tc, finish_reason="tool_calls"),
                LLMResponseChunk(usage=UsageInfo(input_tokens=10, output_tokens=5, total_tokens=15)),
            ]
        )

        result = await _stream_llm_call(model, "test", StreamingHandler(), stop=None)

        assert result.tool_calls == tc
        assert result.model == "gpt-4o"
        assert result.finish_reason == "tool_calls"
        assert result.usage.total_tokens == 15
        assert tool_calls_var.get() is not None

    @pytest.mark.asyncio
    async def test_accumulates_reasoning(self):
        model = _make_chunk_model(
            [
                LLMResponseChunk(delta_reasoning="Let me ", model="gpt-4o"),
                LLMResponseChunk(delta_reasoning="think..."),
                LLMResponseChunk(delta_content="42", finish_reason="stop"),
                LLMResponseChunk(usage=UsageInfo(input_tokens=5, output_tokens=3, total_tokens=8)),
            ]
        )

        result = await _stream_llm_call(model, "test", StreamingHandler(), stop=None)

        assert result.content == "42"
        assert result.reasoning == "Let me think..."
        assert result.model == "gpt-4o"
        assert result.finish_reason == "stop"
        assert reasoning_trace_var.get() == "Let me think..."

    @pytest.mark.asyncio
    async def test_text_only(self):
        model = _make_chunk_model(
            [
                LLMResponseChunk(delta_content="Hello", model="gpt-4o"),
                LLMResponseChunk(delta_content=" world", finish_reason="stop"),
                LLMResponseChunk(usage=UsageInfo(input_tokens=5, output_tokens=2, total_tokens=7)),
            ]
        )

        result = await _stream_llm_call(model, "test", StreamingHandler(), stop=None)

        assert result.content == "Hello world"
        assert result.tool_calls is None
        assert result.reasoning is None
        assert result.model == "gpt-4o"
        assert result.finish_reason == "stop"
        assert result.usage.total_tokens == 7

    @pytest.mark.asyncio
    async def test_request_id_accumulated(self):
        model = _make_chunk_model(
            [
                LLMResponseChunk(delta_content="hi", request_id="req-123", model="gpt-4o"),
                LLMResponseChunk(finish_reason="stop"),
            ]
        )

        result = await _stream_llm_call(model, "test", StreamingHandler(), stop=None)

        assert result.request_id == "req-123"

    @pytest.mark.asyncio
    async def test_clears_tool_calls_var_when_none(self):
        tool_calls_var.set([{"id": "stale"}])

        model = _make_chunk_model(
            [
                LLMResponseChunk(delta_content="no tools", finish_reason="stop"),
            ]
        )

        await _stream_llm_call(model, "test", StreamingHandler(), stop=None)

        assert tool_calls_var.get() is None

    @pytest.mark.asyncio
    async def test_clears_reasoning_var_when_none(self):
        reasoning_trace_var.set("stale")

        model = _make_chunk_model(
            [
                LLMResponseChunk(delta_content="no reasoning", finish_reason="stop"),
            ]
        )

        await _stream_llm_call(model, "test", StreamingHandler(), stop=None)

        assert reasoning_trace_var.get() is None

    @pytest.mark.asyncio
    async def test_provider_metadata_stored_flat(self):
        model = _make_chunk_model(
            [
                LLMResponseChunk(
                    delta_content="hi",
                    provider_metadata={"system_fingerprint": "fp_abc"},
                    finish_reason="stop",
                ),
            ]
        )

        await _stream_llm_call(model, "test", StreamingHandler(), stop=None)

        metadata = llm_response_metadata_var.get()
        assert metadata == {"system_fingerprint": "fp_abc"}

    @pytest.mark.asyncio
    async def test_clears_metadata_var_when_none(self):
        llm_response_metadata_var.set({"stale": True})

        model = _make_chunk_model(
            [
                LLMResponseChunk(delta_content="no metadata", finish_reason="stop"),
            ]
        )

        await _stream_llm_call(model, "test", StreamingHandler(), stop=None)

        assert llm_response_metadata_var.get() is None
