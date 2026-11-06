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

from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.exceptions import StreamingNotSupportedError
from nemoguardrails.streaming import StreamingHandler
from nemoguardrails.testing.fake_model import FakeLLMModel
from nemoguardrails.types import LLMResponse, ToolCall, ToolCallChunk, ToolCallDeltaChunk, ToolCallFunction

pytestmark = pytest.mark.asyncio


def make_rails(passthrough, content="", tool_calls=True):
    config = RailsConfig.from_content(
        config={
            "models": [],
            "passthrough": passthrough,
            "streaming": True,
            "rails": {"tool_call": {"flows": ["reject tool calls"]}},
        },
        colang_content='define flow reject tool calls\n  bot say "Tool call rejected."\n  abort\n',
    )
    calls = [ToolCall(id="call_weather", function=ToolCallFunction("weather", {"city": "Paris"}))]
    return LLMRails(
        config,
        llm=FakeLLMModel(llm_responses=[LLMResponse(content=content, tool_calls=calls if tool_calls else None)]),
    )


@pytest.mark.parametrize("passthrough", [False, True])
@pytest.mark.parametrize("include_metadata", [False, True])
@pytest.mark.parametrize("options", [None, {"rails": {"tool_call": True}}, {"rails": {"tool_call": False}}])
async def test_streamed_calls_require_tool_call_rails_to_be_disabled(passthrough, include_metadata, options):
    rails = make_rails(passthrough)
    emitted = []
    stream = rails.stream_async(prompt="Weather?", options=options, include_metadata=include_metadata)
    if options and options["rails"]["tool_call"] is False:
        emitted = [chunk async for chunk in stream]
        calls = [chunk for chunk in emitted if isinstance(chunk, ToolCallChunk)]
        assert len(calls) == 1
        assert calls[0].tool_calls[0].function.arguments == {"city": "Paris"}
    else:
        with pytest.raises(StreamingNotSupportedError, match="tool-call rails"):
            async for chunk in stream:
                emitted.append(chunk)
        assert not any(isinstance(chunk, (ToolCallChunk, ToolCallDeltaChunk)) for chunk in emitted)


@pytest.mark.parametrize("passthrough", [False, True])
async def test_direct_handler_does_not_release_calls_rejected_by_rails(passthrough):
    rails = make_rails(passthrough)
    handler = StreamingHandler()
    result = await rails.generate_async(messages=[{"role": "user", "content": "Weather?"}], streaming_handler=handler)
    assert not result.get("tool_calls")
    assert result["content"] == "Tool call rejected."
    with pytest.raises(StreamingNotSupportedError, match="tool-call rails"):
        async for chunk in handler:
            assert not isinstance(chunk, (ToolCallChunk, ToolCallDeltaChunk))


@pytest.mark.parametrize("passthrough", [False, True])
async def test_tool_call_rails_do_not_prevent_text_streaming(passthrough):
    rails = make_rails(passthrough, content="Sunny.", tool_calls=False)
    assert "".join([chunk async for chunk in rails.stream_async(prompt="Weather?")]) == "Sunny."
