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

import copy
import json
from unittest.mock import AsyncMock

import pytest

from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.rails.llm.options import GenerationResponse
from nemoguardrails.testing.fake_model import FakeLLMModel
from nemoguardrails.types import LLMResponse, ToolCall, ToolCallFunction


def assistant_message(response):
    """Extract the first assistant message from a GenerationResponse or return the plain response."""
    if isinstance(response, GenerationResponse):
        assert isinstance(response.response, list)
        return response.response[0]
    return response


@pytest.mark.asyncio
@pytest.mark.parametrize("options", [None, {"llm_params": {"temperature": 0}}])
async def test_cached_history_preserves_updated_context_and_accepts_new_context(options):
    """Preserve evolved context on cache reuse while allowing an explicit new context message to override it."""
    config = RailsConfig.from_content(
        yaml_content="""
models: []
rails:
  input:
    flows: [count turns]
  output:
    flows: [return count]
""",
        colang_content="""
define flow count turns
  $counter = $counter + 1

define flow return count
  $bot_message = execute format_counter(counter=$counter)
""",
    )
    rails = LLMRails(config, llm=FakeLLMModel(responses=["Answer"] * 3))

    async def format_counter(counter):
        """Expose the current rail-context counter as response text."""
        return str(counter)

    rails.register_action(format_counter, name="format_counter")
    messages = [{"role": "context", "content": {"counter": 0}}, {"role": "user", "content": "Hello"}]
    first = assistant_message(await rails.generate_async(messages=messages, options=options))
    assert first == {"role": "assistant", "content": "1"}
    messages.extend([first, {"role": "user", "content": "Hello again"}])
    second = assistant_message(await rails.generate_async(messages=messages, options=options))
    assert second == {"role": "assistant", "content": "2"}
    messages.extend([second, {"role": "context", "content": {"counter": 10}}, {"role": "user", "content": "Reset"}])
    third = assistant_message(await rails.generate_async(messages=messages, options=options))
    assert third == {"role": "assistant", "content": "11"}
    assert messages[0]["content"] == {"counter": 0}


@pytest.mark.asyncio
@pytest.mark.parametrize("safe_options", [None, {"rails": {"input": True}}])
async def test_cached_history_isolated_from_input_rail_bypass(safe_options):
    """Prevent a request that bypasses input rails from replacing validated cached history."""
    config = RailsConfig.from_content(
        yaml_content="""
models: []
passthrough: false
rails:
  input:
    flows: [sanitize]
""",
        colang_content="""
define flow sanitize
  if $user_message == "PRIVATE"
    $user_message = "REDACTED"
""",
    )
    model = FakeLLMModel(responses=["OK", "OK", "Final"])
    model.generate_async = AsyncMock(wraps=model.generate_async)
    rails = LLMRails(config, llm=model)
    messages = [{"role": "user", "content": "PRIVATE"}]
    first = assistant_message(await rails.generate_async(messages=messages, options=safe_options))
    bypassed = assistant_message(await rails.generate_async(messages=messages, options={"rails": {"input": False}}))
    assert first == bypassed
    assert "PRIVATE" in str(model.generate_async.await_args.args[0])
    await rails.generate_async(
        messages=messages + [first, {"role": "user", "content": "Continue"}], options=safe_options
    )
    prompt = str(model.generate_async.await_args.args[0])
    assert "PRIVATE" not in prompt
    assert "REDACTED" in prompt


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_turn", [False, True])
async def test_passthrough_preserves_sanitized_history_without_mutating_messages(tool_turn):
    """Keep sanitized history across ordinary and tool turns without modifying caller-owned messages."""
    config = RailsConfig.from_content(
        yaml_content="""
models: []
passthrough: true
rails:
  input:
    flows: [sanitize]
""",
        colang_content="""
define flow sanitize
  if $user_message == "PRIVATE"
    $user_message = "REDACTED"
""",
    )
    first_response = (
        LLMResponse(
            content="",
            tool_calls=[ToolCall(id="call_weather", function=ToolCallFunction(name="weather", arguments={}))],
        )
        if tool_turn
        else LLMResponse(content="First answer")
    )
    model = FakeLLMModel(
        llm_responses=[first_response, LLMResponse(content="Second answer"), LLMResponse(content="Third answer")]
    )
    model.generate_async = AsyncMock(wraps=model.generate_async)
    rails = LLMRails(config, llm=model)
    messages = [{"role": "system", "content": "Instructions"}, {"role": "user", "content": "PRIVATE"}]
    original = copy.deepcopy(messages)
    first = await rails.generate_async(messages=messages)
    assert messages == original
    next_message = (
        {"role": "tool", "tool_call_id": "call_weather", "content": "Sunny"}
        if tool_turn
        else {"role": "user", "content": "Continue"}
    )
    messages.extend([first, next_message])
    original = copy.deepcopy(messages)
    second = await rails.generate_async(messages=messages)
    assert messages == original
    messages.extend([second, {"role": "user", "content": "PRIVATE"}])
    original = copy.deepcopy(messages)
    await rails.generate_async(messages=messages)
    assert messages == original
    for call in model.generate_async.await_args_list:
        sent = [message.to_dict() for message in call.args[0]]
        assert sent[0] == {"role": "system", "content": "Instructions"}
        assert sent[1] == {"role": "user", "content": "REDACTED"}
        assert "PRIVATE" not in json.dumps(sent)
    sent = [message.to_dict() for message in model.generate_async.await_args.args[0]]
    assert sent[-1] == {"role": "user", "content": "REDACTED"}
    if tool_turn:
        assert sent[2] == first
        assert sent[3] == next_message


@pytest.mark.asyncio
async def test_passthrough_sanitized_history_stays_aligned_across_blocked_turns():
    """Retain blocked-turn placeholders so later sanitized user messages stay aligned with their turns."""
    config = RailsConfig.from_content(
        yaml_content="""
models: []
passthrough: true
rails:
  input:
    flows: [sanitize]
""",
        colang_content="""
define flow sanitize
  if $user_message == "PRIVATE"
    $user_message = "REDACTED"
  if $user_message == "BLOCK"
    bot refuse request
    abort

define bot refuse request
  "Request rejected."
""",
    )
    model = FakeLLMModel(responses=["Answer"] * 3)
    model.generate_async = AsyncMock(wraps=model.generate_async)
    rails = LLMRails(config, llm=model)
    messages = []
    for content in ["PRIVATE", "BLOCK", "PRIVATE", "Continue"]:
        messages.append({"role": "user", "content": content})
        original = copy.deepcopy(messages)
        response = await rails.generate_async(messages=messages)
        assert messages == original
        messages.append(response)
    sent = [message.to_dict() for message in model.generate_async.await_args.args[0]]
    assert [message["content"] for message in sent if message["role"] == "user"] == [
        "REDACTED",
        "[User message blocked by guardrails.]",
        "REDACTED",
        "Continue",
    ]
    assert model.inference_count == 3
