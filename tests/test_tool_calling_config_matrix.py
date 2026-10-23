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
from nemoguardrails.testing.fake_model import FakeLLMModel
from nemoguardrails.types import LLMResponse, ToolCall, ToolCallFunction


def weather_call():
    return ToolCall(id="call_weather", function=ToolCallFunction(name="get_weather", arguments={"city": "Paris"}))


def history_with_result(content="Sunny"):
    return [
        {"role": "user", "content": "Weather?"},
        {"role": "assistant", "content": "", "tool_calls": [weather_call().to_dict()]},
        {"role": "tool", "tool_call_id": "call_weather", "name": "get_weather", "content": content},
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("passthrough", [False, True])
@pytest.mark.parametrize("chat_prompt", [False, True])
@pytest.mark.parametrize("policy", ["allow", "sanitize", "block", "disabled"])
async def test_tool_result_rails_control_generation(passthrough, policy, chat_prompt):
    config = RailsConfig.from_content(
        yaml_content=json.dumps(
            {"models": [], "passthrough": passthrough, "rails": {"tool_result": {"flows": ["check result"]}}}
        ),
        colang_content="""
define flow check result
  if $tool_message == "BLOCK"
    bot refuse result
    abort
  if $tool_message == "PRIVATE"
    $tool_message = "REDACTED"

define bot refuse result
  "Tool result rejected."
""",
    )
    if chat_prompt:
        from nemoguardrails.rails.llm.config import TaskPrompt

        config.prompts = [TaskPrompt(task="general", models=["unknown"], messages=["{{ history | to_chat_messages }}"])]
    model = FakeLLMModel(responses=["Final answer.", "Later answer."])
    model.generate_async = AsyncMock(wraps=model.generate_async)
    rails = LLMRails(config, llm=model)
    content = {"allow": "Sunny", "sanitize": "PRIVATE", "block": "BLOCK", "disabled": "BLOCK"}[policy]
    messages = history_with_result(content)
    original = copy.deepcopy(messages)
    result = await rails.generate_async(messages=messages, options={"rails": {"tool_result": policy != "disabled"}})
    if policy == "block":
        assert result.response == [{"role": "assistant", "content": "Tool result rejected."}]
        assert model.inference_count == 0
    else:
        assert result.response == [{"role": "assistant", "content": "Final answer."}]
        assert model.inference_count == 1
        sent = [message.to_dict() for message in model.generate_async.await_args.args[0]]
        assert sent[-1]["role"] == "tool"
        assert sent[-1]["content"] == ("REDACTED" if policy == "sanitize" else content)
    assert messages == original
    messages.extend([result.response[0], {"role": "user", "content": "What did the tool say?"}])
    await rails.generate_async(messages=messages)
    sent = [message.to_dict() for message in model.generate_async.await_args.args[0]]
    returned_content = next(message["content"] for message in sent if message["role"] == "tool")
    assert returned_content == {"block": "[Tool result blocked by guardrails.]", "sanitize": "REDACTED"}.get(
        policy, content
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("passthrough", [False, True])
async def test_tool_result_rails_validate_each_current_batch_with_resolved_names(passthrough):
    config = RailsConfig.from_content(
        yaml_content=json.dumps(
            {"models": [], "passthrough": passthrough, "rails": {"tool_result": {"flows": ["check result"]}}}
        ),
        colang_content="""
define flow check result
  $tool_message = execute check_result(tool_message=$tool_message, tool_name=$tool_name, tool_call_id=$tool_call_id)
""",
    )
    calls = [
        ToolCall(id=city, function=ToolCallFunction(name="weather_" + city, arguments={}))
        for city in ("Paris", "Berlin", "Rome")
    ]
    model = FakeLLMModel(
        llm_responses=[LLMResponse(content="", tool_calls=[calls[2]]), LLMResponse(content="Final answer.")]
    )
    model.generate_async = AsyncMock(wraps=model.generate_async)
    rails = LLMRails(config, llm=model)
    checked = []

    async def check_result(tool_message, tool_name, tool_call_id):
        checked.append((tool_name, tool_call_id))
        return "SAFE " + tool_message

    rails.register_action(check_result, name="check_result")
    messages = [
        {"role": "user", "content": "Compare the cities."},
        {"role": "assistant", "tool_calls": [call.to_dict() for call in calls[:2]]},
        {"role": "tool", "tool_call_id": "Berlin", "content": "Berlin result"},
        {"role": "tool", "tool_call_id": "Paris", "content": "Paris result"},
    ]
    original = copy.deepcopy(messages)
    first = await rails.generate_async(messages=messages)
    assert messages == original
    assert checked == [("weather_Berlin", "Berlin"), ("weather_Paris", "Paris")]
    messages.extend([first, {"role": "tool", "tool_call_id": "Rome", "content": "Rome result"}])
    final = await rails.generate_async(messages=messages)
    assert final == {"role": "assistant", "content": "Final answer."}
    assert checked == [("weather_Berlin", "Berlin"), ("weather_Paris", "Paris"), ("weather_Rome", "Rome")]
    sent = [message.to_dict() for message in model.generate_async.await_args.args[0]]
    assert [(message["tool_call_id"], message["content"]) for message in sent if message["role"] == "tool"] == [
        ("Berlin", "SAFE Berlin result"),
        ("Paris", "SAFE Paris result"),
        ("Rome", "SAFE Rome result"),
    ]
