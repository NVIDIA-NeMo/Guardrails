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

import json
from typing import Any, Optional

import pytest

from nemoguardrails.actions.rail_outcome import RailOutcome
from nemoguardrails.guardrails.tool_schema import Tool, ToolResult
from nemoguardrails.library.tool_safety_check.actions import tool_safety_check_input, tool_safety_check_output
from nemoguardrails.testing.fake_model import FakeLLMModel
from nemoguardrails.types import ToolCall, ToolCallFunction
from tests.guardrails.tool_helpers import WEATHER_SCHEMA, assert_outcome_blocked


class _FakeTaskManager:
    """Records the task/context it was asked to render, and returns a scripted parse result."""

    def __init__(self, parsed: Any = None, has_output_parser: bool = False):
        self.parsed = parsed if parsed is not None else [True]
        self.rendered_task: Optional[str] = None
        self.rendered_context: Optional[dict] = None
        self.forced_output_parser: Optional[str] = None
        self._has_output_parser = has_output_parser

    def render_task_prompt(self, task: Any, context: dict) -> str:
        self.rendered_task = task
        self.rendered_context = context
        return "prompt"

    def get_stop_tokens(self, task: Any) -> list:
        return []

    def get_max_tokens(self, task: Any) -> None:
        return None

    def has_output_parser(self, task: Any) -> bool:
        return self._has_output_parser

    def parse_task_output(self, task: Any, output: str, forced_output_parser: Optional[str] = None) -> Any:
        self.forced_output_parser = forced_output_parser
        return self.parsed


def _weather_call(arguments: dict, call_id: str = "call_1") -> ToolCall:
    return ToolCall(id=call_id, type="function", function=ToolCallFunction(name="get_weather", arguments=arguments))


def _weather_tool() -> Tool:
    return Tool(name="get_weather", arguments_schema=WEATHER_SCHEMA)


@pytest.mark.asyncio
async def test_output_blocks_undeclared_tool_before_llm_call():
    task_manager = _FakeTaskManager()

    outcome = await tool_safety_check_output(
        llms={"llama_guard": FakeLLMModel(responses=[])},
        llm_task_manager=task_manager,
        tool_call=_weather_call({"city": "Paris"}),
        tool_definition=None,
        model_name="llama_guard",
        variant="weather_check",
    )

    assert_outcome_blocked(outcome, "not an allowed tool")
    assert task_manager.rendered_task is None, "the judge LLM must not be called for an undeclared tool"


@pytest.mark.asyncio
async def test_output_blocks_schema_violation_before_llm_call():
    task_manager = _FakeTaskManager()

    outcome = await tool_safety_check_output(
        llms={"llama_guard": FakeLLMModel(responses=[])},
        llm_task_manager=task_manager,
        tool_call=_weather_call({}),  # missing required "city"
        tool_definition=_weather_tool(),
        model_name="llama_guard",
        variant="weather_check",
    )

    assert outcome.is_blocked
    assert task_manager.rendered_task is None, "the judge LLM must not be called for a schema violation"


@pytest.mark.asyncio
async def test_output_missing_model_name_raises_before_llm_call():
    task_manager = _FakeTaskManager()

    with pytest.raises(ValueError, match="Model name is required"):
        await tool_safety_check_output(
            llms={},
            llm_task_manager=task_manager,
            tool_call=_weather_call({"city": "Paris"}),
            tool_definition=_weather_tool(),
            model_name=None,
            variant="weather_check",
        )

    assert task_manager.rendered_task is None


@pytest.mark.asyncio
async def test_output_unknown_model_name_raises():
    task_manager = _FakeTaskManager()

    with pytest.raises(ValueError, match="not found in the list of available models"):
        await tool_safety_check_output(
            llms={},
            llm_task_manager=task_manager,
            tool_call=_weather_call({"city": "Paris"}),
            tool_definition=_weather_tool(),
            model_name="does_not_exist",
            variant="weather_check",
        )


@pytest.mark.asyncio
async def test_output_builds_task_name_from_model_and_variant():
    task_manager = _FakeTaskManager()

    await tool_safety_check_output(
        llms={"llama_guard": FakeLLMModel(responses=["safe"])},
        llm_task_manager=task_manager,
        tool_call=_weather_call({"city": "Paris"}),
        tool_definition=_weather_tool(),
        model_name="llama_guard",
        variant="weather_check",
    )

    assert task_manager.rendered_task == "tool_safety_check_output $model=llama_guard $variant=weather_check"
    assert task_manager.forced_output_parser == "is_content_safe"


@pytest.mark.asyncio
async def test_output_scopes_arguments_to_the_named_field():
    task_manager = _FakeTaskManager()

    await tool_safety_check_output(
        llms={"llama_guard": FakeLLMModel(responses=["safe"])},
        llm_task_manager=task_manager,
        tool_call=_weather_call({"city": "Paris", "units": "metric"}),
        tool_definition=Tool(
            name="get_weather",
            arguments_schema={
                "type": "object",
                "properties": {"city": {"type": "string"}, "units": {"type": "string"}},
            },
        ),
        model_name="llama_guard",
        variant="weather_check",
        argument_name="city",
    )

    assert task_manager.rendered_context is not None
    assert task_manager.rendered_context["tool_call_arguments"] == json.dumps({"city": "Paris"})
    assert task_manager.rendered_context["tool_name"] == "get_weather"


@pytest.mark.asyncio
async def test_output_allows_safe_response():
    task_manager = _FakeTaskManager(parsed=[True])

    outcome = await tool_safety_check_output(
        llms={"llama_guard": FakeLLMModel(responses=["safe"])},
        llm_task_manager=task_manager,
        tool_call=_weather_call({"city": "Paris"}),
        tool_definition=_weather_tool(),
        model_name="llama_guard",
        variant="weather_check",
    )

    assert outcome == RailOutcome.allow(metadata={"violations": []})


@pytest.mark.asyncio
async def test_output_blocks_unsafe_response_with_violations():
    task_manager = _FakeTaskManager(parsed=[False, "leaks credentials"])

    outcome = await tool_safety_check_output(
        llms={"llama_guard": FakeLLMModel(responses=["unsafe: leaks credentials"])},
        llm_task_manager=task_manager,
        tool_call=_weather_call({"city": "Paris"}),
        tool_definition=_weather_tool(),
        model_name="llama_guard",
        variant="weather_check",
    )

    assert outcome == RailOutcome.block(reason="leaks credentials", metadata={"violations": ["leaks credentials"]})


@pytest.mark.asyncio
async def test_output_unregistered_output_parser_fails_closed():
    """An unregistered `output_parser` makes parse_task_output return the raw completion
    string. Unpacking that as [is_safe, *violations] would read its first character as a
    truthy is_safe, silently allowing an "unsafe: ..." verdict -- this must block instead."""
    task_manager = _FakeTaskManager(parsed="unsafe: leaks credentials", has_output_parser=True)

    outcome = await tool_safety_check_output(
        llms={"llama_guard": FakeLLMModel(responses=["unsafe: leaks credentials"])},
        llm_task_manager=task_manager,
        tool_call=_weather_call({"city": "Paris"}),
        tool_definition=_weather_tool(),
        model_name="llama_guard",
        variant="weather_check",
    )

    assert outcome.is_blocked


@pytest.mark.asyncio
async def test_input_missing_model_name_raises():
    task_manager = _FakeTaskManager()

    with pytest.raises(ValueError, match="Model name is required"):
        await tool_safety_check_input(
            llms={},
            llm_task_manager=task_manager,
            tool_call=_weather_call({"city": "Paris"}),
            tool_result=ToolResult(call_id="call_1", name="get_weather", content="18C"),
            model_name=None,
            variant="weather_check",
        )


@pytest.mark.asyncio
async def test_input_builds_task_name_from_model_and_variant():
    task_manager = _FakeTaskManager()

    await tool_safety_check_input(
        llms={"llama_guard": FakeLLMModel(responses=["safe"])},
        llm_task_manager=task_manager,
        tool_call=_weather_call({"city": "Paris"}),
        tool_result=ToolResult(call_id="call_1", name="get_weather", content="18C"),
        model_name="llama_guard",
        variant="weather_check",
    )

    assert task_manager.rendered_task == "tool_safety_check_input $model=llama_guard $variant=weather_check"
    assert task_manager.rendered_context == {"tool_name": "get_weather", "tool_result_content": "18C"}


@pytest.mark.asyncio
async def test_input_serializes_list_content_to_json():
    task_manager = _FakeTaskManager()
    blocks = [{"type": "text", "text": "18C"}]

    await tool_safety_check_input(
        llms={"llama_guard": FakeLLMModel(responses=["safe"])},
        llm_task_manager=task_manager,
        tool_call=_weather_call({"city": "Paris"}),
        tool_result=ToolResult(call_id="call_1", name="get_weather", content=blocks),
        model_name="llama_guard",
        variant="weather_check",
    )

    assert task_manager.rendered_context is not None
    assert task_manager.rendered_context["tool_result_content"] == json.dumps(blocks)


@pytest.mark.asyncio
async def test_input_allows_safe_response():
    task_manager = _FakeTaskManager(parsed=[True])

    outcome = await tool_safety_check_input(
        llms={"llama_guard": FakeLLMModel(responses=["safe"])},
        llm_task_manager=task_manager,
        tool_call=_weather_call({"city": "Paris"}),
        tool_result=ToolResult(call_id="call_1", name="get_weather", content="18C"),
        model_name="llama_guard",
        variant="weather_check",
    )

    assert outcome == RailOutcome.allow(metadata={"violations": []})


@pytest.mark.asyncio
async def test_input_blocks_unsafe_response_with_violations():
    task_manager = _FakeTaskManager(parsed=[False, "contains ssn"])

    outcome = await tool_safety_check_input(
        llms={"llama_guard": FakeLLMModel(responses=["unsafe: contains ssn"])},
        llm_task_manager=task_manager,
        tool_call=_weather_call({"city": "Paris"}),
        tool_result=ToolResult(call_id="call_1", name="get_weather", content="ssn: 123-45-6789"),
        model_name="llama_guard",
        variant="weather_check",
    )

    assert outcome == RailOutcome.block(reason="contains ssn", metadata={"violations": ["contains ssn"]})
