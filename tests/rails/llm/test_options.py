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

import pytest

from nemoguardrails.rails.llm.options import (
    GenerationOptions,
    GenerationRailsOptions,
    GenerationResponse,
    Tool,
    ToolCallingOptions,
)


def test_generation_options_initialization():
    """Test GenerationOptions initialization."""

    options = GenerationOptions()
    assert options.rails is not None
    assert isinstance(options.rails, GenerationRailsOptions)
    assert options.rails.input is True
    assert options.rails.output is True
    assert options.rails.retrieval is True
    assert options.rails.dialog is True

    # rails as list
    options = GenerationOptions(rails=["input", "output"])
    assert isinstance(options.rails, GenerationRailsOptions)
    assert options.rails.input is True
    assert options.rails.output is True
    assert options.rails.retrieval is False
    assert options.rails.dialog is False

    # rails as dict
    options = GenerationOptions(rails={"input": True, "output": False})
    assert isinstance(options.rails, GenerationRailsOptions)
    assert options.rails.input is True
    assert options.rails.output is False


def test_generation_options_validation():
    """Test GenerationOptions validation."""

    # valid rails list
    values = {"rails": ["input", "output"]}
    result = GenerationOptions.check_fields(values)
    assert result == values
    assert isinstance(result["rails"], dict)
    assert result["rails"]["input"] is True
    assert result["rails"]["output"] is True
    assert result["rails"]["dialog"] is False
    assert result["rails"]["retrieval"] is False

    # valid rails dict
    values = {"rails": {"input": True, "output": False}}
    result = GenerationOptions.check_fields(values)
    assert result == values
    assert isinstance(result["rails"], dict)
    assert result["rails"]["input"] is True
    assert result["rails"]["output"] is False

    # invalid rails type
    values = {"rails": 123}
    with pytest.raises(ValueError):
        GenerationOptions(**values)


def test_generation_options_log():
    """Test GenerationOptions log handling."""

    options = GenerationOptions(log={"activated_rails": True, "llm_calls": True})
    assert options.log is not None
    assert options.log.activated_rails is True
    assert options.log.llm_calls is True

    with pytest.raises(ValueError):
        GenerationOptions(log="invalid")


def test_generation_options_serialization():
    """Test GenerationOptions serialization."""

    options = GenerationOptions(
        rails={"input": True, "output": False},
        log={"activated_rails": True, "llm_calls": True},
    )

    # serialization to dict
    options_dict = options.dict()
    assert isinstance(options_dict, dict)
    assert "rails" in options_dict
    assert "log" in options_dict
    assert isinstance(options_dict["rails"], dict)
    assert isinstance(options_dict["log"], dict)
    assert options_dict["rails"]["input"] is True
    assert options_dict["rails"]["output"] is False
    assert options_dict["log"]["activated_rails"] is True
    assert options_dict["log"]["llm_calls"] is True

    # serialization to JSON
    options_json = options.json()
    assert isinstance(options_json, str)
    assert '"input":true' in options_json
    assert '"output":false' in options_json
    assert '"activated_rails":true' in options_json
    assert '"llm_calls":true' in options_json


def test_generation_response_initialization():
    """Test GenerationResponse initialization."""
    response = GenerationResponse(response="Hello, world!")
    assert response.response == "Hello, world!"
    assert response.tool_calls is None
    assert response.llm_output is None
    assert response.state is None


def test_generation_response_with_tool_calls():
    test_tool_calls = [
        {
            "name": "get_weather",
            "args": {"location": "NYC"},
            "id": "call_123",
            "type": "tool_call",
        },
        {
            "name": "calculate",
            "args": {"expression": "2+2"},
            "id": "call_456",
            "type": "tool_call",
        },
    ]

    response = GenerationResponse(
        response=[{"role": "assistant", "content": "I'll help you with that."}],
        tool_calls=test_tool_calls,
    )

    assert response.tool_calls == test_tool_calls
    assert len(response.tool_calls) == 2
    assert response.tool_calls[0]["id"] == "call_123"
    assert response.tool_calls[1]["name"] == "calculate"


def test_generation_response_empty_tool_calls():
    """Test GenerationResponse with empty tool calls list."""
    response = GenerationResponse(response="No tools needed", tool_calls=[])

    assert response.tool_calls == []
    assert len(response.tool_calls) == 0


def test_generation_response_serialization_with_tool_calls():
    test_tool_calls = [{"name": "test_func", "args": {}, "id": "call_test", "type": "tool_call"}]

    response = GenerationResponse(response="Response text", tool_calls=test_tool_calls)

    response_dict = response.dict()
    assert "tool_calls" in response_dict
    assert response_dict["tool_calls"] == test_tool_calls

    response_json = response.json()
    assert "tool_calls" in response_json
    assert "test_func" in response_json


def test_generation_response_model_validation():
    """Test GenerationResponse model validation."""
    test_tool_calls = [
        {"name": "valid_function", "args": {}, "id": "call_123", "type": "tool_call"},
        {"name": "another_function", "args": {}, "id": "call_456", "type": "tool_call"},
    ]

    response = GenerationResponse(
        response=[{"role": "assistant", "content": "Test response"}],
        tool_calls=test_tool_calls,
        llm_output={"token_usage": {"total_tokens": 50}},
    )

    assert response.tool_calls is not None
    assert isinstance(response.tool_calls, list)
    assert len(response.tool_calls) == 2
    assert response.llm_output["token_usage"]["total_tokens"] == 50


def test_generation_response_with_reasoning_content():
    test_reasoning = "Step 1: Analyze\nStep 2: Respond"

    response = GenerationResponse(response="Final answer", reasoning_content=test_reasoning)

    assert response.reasoning_content == test_reasoning
    assert response.response == "Final answer"


def test_generation_response_reasoning_content_defaults_to_none():
    response = GenerationResponse(response="Answer")

    assert response.reasoning_content is None


def test_generation_response_reasoning_content_can_be_empty_string():
    response = GenerationResponse(response="Answer", reasoning_content="")

    assert response.reasoning_content == ""


def test_generation_response_serialization_with_reasoning_content():
    test_reasoning = "Thinking process here"

    response = GenerationResponse(response="Response", reasoning_content=test_reasoning)

    response_dict = response.dict()
    assert "reasoning_content" in response_dict
    assert response_dict["reasoning_content"] == test_reasoning

    response_json = response.json()
    assert "reasoning_content" in response_json
    assert test_reasoning in response_json


def test_generation_response_with_all_fields():
    test_tool_calls = [{"name": "test_func", "args": {}, "id": "call_123", "type": "tool_call"}]
    test_reasoning = "Detailed reasoning"

    response = GenerationResponse(
        response=[{"role": "assistant", "content": "Response"}],
        tool_calls=test_tool_calls,
        reasoning_content=test_reasoning,
        llm_output={"token_usage": {"total_tokens": 100}},
    )

    assert response.tool_calls == test_tool_calls
    assert response.reasoning_content == test_reasoning
    assert response.llm_output["token_usage"]["total_tokens"] == 100


def test_tool_calling_options_parses_chat_completions_payload():
    """A full OpenAI /chat/completions tool payload parses into typed models."""
    options = ToolCallingOptions(
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the weather for a city.",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ],
        tool_choice="auto",
        parallel_tool_calls=True,
    )

    assert len(options.tools) == 1
    assert options.tools[0].type == "function"
    assert options.tools[0].function.name == "get_weather"
    assert options.tools[0].function.parameters["required"] == ["city"]
    assert options.tool_choice == "auto"
    assert options.parallel_tool_calls is True


@pytest.mark.parametrize("mode", ["none", "auto", "required"])
def test_tool_calling_options_tool_choice_modes(mode):
    """The string tool_choice modes are accepted."""
    options = ToolCallingOptions(tool_choice=mode)
    assert options.tool_choice == mode


def test_tool_calling_options_named_tool_choice():
    """A named tool choice parses into the nested /chat/completions shape."""
    options = ToolCallingOptions(
        tool_choice={"type": "function", "function": {"name": "get_weather"}},
    )
    assert options.tool_choice.type == "function"
    assert options.tool_choice.function.name == "get_weather"


def test_tool_calling_options_invalid_tool_choice_string_rejected():
    """An unknown tool_choice string is rejected by the typed union."""
    with pytest.raises(ValueError):
        ToolCallingOptions(tool_choice="sometimes")


def test_tool_function_requires_function_definition():
    """A function-type tool without a function definition is rejected."""
    with pytest.raises(ValueError):
        Tool(type="function")


def test_tool_allows_non_function_type_for_responses_forward_compat():
    """Non-function tool types and unknown fields are preserved for future /responses support."""
    tool = Tool.model_validate({"type": "web_search_preview", "search_context_size": "high"})

    assert tool.type == "web_search_preview"
    assert tool.function is None

    dumped = tool.model_dump(exclude_none=True)
    assert dumped["type"] == "web_search_preview"
    assert dumped["search_context_size"] == "high"


def test_tool_calling_options_round_trip_matches_chat_completions_body():
    """model_dump(exclude_none=True) reproduces the /chat/completions request fragment verbatim."""
    payload = {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the weather for a city.",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ],
        "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
        "parallel_tool_calls": False,
    }

    options = ToolCallingOptions(**payload)
    assert options.model_dump(exclude_none=True) == payload


def test_generation_options_tool_calling_defaults_to_none():
    """tool_calling is absent by default so existing behavior is unchanged."""
    options = GenerationOptions()
    assert options.tool_calling is None


def test_generation_options_accepts_tool_calling_from_dict():
    """GenerationOptions parses a nested tool_calling dict into ToolCallingOptions."""
    options = GenerationOptions(
        tool_calling={
            "tools": [{"type": "function", "function": {"name": "lookup"}}],
            "tool_choice": "required",
        }
    )

    assert isinstance(options.tool_calling, ToolCallingOptions)
    assert options.tool_calling.tools[0].function.name == "lookup"
    assert options.tool_calling.tool_choice == "required"
