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

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.actions import action
from nemoguardrails.logging.processing_log import compute_generation_log
from nemoguardrails.rails.llm.options import GenerationResponse

VALIDATED_TOOL_NAMES = []


def print_result(case_name, expected, actual):
    print(f"\n{case_name}")
    print(f"  Expected: {expected}")
    print(f"  Actual:   {actual}")


@action(is_system_action=True)
async def validate_tool_parameters(tool_calls, context=None, **kwargs):
    tool_calls = tool_calls or (context.get("tool_calls", []) if context else [])

    for tool_call in tool_calls:
        tool_name = tool_call.get("function", {}).get("name")
        if tool_name:
            VALIDATED_TOOL_NAMES.append(tool_name)

        args = tool_call.get("function", {}).get("arguments", {})
        for value in args.values():
            if isinstance(value, str) and "eval(" in value:
                return False

    return True


async def validate_dialog_disabled_tool_output_rails():
    config = RailsConfig.from_content(
        """
        define subflow validate tool parameters
          $valid = execute validate_tool_parameters(tool_calls=$tool_calls)

          if not $valid
            bot refuse dangerous tool parameters
            abort

        define bot refuse dangerous tool parameters
          "I cannot execute this tool request because the parameters may be unsafe."
        """,
        """
        models: []
        passthrough: true
        rails:
          tool_output:
            flows:
              - validate tool parameters
        """,
    )
    rails = LLMRails(config)
    rails.runtime.register_action(validate_tool_parameters, name="validate_tool_parameters")
    VALIDATED_TOOL_NAMES.clear()

    result = await rails.generate_async(
        messages=[
            {"role": "user", "content": "Use the requested tool"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_bad",
                        "type": "function",
                        "function": {
                            "name": "dangerous_tool",
                            "arguments": {"param": "eval('malicious code')"},
                        },
                    }
                ],
            },
        ],
        options={"rails": {"dialog": False}},
    )

    assert isinstance(result, GenerationResponse)
    assert isinstance(result.response, list)
    actual_content = result.response[0]["content"]

    print_result(
        "TC-01: Tool output rails with dialog disabled",
        "Unsafe assistant tool call is blocked by the tool_output rail.",
        f"Assistant content: {actual_content!r}",
    )

    assert "parameters may be unsafe" in actual_content
    print("  Result:   PASS")


async def validate_dialog_disabled_text_output_rails_regression():
    config = RailsConfig.from_content(
        """
        define subflow block unsafe bot text
          if $bot_message == "unsafe text"
            bot refuse unsafe text
            abort

        define bot refuse unsafe text
          "The text output was blocked."
        """,
        """
        models: []
        passthrough: true
        rails:
          output:
            flows:
              - block unsafe bot text
        """,
    )
    rails = LLMRails(config)

    result = await rails.generate_async(
        messages=[
            {"role": "user", "content": "Return the final answer"},
            {"role": "assistant", "content": "unsafe text"},
        ],
        options={"rails": {"dialog": False}},
    )

    assert isinstance(result, GenerationResponse)
    assert isinstance(result.response, list)
    actual_content = result.response[0]["content"]

    print_result(
        "TC-02: Plain text output rails regression path",
        "Plain assistant text is still evaluated by output rails when dialog is disabled.",
        f"Assistant content: {actual_content!r}",
    )

    assert actual_content == "The text output was blocked."
    print("  Result:   PASS")


async def validate_output_and_tool_output_rails_together():
    config = RailsConfig.from_content(
        """
        define subflow block empty bot text
          if $bot_message == ""
            bot refuse empty text
            abort

        define bot refuse empty text
          "The empty text output was blocked."

        define subflow validate tool parameters
          $valid = execute validate_tool_parameters(tool_calls=$tool_calls)

          if not $valid
            bot refuse dangerous tool parameters
            abort

        define bot refuse dangerous tool parameters
          "I cannot execute this tool request because the parameters may be unsafe."
        """,
        """
        models: []
        passthrough: true
        rails:
          output:
            flows:
              - block empty bot text
          tool_output:
            flows:
              - validate tool parameters
        """,
    )
    rails = LLMRails(config)
    rails.runtime.register_action(validate_tool_parameters, name="validate_tool_parameters")
    VALIDATED_TOOL_NAMES.clear()

    result = await rails.generate_async(
        messages=[
            {"role": "user", "content": "Use the requested tool"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_bad",
                        "type": "function",
                        "function": {
                            "name": "dangerous_tool",
                            "arguments": {"param": "eval('malicious code')"},
                        },
                    }
                ],
            },
        ],
        options={"rails": {"dialog": False}},
    )

    assert isinstance(result, GenerationResponse)
    assert isinstance(result.response, list)
    actual_content = result.response[0]["content"]

    print_result(
        "TC-03: Output rails and tool_output rails configured together",
        "Assistant tool calls are evaluated by tool_output rails, not blocked as empty text by output rails.",
        f"Assistant content: {actual_content!r}",
    )

    assert "parameters may be unsafe" in actual_content
    assert actual_content != "The empty text output was blocked."
    print("  Result:   PASS")


async def validate_safe_tool_call_with_list_form_rails_option():
    config = RailsConfig.from_content(
        """
        define subflow validate tool parameters
          $valid = execute validate_tool_parameters(tool_calls=$tool_calls)

          if not $valid
            bot refuse dangerous tool parameters
            abort

        define bot refuse dangerous tool parameters
          "I cannot execute this tool request because the parameters may be unsafe."
        """,
        """
        models: []
        passthrough: true
        rails:
          tool_output:
            flows:
              - validate tool parameters
        """,
    )
    rails = LLMRails(config)
    rails.runtime.register_action(validate_tool_parameters, name="validate_tool_parameters")
    VALIDATED_TOOL_NAMES.clear()

    result = await rails.generate_async(
        messages=[
            {"role": "user", "content": "Use the requested tool"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_safe",
                        "type": "function",
                        "function": {
                            "name": "safe_tool",
                            "arguments": {"param": "safe value"},
                        },
                    }
                ],
            },
        ],
        options={"rails": ["tool_output"]},
    )

    assert isinstance(result, GenerationResponse)
    assert isinstance(result.response, list)
    actual_content = result.response[0]["content"]
    actual_validated_tools = VALIDATED_TOOL_NAMES.copy()

    print_result(
        "TC-04: Safe tool call with list-form rails option",
        "Safe assistant tool call reaches tool_output rails and is not refused when options rails=['tool_output'].",
        f"Assistant content: {actual_content!r}, validated_tools={actual_validated_tools}",
    )

    assert "parameters may be unsafe" not in actual_content
    assert actual_validated_tools == ["safe_tool"]
    print("  Result:   PASS")


def validate_generation_log_tool_rails():
    generation_log = compute_generation_log(
        [
            {"type": "step", "flow_id": "process bot tool call", "timestamp": 0.0, "next_steps": []},
            {"type": "event", "timestamp": 1.0, "data": {"type": "StartToolOutputRail", "flow_id": "check tool call"}},
            {"type": "event", "timestamp": 1.25, "data": {"type": "ToolOutputRailFinished"}},
            {"type": "step", "flow_id": "process user tool messages", "timestamp": 2.0, "next_steps": []},
            {"type": "event", "timestamp": 3.0, "data": {"type": "StartToolInputRail", "flow_id": "check tool result"}},
            {"type": "event", "timestamp": 3.5, "data": {"type": "ToolInputRailFinished"}},
        ]
    )

    activated_rails = generation_log.activated_rails
    actual_types = [rail.type for rail in activated_rails]
    actual_names = [rail.name for rail in activated_rails]
    actual_durations = [rail.duration for rail in activated_rails]

    print_result(
        "TC-05: Generation log tool rail entries",
        "Activated rails include tool_output/tool_input entries with names and durations.",
        f"types={actual_types}, names={actual_names}, durations={actual_durations}",
    )

    assert actual_types == ["tool_output", "tool_input"]
    assert actual_names == ["check tool call", "check tool result"]
    assert actual_durations == [0.25, 0.5]
    print("  Result:   PASS")


async def main():
    await validate_dialog_disabled_tool_output_rails()
    await validate_dialog_disabled_text_output_rails_regression()
    await validate_output_and_tool_output_rails_together()
    await validate_safe_tool_call_with_list_form_rails_option()
    validate_generation_log_tool_rails()
    print("\nAll tool-call rail validation checks passed.")


if __name__ == "__main__":
    asyncio.run(main())
