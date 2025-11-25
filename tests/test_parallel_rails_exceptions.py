# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os

import pytest

from nemoguardrails import RailsConfig
from tests.utils import TestChat

CONFIGS_FOLDER = os.path.join(os.path.dirname(__file__), "test_configs")

OPTIONS = {
    "log": {
        "activated_rails": True,
    }
}


@pytest.mark.asyncio
async def test_parallel_rails_exception_stop_flag():
    """Test that stop flag is set when a rail raises an exception in parallel mode."""

    config = RailsConfig.from_path(os.path.join(CONFIGS_FOLDER, "parallel_rails_with_exceptions"))
    chat = TestChat(
        config,
        llm_completions=[
            "Hi there!",
        ],
    )

    chat >> "This message contains an unsafe term"
    result = await chat.app.generate_async(messages=chat.history, options=OPTIONS)

    stopped_rails = [rail for rail in result.log.activated_rails if rail.stop]

    assert len(stopped_rails) >= 1, (
        f"Expected at least one stopped rail, but found {len(stopped_rails)}. "
        f"This indicates the stop flag is not being set for exception-based rails."
    )

    for rail in stopped_rails:
        assert "stop" in rail.decisions, (
            f"Stopped rail '{rail.name}' has stop=True but 'stop' not in decisions: {rail.decisions}. "
            f"This indicates inconsistency between stop flag and decisions list."
        )

    has_exception = any(msg.get("role") == "exception" for msg in result.response if isinstance(msg, dict))
    assert has_exception, "Expected response to contain exception when rail blocks with exception"


@pytest.mark.asyncio
async def test_parallel_rails_exception_no_duplicates():
    """Test that exception-based rails don't create duplicate activated_rails entries."""

    config = RailsConfig.from_path(os.path.join(CONFIGS_FOLDER, "parallel_rails_with_exceptions"))
    chat = TestChat(
        config,
        llm_completions=[
            "Hi there!",
        ],
    )

    chat >> "This is an offtopic message"
    result = await chat.app.generate_async(messages=chat.history, options=OPTIONS)

    stopped_rails = [rail for rail in result.log.activated_rails if rail.stop]
    assert len(stopped_rails) > 0, "Expected at least one stopped rail. Cannot test for duplicates if no rails stopped."

    rail_names_with_actions = {}
    for rail in result.log.activated_rails:
        if rail.executed_actions:
            name = rail.name
            if name in rail_names_with_actions:
                rail_names_with_actions[name] += 1
            else:
                rail_names_with_actions[name] = 1

    duplicates = {k: v for k, v in rail_names_with_actions.items() if v > 1}
    assert len(duplicates) == 0, (
        f"Found duplicate rails with executed actions: {duplicates}. "
        f"This indicates stopped rail events are being processed multiple times."
    )


@pytest.mark.asyncio
async def test_parallel_rails_exception_single_stop_in_decisions():
    """Test that 'stop' appears only once in decisions list.

    This test would have FAILED before the final fix because both the exception
    handler and end-of-processing logic were adding 'stop' without checking.
    """
    config = RailsConfig.from_path(os.path.join(CONFIGS_FOLDER, "parallel_rails_with_exceptions"))
    chat = TestChat(
        config,
        llm_completions=[
            "Hi there!",
        ],
    )

    chat >> "This contains both unsafe and offtopic terms"
    result = await chat.app.generate_async(messages=chat.history, options=OPTIONS)

    stopped_rails = [rail for rail in result.log.activated_rails if rail.stop]

    assert len(stopped_rails) > 0, "Expected at least one stopped rail. Cannot test 'stop' count if no rails stopped."

    for rail in stopped_rails:
        stop_count = rail.decisions.count("stop")
        assert stop_count == 1, (
            f"Rail '{rail.name}' has 'stop' appearing {stop_count} times in decisions: {rail.decisions}. "
            f"Expected exactly 1. This indicates duplicate 'stop' additions."
        )


@pytest.mark.asyncio
async def test_parallel_rails_exception_consistency_with_sequential():
    """Test that parallel mode with exceptions behaves like sequential mode.

    This test verifies that the fix makes parallel and sequential modes consistent.
    """
    config_parallel = RailsConfig.from_path(os.path.join(CONFIGS_FOLDER, "parallel_rails_with_exceptions"))

    chat_parallel = TestChat(
        config_parallel,
        llm_completions=["Hi there!"],
    )
    chat_parallel >> "This message has an unsafe word"
    result_parallel = await chat_parallel.app.generate_async(messages=chat_parallel.history, options=OPTIONS)

    config_sequential = RailsConfig.from_path(os.path.join(CONFIGS_FOLDER, "parallel_rails_with_exceptions"))
    config_sequential.rails.input.parallel = False
    config_sequential.rails.output.parallel = False

    chat_sequential = TestChat(
        config_sequential,
        llm_completions=["Hi there!"],
    )
    chat_sequential >> "This message has an unsafe word"
    result_sequential = await chat_sequential.app.generate_async(messages=chat_sequential.history, options=OPTIONS)

    parallel_stopped = [r for r in result_parallel.log.activated_rails if r.stop]
    sequential_stopped = [r for r in result_sequential.log.activated_rails if r.stop]

    assert len(parallel_stopped) > 0, "Parallel mode should have stopped rails"
    assert len(sequential_stopped) > 0, "Sequential mode should have stopped rails"

    assert len(parallel_stopped) == len(sequential_stopped), (
        f"Parallel and sequential modes should stop the same number of rails. "
        f"Parallel: {len(parallel_stopped)}, Sequential: {len(sequential_stopped)}"
    )

    parallel_stopped_names = {r.name for r in parallel_stopped}
    sequential_stopped_names = {r.name for r in sequential_stopped}

    assert parallel_stopped_names == sequential_stopped_names, (
        f"Parallel and sequential modes should stop the same rails. "
        f"Parallel stopped: {parallel_stopped_names}, Sequential stopped: {sequential_stopped_names}"
    )


@pytest.mark.asyncio
async def test_parallel_rails_exception_with_passing_rails():
    """Test that when one rail stops with exception, others that passed are tracked correctly.

    This ensures that only the blocking rail has stop=True, not all rails.
    """
    config = RailsConfig.from_path(os.path.join(CONFIGS_FOLDER, "parallel_rails_with_exceptions"))
    chat = TestChat(
        config,
        llm_completions=["Hi there!"],
    )

    # this message triggers safety check but not topic check
    chat >> "This message is unsafe but on topic"
    result = await chat.app.generate_async(messages=chat.history, options=OPTIONS)

    stopped_rails = [r for r in result.log.activated_rails if r.stop]
    passed_rails = [r for r in result.log.activated_rails if not r.stop]

    assert len(stopped_rails) > 0, "Should have at least one stopped rail"
    assert len(passed_rails) > 0, "Should have at least one passed rail"

    safety_rail = next((r for r in result.log.activated_rails if "safety" in r.name.lower()), None)
    assert safety_rail is not None, "Safety rail should have executed"
    assert safety_rail.stop, "Safety rail should have stopped"
    assert any("check_safety_action" in action.action_name for action in safety_rail.executed_actions), (
        "check_safety_action should have fired"
    )

    # stopped rails should have "stop" in decisions
    for rail in stopped_rails:
        assert "stop" in rail.decisions, f"Stopped rail {rail.name} missing 'stop' in decisions"

    # passed rails should NOT have "stop" in decisions
    for rail in passed_rails:
        assert "stop" not in rail.decisions, f"Passed rail {rail.name} incorrectly has 'stop' in decisions"


@pytest.mark.asyncio
async def test_parallel_rails_multiple_simultaneous_violations():
    """Test when multiple rails detect violations simultaneously.

    This is a critical edge case that tests race conditions - when 2+ rails
    both detect violations at nearly the same time, our exception detection
    should handle whichever completes first.
    """
    config = RailsConfig.from_path(os.path.join(CONFIGS_FOLDER, "parallel_rails_with_exceptions"))
    chat = TestChat(
        config,
        llm_completions=["Hi there!"],
    )

    # this input violates BOTH safety (has "unsafe") AND topic (has "stupid")
    chat >> "This is an unsafe and stupid message"
    result = await chat.app.generate_async(messages=chat.history, options=OPTIONS)

    stopped_rails = [r for r in result.log.activated_rails if r.stop]

    assert len(stopped_rails) > 0, "Expected at least one stopped rail when multiple violations occur"

    stopped_rail_names = {r.name for r in stopped_rails}
    assert "check safety with exception" in stopped_rail_names or "check topic with exception" in stopped_rail_names, (
        f"Expected safety or topic rail to stop, got: {stopped_rail_names}"
    )

    for rail in stopped_rails:
        assert rail.stop is True, f"Stopped rail '{rail.name}' should have stop=True"
        assert "stop" in rail.decisions, f"Stopped rail '{rail.name}' should have 'stop' in decisions"

    has_exception = any(msg.get("role") == "exception" for msg in result.response if isinstance(msg, dict))
    assert has_exception, "Expected exception in response when rails block"


@pytest.mark.asyncio
async def test_parallel_output_rails_exception_stop_flag():
    """Test that stop flag is set for output rails with exceptions in parallel mode.

    This verifies that our fix works for output rails, not just input rails,
    since both use the same _run_flows_in_parallel mechanism.
    """
    config = RailsConfig.from_path(os.path.join(CONFIGS_FOLDER, "parallel_rails_with_exceptions"))
    chat = TestChat(
        config,
        llm_completions=[
            "This is a response with harmful content that should be blocked",
        ],
    )

    chat >> "Hello"
    result = await chat.app.generate_async(messages=chat.history, options=OPTIONS)

    output_rails = [r for r in result.log.activated_rails if r.type == "output"]
    assert len(output_rails) > 0, "Expected at least one output rail to be activated"

    stopped_output_rails = [r for r in output_rails if r.stop]

    assert len(stopped_output_rails) > 0, "Expected at least one stopped output rail when output is blocked"

    output_safety_rail = next((r for r in output_rails if "output safety" in r.name.lower()), None)
    assert output_safety_rail is not None, "Output safety rail should have executed"
    assert output_safety_rail.stop, "Output safety rail should have stopped"
    assert any("check_output_safety_action" in action.action_name for action in output_safety_rail.executed_actions), (
        "check_output_safety_action should have fired"
    )

    for rail in stopped_output_rails:
        assert "stop" in rail.decisions, f"Stopped output rail '{rail.name}' should have 'stop' in decisions"

    has_exception = any(msg.get("role") == "exception" for msg in result.response if isinstance(msg, dict))
    assert has_exception, "Expected exception when output rail blocks"


@pytest.mark.asyncio
async def test_parallel_rails_context_updates_preserved():
    """Test that context updates are preserved when a rail stops with exception.

    This verifies that even when a rail raises an exception, any context updates
    it made before stopping are still captured.
    """
    config = RailsConfig.from_path(os.path.join(CONFIGS_FOLDER, "parallel_rails_with_exceptions"))

    config.rails.input.flows.append("check with context update")

    chat = TestChat(
        config,
        llm_completions=["Hi there!"],
    )

    chat >> "This message is blocked and should increment violation count"
    result = await chat.app.generate_async(messages=chat.history, options=OPTIONS)

    stopped_rails = [r for r in result.log.activated_rails if r.stop]

    assert len(stopped_rails) > 0, "Expected at least one stopped rail"

    for rail in stopped_rails:
        assert rail.stop is True
        assert "stop" in rail.decisions


@pytest.mark.asyncio
async def test_input_rails_only_parallel_with_exceptions():
    """Test parallel input rails with NO output rails configured."""

    config = RailsConfig.from_path(os.path.join(CONFIGS_FOLDER, "parallel_rails_with_exceptions"))

    chat = TestChat(
        config,
        llm_completions=["Hi there!"],
    )

    options_input_only = {
        "log": {
            "activated_rails": True,
        },
        "rails": {
            "input": True,
            "output": False,
            "dialog": False,
            "retrieval": False,
        },
    }

    chat >> "This is an unsafe message"
    result = await chat.app.generate_async(messages=chat.history, options=options_input_only)

    input_rails = [r for r in result.log.activated_rails if r.type == "input"]
    output_rails = [r for r in result.log.activated_rails if r.type == "output"]

    assert len(input_rails) > 0, "Should have input rails"
    assert len(output_rails) == 0, "Should have no output rails"

    stopped_input_rails = [r for r in input_rails if r.stop]
    assert len(stopped_input_rails) > 0, "Expected at least one stopped input rail when input is blocked"

    for rail in stopped_input_rails:
        assert rail.stop is True
        assert "stop" in rail.decisions


@pytest.mark.asyncio
async def test_output_rails_only_parallel_with_exceptions():
    """Test parallel output rails with NO input rails configured."""
    config = RailsConfig.from_path(os.path.join(CONFIGS_FOLDER, "parallel_rails_with_exceptions"))

    chat = TestChat(
        config,
        llm_completions=[
            "This response contains harmful content",
        ],
    )

    options_output_only = {
        "log": {
            "activated_rails": True,
        },
        "rails": {
            "input": False,
            "output": True,
            "dialog": False,
            "retrieval": False,
        },
    }

    chat >> "Hello"
    result = await chat.app.generate_async(messages=chat.history, options=options_output_only)

    input_rails = [r for r in result.log.activated_rails if r.type == "input"]
    output_rails = [r for r in result.log.activated_rails if r.type == "output"]

    assert len(input_rails) == 0, "Should have no input rails"
    assert len(output_rails) > 0, "Should have output rails"

    stopped_output_rails = [r for r in output_rails if r.stop]
    assert len(stopped_output_rails) > 0, "Expected at least one stopped output rail when output is blocked"

    for rail in stopped_output_rails:
        assert rail.stop is True
        assert "stop" in rail.decisions
