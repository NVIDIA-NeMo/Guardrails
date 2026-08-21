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
import json
import os.path

import pytest

from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.actions.rail_outcome import RailOutcome, TransformTarget
from nemoguardrails.rails.llm.options import (
    GenerationLog,
    GenerationResponse,
    GenerationStats,
)
from tests.utils import TestChat

_RAIL_OUTCOME_CASES = [
    pytest.param(
        RailOutcome.allow(reason="allowed", metadata={"score": 0.9}),
        id="allow",
    ),
    pytest.param(
        RailOutcome.block(reason="blocked", metadata={"categories": ["unsafe"]}),
        id="block",
    ),
    pytest.param(
        RailOutcome.transform(
            [
                (TransformTarget.USER_MESSAGE, "masked input"),
                (TransformTarget.BOT_MESSAGE, "masked output"),
                (TransformTarget.RELEVANT_CHUNKS, "masked chunks"),
            ],
            reason="transformed",
            metadata={"entities": ["name"]},
        ),
        id="transform",
    ),
]


def _generate_with_rail_outcome(outcome, *, log=None, output_vars=None):
    config = RailsConfig.from_content(
        colang_content="""
            define subflow outcome input rail
              $response = execute outcome_action
        """,
        yaml_content="""
            rails:
              input:
                flows:
                  - outcome input rail
        """,
    )
    chat = TestChat(config, llm_completions=[])

    async def outcome_action():
        return outcome

    chat.app.register_action(outcome_action)
    options = {"rails": ["input"]}
    if log is not None:
        options["log"] = log
    if output_vars is not None:
        options["output_vars"] = output_vars

    return chat.app.generate("Hello!", options=options)


def _json_round_trip(model):
    return json.loads(json.dumps(model.model_dump()))


def _expected_serialized_outcome(outcome):
    return {
        "decision": outcome.decision.value,
        "reason": outcome.reason,
        "metadata": dict(outcome.metadata),
        "transforms": [{"target": spec.target.value, "text": spec.text} for spec in outcome.transforms],
        "failed": outcome.failed,
    }


def _get_outcome_action_event(events):
    action_events = [
        event
        for event in events
        if event["type"] == "InternalSystemActionFinished" and event["action_name"] == "outcome_action"
    ]
    assert len(action_events) == 1, f"Expected one outcome_action completion event, found {len(action_events)}"
    return action_events[0]


def test_output_vars_1():
    config = RailsConfig.from_content(
        colang_content="""
                define user express greeting
                  "hi"

                define flow
                  user express greeting
                  $user_greeted = True
                  bot express greeting

                define bot express greeting
                  "Hello! How can I assist you today?"
            """,
    )
    chat = TestChat(
        config,
        llm_completions=[
            "  express greeting",
            "  express greeting",
        ],
    )

    res = chat.app.generate("hi", options={"output_vars": ["user_greeted", "something_else"]})
    output_data = res.dict().get("output_data", {})

    # We check also that a non-existent variable returns None.
    assert output_data == {"user_greeted": True, "something_else": None}

    # We also test again by trying to return the full context
    res = chat.app.generate("hi", options={"output_vars": True})
    output_data = res.dict().get("output_data", {})

    # There should be many keys
    assert len(output_data.keys()) > 5


def test_triggered_rails_info_1():
    config = RailsConfig.from_content(
        colang_content='''
            define user express greeting
              "hi"

            define flow
              user express greeting
              $user_greeted = True
              bot express greeting

            define bot express greeting
              "Hello! How can I assist you today?"

            define subflow dummy input rail
              """A dummy input rail which checks if the word "dummy" is included in the text."""
              if "dummy" in $user_message
                bot refuse to respond
                stop
        ''',
        yaml_content="""
            rails:
              input:
                flows:
                  - dummy input rail
        """,
    )
    chat = TestChat(
        config,
        llm_completions=[
            "  express greeting",
            "  express greeting",
        ],
    )

    res: GenerationResponse = chat.app.generate(
        "you are dummy", options={"log": {"activated_rails": True}, "output_vars": True}
    )

    assert res.response == "I'm sorry, I can't respond to that."

    output_data = res.output_data
    assert output_data["triggered_input_rail"] == "dummy input rail"


def test_triggered_rails_info_2():
    config = RailsConfig.from_content(
        colang_content='''
            define user express greeting
              "hi"

            define flow
              user express greeting
              $user_greeted = True
              bot express greeting

            define subflow dummy input rail
              """A dummy input rail which checks if the word "dummy" is included in the text."""
              if "dummy" in $user_message
                bot refuse to respond
                stop

            define subflow dummy output rail
              """A dummy input rail which checks if the word "dummy" is included in the text."""
              if "dummy" in $bot_message
                bot refuse to respond
                stop
        ''',
        yaml_content="""
            rails:
              input:
                flows:
                  - dummy input rail
              output:
                flows:
                  - dummy output rail
        """,
    )
    chat = TestChat(
        config,
        llm_completions=[
            "  express greeting",
            '  "You are dummy!"',
        ],
    )

    res: GenerationResponse = chat.app.generate(
        "Hello!",
        options={
            "log": {
                "activated_rails": True,
                "llm_calls": True,
                "internal_events": True,
                "colang_history": True,
            },
            "output_vars": True,
        },
    )

    assert res.response == "I'm sorry, I can't respond to that."

    assert len(res.log.activated_rails) == 5
    assert len(res.log.llm_calls) == 2
    assert len(res.log.internal_events) > 0
    assert len(res.log.colang_history) > 0


@pytest.mark.skip(reason="Run manually.")
def test_triggered_abc_bot():
    config = RailsConfig.from_path(os.path.join(os.path.dirname(__file__), "..", "examples/bots/abc"))

    rails = LLMRails(config)
    res: GenerationResponse = rails.generate("Hello!", options={"log": {"activated_rails": True}, "output_vars": True})

    print("############################")
    print(json.dumps(res.log.dict(), indent=True))

    res.log.print_summary()


@pytest.mark.parametrize("outcome", _RAIL_OUTCOME_CASES)
@pytest.mark.parametrize("log_field", ["activated_rails", "internal_events"])
def test_generation_log_with_rail_outcome_is_json_serializable(outcome, log_field):
    res = _generate_with_rail_outcome(outcome, log={log_field: True})

    data = _json_round_trip(res.log)

    if log_field == "activated_rails":
        serialized_outcome = data["activated_rails"][0]["executed_actions"][0]["return_value"]
    else:
        action_event = _get_outcome_action_event(data["internal_events"])
        serialized_outcome = action_event["return_value"]

    assert serialized_outcome == _expected_serialized_outcome(outcome)


@pytest.mark.parametrize("outcome", _RAIL_OUTCOME_CASES)
@pytest.mark.parametrize(
    "surface",
    [
        "activated_rails",
        "internal_events",
        "selected_output_vars",
        "all_output_vars",
    ],
)
def test_generation_response_with_rail_outcome_is_json_serializable(outcome, surface):
    if surface in {"activated_rails", "internal_events"}:
        res = _generate_with_rail_outcome(outcome, log={surface: True})
    elif surface == "selected_output_vars":
        res = _generate_with_rail_outcome(outcome, output_vars=["response"])
    else:
        res = _generate_with_rail_outcome(outcome, output_vars=True)

    data = _json_round_trip(res)

    if surface == "activated_rails":
        serialized_outcome = data["log"]["activated_rails"][0]["executed_actions"][0]["return_value"]
    elif surface == "internal_events":
        action_event = _get_outcome_action_event(data["log"]["internal_events"])
        serialized_outcome = action_event["return_value"]
    else:
        serialized_outcome = data["output_data"]["response"]

    assert serialized_outcome == _expected_serialized_outcome(outcome)


@pytest.mark.skip(reason="Run manually.")
def test_triggered_rails_info_3():
    config = RailsConfig.from_content(
        yaml_content="""
            models:
              - type: main
                engine: openai
                model: gpt-3.5-turbo-instruct
        """,
    )
    rails = LLMRails(config)
    res: GenerationResponse = rails.generate(
        "Hello!",
        options={
            "log": {
                "activated_rails": True,
                "llm_calls": True,
                "internal_events": True,
                "colang_history": True,
            },
            "llm_output": True,
            "output_vars": True,
        },
    )

    print("############################")
    # print(json.dumps(res.log.dict(), indent=True))
    print(json.dumps(res.dict(), indent=True))
    res.log.print_summary()


def test_only_input_output_validation():
    config = RailsConfig.from_content(
        colang_content='''
            define user express greeting
              "hi"

            define flow
              user express greeting
              $user_greeted = True
              bot express greeting

            define subflow dummy input rail
              """A dummy input rail which checks if the word "dummy" is included in the text."""
              if "dummy" in $user_message
                bot refuse to respond
                stop

            define subflow dummy output rail
              """A dummy input rail which checks if the word "dummy" is included in the text."""
              if "dummy" in $bot_message
                bot refuse to respond
                stop
        ''',
        yaml_content="""
            rails:
              input:
                flows:
                  - dummy input rail
              output:
                flows:
                  - dummy output rail
        """,
    )
    chat = TestChat(
        config,
        llm_completions=[],
    )

    # Test only input

    res: GenerationResponse = chat.app.generate(
        "Hello!",
        options={
            "rails": ["input"],
            "log": {
                "activated_rails": True,
            },
        },
    )

    assert res.response == "Hello!"

    res = chat.app.generate(
        "Hello dummy!",
        options={
            "rails": ["input"],
            "log": {
                "activated_rails": True,
            },
        },
    )

    assert res.response == "I'm sorry, I can't respond to that."

    # Test only output

    res: GenerationResponse = chat.app.generate(
        messages=[
            {"role": "user", "content": "hi!"},
            {"role": "assistant", "content": "Hi there!"},
        ],
        options={
            "rails": ["output"],
            "log": {
                "activated_rails": True,
            },
        },
    )

    assert res.response == [{"content": "Hi there!", "role": "assistant"}]

    res = chat.app.generate(
        messages=[
            {"role": "user", "content": "hi!"},
            {"role": "assistant", "content": "Hi dummy!"},
        ],
        options={
            "rails": ["output"],
            "log": {
                "activated_rails": True,
            },
        },
    )

    assert res.response == [{"content": "I'm sorry, I can't respond to that.", "role": "assistant"}]


def test_generation_log_print_summary(capsys):
    """Test printing rais stats with dummy data"""

    stats = GenerationStats(
        input_rails_duration=1.0,
        dialog_rails_duration=2.0,
        generation_rails_duration=3.0,
        output_rails_duration=4.0,
        total_duration=10.0,  # Sum of all previous rail durations
        llm_calls_duration=8.0,  # Less than total duration
        llm_calls_count=4,  # Input, dialog, generation and output calls
        llm_calls_total_prompt_tokens=1000,
        llm_calls_total_completion_tokens=2000,
        llm_calls_total_tokens=3000,  # Sum of prompt and completion tokens
    )

    generation_log = GenerationLog(activated_rails=[], stats=stats)

    generation_log.print_summary()
    capture = capsys.readouterr()
    capture_lines = capture.out.splitlines()

    # Check the correct times were printed
    assert capture_lines[1] == "# General stats"
    assert capture_lines[3] == "- Total time: 10.00s"
    assert capture_lines[4] == "  - [1.00s][10.0%]: INPUT Rails"
    assert capture_lines[5] == "  - [2.00s][20.0%]: DIALOG Rails"
    assert capture_lines[6] == "  - [3.00s][30.0%]: GENERATION Rails"
    assert capture_lines[7] == "  - [4.00s][40.0%]: OUTPUT Rails"
    assert (
        capture_lines[8]
        == "- 4 LLM calls, 8.00s total duration, 1000 total prompt tokens, 2000 total completion tokens, 3000 total tokens."
    )


@pytest.mark.parametrize(
    "input_opt,output_opt,dialog_opt,expect_input,expect_output",
    [
        (True, True, True, True, True),
        (True, True, False, True, False),
        (True, False, True, True, False),
        (True, False, False, True, False),
        (False, True, True, False, True),
        (False, True, False, False, False),
        (False, False, True, False, False),
        (False, False, False, False, False),
    ],
)
@pytest.mark.asyncio
async def test_rails_options_combinations(input_opt, output_opt, dialog_opt, expect_input, expect_output):
    """
    Test all combinations of input/output/dialog options.
    When dialog=False and no bot_message is provided, output rails should skip.
    """
    config = RailsConfig.from_content(
        colang_content="""
            define user express greeting
              "hi"

            define flow
              user express greeting
              bot express greeting

            define subflow dummy input rail
              if "block" in $user_message
                bot refuse to respond
                stop

            define subflow dummy output rail
              if "block" in $bot_message
                bot refuse to respond
                stop
        """,
        yaml_content="""
            rails:
              input:
                flows:
                  - dummy input rail
              output:
                flows:
                  - dummy output rail
        """,
    )
    chat = TestChat(
        config,
        llm_completions=["  express greeting", '  "Hello!"'] if dialog_opt else [],
    )

    res: GenerationResponse = await chat.app.generate_async(
        "Hello!",
        options={
            "rails": {"input": input_opt, "output": output_opt, "dialog": dialog_opt},
            "log": {"activated_rails": True},
        },
    )

    activated_rails = res.log.activated_rails if res.log else []
    rail_names = [r.name for r in activated_rails]

    input_rails_ran = any("input" in name.lower() for name in rail_names)
    output_rails_ran = any("output" in name.lower() for name in rail_names)

    assert input_rails_ran == expect_input, f"Input rails: expected {expect_input}, got {rail_names}"
    assert output_rails_ran == expect_output, f"Output rails: expected {expect_output}, got {rail_names}"
