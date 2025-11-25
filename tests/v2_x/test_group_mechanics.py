# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Test the core flow mechanics"""

import copy
import logging

from rich.logging import RichHandler

from nemoguardrails.colang.v2_x.runtime.statemachine import (
    InternalEvent,
    run_to_completion,
)
from tests.utils import _init_state, is_data_in_events

FORMAT = "%(message)s"
logging.basicConfig(
    level=logging.DEBUG,
    format=FORMAT,
    datefmt="[%X,%f]",
    handlers=[RichHandler(markup=True)],
)

start_main_flow_event = InternalEvent(name="StartFlow", arguments={"flow_id": "main"})

# TODO: Think about how to add back duplicate removal in normalization
# def test_start_and_grouping():
#     """"""

#     content = """
#     flow bot say $script
#       await UtteranceBotAction(script=$script)

#     flow main
#         start bot say "A"
#           and bot say "B"
#           and UtteranceBotAction(script="C")
#           and bot say "A"
#     """

#     state = run_to_completion(_init_state(content), start_main_flow_event)
#     assert is_data_in_events(
#         state.outgoing_events,
#         [
#             {
#                 "type": "StartUtteranceBotAction",
#                 "script": "A",
#             },
#             {
#                 "type": "StartUtteranceBotAction",
#                 "script": "B",
#             },
#             {
#                 "type": "StartUtteranceBotAction",
#                 "script": "C",
#             },
#             {
#                 "type": "StopUtteranceBotAction",
#             },
#             {
#                 "type": "StopUtteranceBotAction",
#             },
#             {
#                 "type": "StopUtteranceBotAction",
#             },
#         ],
#     )


def test_match_and_grouping():
    """Test basic and-grouping mechanic with start and match statement."""

    content = """
    flow bot say $script
      await UtteranceBotAction(script=$script)

    flow main
        start bot say "A" as $ref_a
          and bot say "B" as $ref_b
          and UtteranceBotAction(script="C") as $ref_c
        match $ref_a.Finished()
          and $ref_b.Finished()
          and $ref_c.Finished()
        start bot say "Done"
    """

    state = run_to_completion(_init_state(content), start_main_flow_event)
    events = copy.deepcopy(state.outgoing_events)
    assert is_data_in_events(
        state.outgoing_events,
        [
            {
                "type": "StartUtteranceBotAction",
                "script": "A",
            },
            {
                "type": "StartUtteranceBotAction",
                "script": "B",
            },
            {
                "type": "StartUtteranceBotAction",
                "script": "C",
            },
        ],
    )
    state = run_to_completion(
        state,
        {
            "type": "UtteranceBotActionFinished",
            "final_script": "A",
            "action_uid": events[0]["action_uid"],
        },
    )
    assert is_data_in_events(state.outgoing_events, [])
    state = run_to_completion(
        state,
        {
            "type": "UtteranceBotActionFinished",
            "final_script": "B",
            "action_uid": events[1]["action_uid"],
        },
    )
    assert is_data_in_events(state.outgoing_events, [])
    state = run_to_completion(
        state,
        {
            "type": "UtteranceBotActionFinished",
            "final_script": "C",
            "action_uid": events[2]["action_uid"],
        },
    )
    assert is_data_in_events(
        state.outgoing_events,
        [
            {
                "type": "StartUtteranceBotAction",
                "script": "Done",
            },
            {
                "type": "StopUtteranceBotAction",
            },
        ],
    )


def test_start_or_grouping():
    """Test basic or-grouping mechanic with start statement."""

    content = """
    flow bot say $script
      await UtteranceBotAction(script=$script)

    flow main
        $number = 0
        while $number < 10
          start bot say "Hi"
            or bot say "Hello"
            or bot say "Welcome"
          $number = $number + 1
        await bot say "Done"
    """

    state = run_to_completion(_init_state(content), start_main_flow_event)
    assert len(state.outgoing_events) == 11


def test_await_or_grouping():
    """Test basic or-grouping mechanic with await statement."""

    content = """
    flow user said $transcript
      match UtteranceUserAction().Finished(final_transcript=$transcript)

    flow main
        await user said "A"
          or UtteranceBotAction(script="B")
          or user said "C"
        start UtteranceBotAction(script="Match")
    """

    state = run_to_completion(_init_state(content), start_main_flow_event)
    assert is_data_in_events(
        state.outgoing_events,
        [
            {
                "type": "StartUtteranceBotAction",
                "script": "B",
            }
        ],
    )
    state = run_to_completion(
        state,
        {
            "type": "UtteranceUserActionFinished",
            "final_transcript": "A",
        },
    )
    assert is_data_in_events(
        state.outgoing_events,
        [
            {
                "type": "StopUtteranceBotAction",
            },
            {
                "type": "StartUtteranceBotAction",
                "script": "Match",
            },
            {
                "type": "StopUtteranceBotAction",
            },
        ],
    )
    state = run_to_completion(_init_state(content), start_main_flow_event)
    assert is_data_in_events(
        state.outgoing_events,
        [
            {
                "type": "StartUtteranceBotAction",
                "script": "B",
            }
        ],
    )
    state = run_to_completion(
        state,
        {
            "type": "UtteranceBotActionFinished",
            "final_script": "B",
            "action_uid": state.outgoing_events[0]["action_uid"],
        },
    )
    assert is_data_in_events(
        state.outgoing_events,
        [
            {
                "type": "StartUtteranceBotAction",
                "script": "Match",
            },
            {
                "type": "StopUtteranceBotAction",
            },
        ],
    )
    state = run_to_completion(_init_state(content), start_main_flow_event)
    assert is_data_in_events(
        state.outgoing_events,
        [
            {
                "type": "StartUtteranceBotAction",
                "script": "B",
            }
        ],
    )
    state = run_to_completion(
        state,
        {
            "type": "UtteranceUserActionFinished",
            "final_transcript": "C",
        },
    )
    assert is_data_in_events(
        state.outgoing_events,
        [
            {
                "type": "StopUtteranceBotAction",
            },
            {
                "type": "StartUtteranceBotAction",
                "script": "Match",
            },
            {
                "type": "StopUtteranceBotAction",
            },
        ],
    )


def test_when_or_cases_with_same_references():
    """Test when / orwhen statement mechanics."""

    content = """
    flow user said $transcript
      match UtteranceUserAction().Finished(final_transcript=$transcript) as $ref
      $transcript = $ref.final_transcript

    flow main
      while True
        when user said "A" as $ref
          start UtteranceBotAction(script="case A:{$ref.context.transcript}")
        or when user said "B" as $ref
          start UtteranceBotAction(script="case B:{$ref.context.transcript}")
    """

    state = run_to_completion(_init_state(content), start_main_flow_event)
    assert is_data_in_events(
        state.outgoing_events,
        [],
    )
    state = run_to_completion(
        state,
        {
            "type": "UtteranceUserActionFinished",
            "final_transcript": "A",
        },
    )
    assert is_data_in_events(
        state.outgoing_events,
        [
            {
                "type": "StartUtteranceBotAction",
                "script": "case A:A",
            }
        ],
    )
    state = run_to_completion(
        state,
        {
            "type": "UtteranceUserActionFinished",
            "final_transcript": "B",
        },
    )
    assert is_data_in_events(
        state.outgoing_events,
        [
            {
                "type": "StartUtteranceBotAction",
                "script": "case B:B",
            }
        ],
    )


def test_await_actions_with_same_references():
    """Test references with action grouping."""

    content = """
    flow main
      await UtteranceBotAction(script="A") as $ref and GestureBotAction(gesture="B") as $ref
      start UtteranceBotAction(script="match")
    """

    state = run_to_completion(_init_state(content), start_main_flow_event)
    assert is_data_in_events(
        state.outgoing_events,
        [
            {
                "type": "StartUtteranceBotAction",
                "script": "A",
            },
            {
                "type": "StartGestureBotAction",
                "gesture": "B",
            },
        ],
    )
    utterance_action_uid = state.outgoing_events[0]["action_uid"]
    gesture_action_uid = state.outgoing_events[1]["action_uid"]
    state = run_to_completion(
        state,
        {
            "type": "UtteranceBotActionFinished",
            "script": "A",
            "action_uid": utterance_action_uid,
        },
    )
    assert is_data_in_events(
        state.outgoing_events,
        [],
    )
    state = run_to_completion(
        state,
        {
            "type": "GestureBotActionFinished",
            "gesture": "B",
            "action_uid": gesture_action_uid,
        },
    )
    assert is_data_in_events(
        state.outgoing_events,
        [
            {
                "type": "StartUtteranceBotAction",
                "script": "match",
            },
            {
                "type": "StopUtteranceBotAction",
            },
        ],
    )


def test_await_flows_with_same_references():
    """Test references with flow grouping."""

    content = """
    flow user said $transcript
      match UtteranceUserAction().Finished(final_transcript=$transcript) as $ref
      $transcript = $ref.arguments.final_transcript

    flow main
      while True
        await user said "A" as $ref or user said "B" as $ref
        start UtteranceBotAction(script=$ref.context.transcript)
    """

    state = run_to_completion(_init_state(content), start_main_flow_event)
    assert is_data_in_events(
        state.outgoing_events,
        [],
    )
    state = run_to_completion(
        state,
        {
            "type": "UtteranceUserActionFinished",
            "final_transcript": "A",
        },
    )
    assert is_data_in_events(
        state.outgoing_events,
        [
            {
                "type": "StartUtteranceBotAction",
                "script": "A",
            }
        ],
    )
    state = run_to_completion(
        state,
        {
            "type": "UtteranceUserActionFinished",
            "final_transcript": "B",
        },
    )
    assert is_data_in_events(
        state.outgoing_events,
        [
            {
                "type": "StartUtteranceBotAction",
                "script": "B",
            }
        ],
    )


def test_await_or_group_finish():
    """Test or-grouping with await flow statements."""

    content = """
    flow bot say $text
      await UtteranceBotAction(script=$text) as $action

    flow bot express greeting
      bot say "Hi there!"
        or bot say "Welcome!"

    flow main
      bot express greeting
    """

    config = _init_state(content)
    state = run_to_completion(config, start_main_flow_event)
    assert is_data_in_events(
        state.outgoing_events,
        [
            {
                "type": "StartUtteranceBotAction",
            }
        ],
    )
    state = run_to_completion(
        state,
        {
            "type": "UtteranceBotActionFinished",
            "final_script": state.outgoing_events[0]["script"],
            "action_uid": state.outgoing_events[0]["action_uid"],
        },
    )
    assert is_data_in_events(
        state.outgoing_events,
        [],
    )


def test_await_and_or_grouping():
    """Test mixed and-or-grouping with await flow statements."""

    content = """
    flow user said $transcript
      match UtteranceUserAction().Finished(final_transcript=$transcript)

    flow main
        await (user said "A" and user said "B")
          or (user said "C" and user said "D")
        start UtteranceBotAction(script="Match")
    """

    state = run_to_completion(_init_state(content), start_main_flow_event)
    assert is_data_in_events(
        state.outgoing_events,
        [],
    )
    state = run_to_completion(
        state,
        {
            "type": "UtteranceUserActionFinished",
            "final_transcript": "A",
        },
    )
    assert is_data_in_events(
        state.outgoing_events,
        [],
    )
    state = run_to_completion(
        state,
        {
            "type": "UtteranceUserActionFinished",
            "final_transcript": "C",
        },
    )
    assert is_data_in_events(
        state.outgoing_events,
        [],
    )
    state = run_to_completion(
        state,
        {
            "type": "UtteranceUserActionFinished",
            "final_transcript": "B",
        },
    )
    assert is_data_in_events(
        state.outgoing_events,
        [
            {
                "type": "StartUtteranceBotAction",
                "script": "Match",
            },
            {
                "type": "StopUtteranceBotAction",
            },
        ],
    )

    state = run_to_completion(_init_state(content), start_main_flow_event)
    assert is_data_in_events(
        state.outgoing_events,
        [],
    )
    state = run_to_completion(
        state,
        {
            "type": "UtteranceUserActionFinished",
            "final_transcript": "B",
        },
    )
    assert is_data_in_events(
        state.outgoing_events,
        [],
    )
    state = run_to_completion(
        state,
        {
            "type": "UtteranceUserActionFinished",
            "final_transcript": "C",
        },
    )
    assert is_data_in_events(
        state.outgoing_events,
        [],
    )
    state = run_to_completion(
        state,
        {
            "type": "UtteranceUserActionFinished",
            "final_transcript": "D",
        },
    )
    assert is_data_in_events(
        state.outgoing_events,
        [
            {
                "type": "StartUtteranceBotAction",
                "script": "Match",
            },
            {
                "type": "StopUtteranceBotAction",
            },
        ],
    )


def test_await_and_or_group_with_flow_hierarchy():
    """Test and-or-grouping with flow hierarchies."""

    content = """
    flow _bot_say $text
      await UtteranceBotAction(script=$text) as $action

    flow bot say $text
      await _bot_say $text

    flow bot gesture $gesture
      await GestureBotAction(gesture=$gesture)

    flow bot express $text
      await _bot_say $text

    flow bot express greeting
      (bot express "Hi there!"
        or bot express "Welcome!")
        and bot gesture "Wave with one hand"

    flow main
      bot express greeting
        and bot gesture "Smile"
      UtteranceBotAction(script="Success")
    """

    state = run_to_completion(_init_state(content), start_main_flow_event)
    assert is_data_in_events(
        state.outgoing_events,
        [
            {
                "type": "StartUtteranceBotAction",
            },
            {"type": "StartGestureBotAction", "gesture": "Wave with one hand"},
            {"type": "StartGestureBotAction", "gesture": "Smile"},
        ],
    )
    utterance_action_uid = state.outgoing_events[0]["action_uid"]
    gesture_1_action_uid = state.outgoing_events[1]["action_uid"]
    gesture_2_action_uid = state.outgoing_events[2]["action_uid"]
    state = run_to_completion(
        state,
        {
            "type": "UtteranceBotActionFinished",
            "action_uid": utterance_action_uid,
        },
    )
    assert is_data_in_events(
        state.outgoing_events,
        [],
    )
    state = run_to_completion(
        state,
        {
            "type": "GestureBotActionFinished",
            "action_uid": gesture_1_action_uid,
        },
    )
    assert is_data_in_events(
        state.outgoing_events,
        [],
    )
    state = run_to_completion(
        state,
        {
            "type": "GestureBotActionFinished",
            "action_uid": gesture_2_action_uid,
        },
    )
    assert is_data_in_events(
        state.outgoing_events,
        [{"type": "StartUtteranceBotAction", "script": "Success"}],
    )


if __name__ == "__main__":
    test_when_or_cases_with_same_references()
