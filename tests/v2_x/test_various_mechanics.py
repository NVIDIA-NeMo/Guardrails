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


def test_multi_flow_level_member_access():
    """Test access of child of child flow attributes."""

    content = """
    flow user said $transcript
      match UtteranceUserAction.Finished(final_transcript=$transcript) as $event
      $final_transcript = $event.final_transcript

    flow user instructed bot
      user said "do something" as $user_said_flow
      $instruction = $user_said_flow.context.final_transcript

    flow main
      await user instructed bot as $ref
      start UtteranceBotAction(script=$ref.context.instruction)
    """

    config = _init_state(content)
    state = run_to_completion(config, start_main_flow_event)
    assert is_data_in_events(
        state.outgoing_events,
        [],
    )
    state = run_to_completion(
        state,
        {
            "type": "UtteranceUserActionFinished",
            "final_transcript": "do something",
        },
    )
    assert is_data_in_events(
        state.outgoing_events,
        [
            {
                "type": "StartUtteranceBotAction",
                "script": "do something",
            },
            {
                "type": "StopUtteranceBotAction",
            },
        ],
    )


def test_flow_start_event_fallback():
    """Test the flow start fallback case."""

    content = """
    flow a
      match StartFlow() as $ref
      start UtteranceBotAction(script="Success")
      send FlowStarted(flow_id=$ref.arguments.flow_id, flow_instance_uid=$ref.arguments.flow_instance_uid ,param="test")

    flow main
      start a
      start unknown fl $param="test"
      start UtteranceBotAction(script="End")
    """

    config = _init_state(content)
    state = run_to_completion(config, start_main_flow_event)
    assert is_data_in_events(
        state.outgoing_events,
        [
            {
                "type": "StartUtteranceBotAction",
                "script": "Success",
            },
            {
                "type": "StopUtteranceBotAction",
            },
            {
                "type": "StartUtteranceBotAction",
                "script": "End",
            },
        ],
    )


def test_multi_level_member_match_from_reference():
    """Test matching based on reference attributes."""

    content = """
    flow a
      match UtteranceUserAction.Finished(final_transcript="Done")

    flow main
      $flow_instance_uid = "(a){uid()}"
      send StartFlow(flow_id="a", flow_instance_uid=$flow_instance_uid)
      match FlowStarted(flow_id="a", flow_instance_uid=$flow_instance_uid) as $event_ref
      match $event_ref.flow.Finished()
      start UtteranceBotAction(script="End")
    """

    config = _init_state(content)
    state = run_to_completion(config, start_main_flow_event)
    assert is_data_in_events(
        state.outgoing_events,
        [],
    )
    state = run_to_completion(
        state,
        {
            "type": "UtteranceUserActionFinished",
            "final_transcript": "Done",
        },
    )
    assert is_data_in_events(
        state.outgoing_events,
        [
            {
                "type": "StartUtteranceBotAction",
                "script": "End",
            },
            {
                "type": "StopUtteranceBotAction",
            },
        ],
    )


def test_flow_end_on_parent_flow_finished():
    """Check that a flow gets stopped when its parent flow has ended."""

    content = """
    flow child
      start UtteranceBotAction(script="Started")
      match UtteranceUserAction.Finished(final_transcript="too late")

    flow parent
      start child

    flow main
      start parent
    """

    config = _init_state(content)
    state = run_to_completion(config, start_main_flow_event)
    assert is_data_in_events(
        state.outgoing_events,
        [
            {
                "type": "StartUtteranceBotAction",
                "script": "Started",
            },
            {
                "type": "StopUtteranceBotAction",
            },
        ],
    )
    state = run_to_completion(
        state,
        {
            "type": "UtteranceUserActionFinished",
            "final_transcript": "too late",
        },
    )
    assert is_data_in_events(
        state.outgoing_events,
        [],
    )


def test_event_action_wrapper_abstraction():
    """Test a common action abstraction use case."""

    content = """
    flow user said $text
      match UtteranceUserAction.Finished(final_transcript=$text) as $event

    flow bot say $text
      await UtteranceBotAction(script=$text) as $action

    flow bot express $text
      bot say $text

    flow bot express greeting
      bot express "hi"

    flow user expressed greeting
      user said "hi"
        or user said "hello"

    flow greeting
      user expressed greeting
      bot express greeting

    flow main
      activate greeting
      match UtteranceUserAction.Finished(final_transcript="End")
    """

    config = _init_state(content)
    state = run_to_completion(config, start_main_flow_event)
    assert is_data_in_events(
        state.outgoing_events,
        [],
    )
    state = run_to_completion(
        state,
        {
            "type": "UtteranceUserActionFinished",
            "final_transcript": "hi",
        },
    )
    assert is_data_in_events(
        state.outgoing_events,
        [
            {
                "type": "StartUtteranceBotAction",
                "script": "hi",
            }
        ],
    )
    state = run_to_completion(
        state,
        {
            "type": "UtteranceBotActionFinished",
            "final_script": "Hello",
            "action_uid": state.outgoing_events[0]["action_uid"],
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
            "final_transcript": "hi",
        },
    )
    assert is_data_in_events(
        state.outgoing_events,
        [
            {
                "type": "StartUtteranceBotAction",
                "script": "hi",
            }
        ],
    )


# def test_stop_flow():
#     """"""

#     content = """
#     flow user said $transcript
#       match UtteranceUserAction.Finished(final_transcript=$transcript)

#     flow user said failed $transcript
#       match (user said $transcript).Failed()
#       start UtteranceBotAction(script="flow user said {$transcript} failed")

#     flow main
#       activate user said failed "hi" and user said failed "bye"
#       start user said "hi" and user said "bye"

#       match UtteranceUserAction.Finished(final_transcript="step 1")
#       stop user said "hi"
#       match UtteranceUserAction.Finished(final_transcript="step 2")
#       stop user said "bye"

#       start user said "hi" and user said "bye"
#       match UtteranceUserAction.Finished(final_transcript="step 3")
#       stop user said
#     """

#     config = _init_state(content)
#     state = run_to_completion(config, start_main_flow_event)
#     assert is_data_in_events(
#         state.outgoing_events,
#         [],
#     )
#     state = run_to_completion(
#         state,
#         {
#             "type": "UtteranceUserActionFinished",
#             "final_transcript": "step 1",
#         },
#     )
#     assert is_data_in_events(
#         state.outgoing_events,
#         [
#             {
#                 "type": "StartUtteranceBotAction",
#                 "script": "flow user said hi failed",
#             },
#         ],
#     )
#     state = run_to_completion(
#         state,
#         {
#             "type": "UtteranceUserActionFinished",
#             "final_transcript": "step 2",
#         },
#     )
#     assert is_data_in_events(
#         state.outgoing_events,
#         [
#             {
#                 "type": "StartUtteranceBotAction",
#                 "script": "flow user said bye failed",
#             },
#         ],
#     )
#     state = run_to_completion(
#         state,
#         {
#             "type": "UtteranceUserActionFinished",
#             "final_transcript": "step 3",
#         },
#     )
#     assert is_data_in_events(
#         state.outgoing_events,
#         [
#             {
#                 "type": "StartUtteranceBotAction",
#                 "script": "flow user said hi failed",
#             },
#             {
#                 "type": "StartUtteranceBotAction",
#                 "script": "flow user said bye failed",
#             },
#         ],
#     )


def test_stop_bot_action():
    """Test stopping an action."""

    content = """
    flow main
      start UtteranceBotAction(script="This is a long sentence...") as $ref
      match UtteranceUserAction.Finished(final_transcript="Stop")
      send $ref.Stop()
    """

    config = _init_state(content)
    state = run_to_completion(config, start_main_flow_event)
    assert is_data_in_events(
        state.outgoing_events,
        [
            {
                "type": "StartUtteranceBotAction",
                "script": "This is a long sentence...",
            }
        ],
    )
    state = run_to_completion(
        state,
        {"type": "UtteranceUserActionFinished", "final_transcript": "Stop"},
    )
    assert is_data_in_events(
        state.outgoing_events,
        [
            {
                "type": "StopUtteranceBotAction",
            }
        ],
    )


def test_list_parameters():
    """Test list flow parameters."""

    content = """
    flow bot say $scripts
      await UtteranceBotAction(script=$scripts[0])

    flow main
      start bot say ["Hi", "Hello"]
    """

    config = _init_state(content)
    state = run_to_completion(config, start_main_flow_event)
    assert is_data_in_events(
        state.outgoing_events,
        [
            {
                "type": "StartUtteranceBotAction",
                "script": "Hi",
            }
        ],
    )


def test_dict_parameters():
    """Test dict flow parameters."""

    content = """
    flow bot say $scripts
      await UtteranceBotAction(script=$scripts["value2"])

    flow main
      start bot say {"value1": "something", "value2": "Hello"}
    """

    config = _init_state(content)
    state = run_to_completion(config, start_main_flow_event)
    assert is_data_in_events(
        state.outgoing_events,
        [
            {
                "type": "StartUtteranceBotAction",
                "script": "Hello",
            }
        ],
    )


def test_mixed_multimodal_group_actions():
    """Testing common used case of mixed multimodal flows."""

    content = """
    flow bot say $text
      await UtteranceBotAction(script=$text) as $action

    flow bot gesture $gesture
      await GestureBotAction(gesture=$gesture) as $action

    flow bot express $text
      await bot say $text

    flow bot express feeling well
      bot express "I am good!"
        and (bot gesture "Thumbs up" or bot gesture "Smile")

    flow bot express feeling bad
      bot express "I am not good!"
        and (bot gesture "Thumbs down" or bot gesture "Sad face")

    flow main
      bot express feeling well
        or bot express feeling bad
      match NeverComingEvent()
    """

    config = _init_state(content)
    state = run_to_completion(config, start_main_flow_event)
    assert is_data_in_events(
        state.outgoing_events,
        [
            {
                "type": "StartUtteranceBotAction",
            },
            {
                "type": "StartGestureBotAction",
            },
        ],
    )
    gesture_action_uid = state.outgoing_events[1]["action_uid"]
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
    state = run_to_completion(
        state,
        {
            "type": "GestureBotActionFinished",
            "action_uid": gesture_action_uid,
        },
    )
    assert is_data_in_events(
        state.outgoing_events,
        [],
    )


def test_iternal_unhandled_event():
    """Test unhandled event handling."""

    content = """
    flow undefined flows
      match UnhandledEvent(event="StartFlow") as $event
      await UtteranceBotAction(script="Undefined flow: {$event.flow_id}")

    flow unexpected user utterance
      match UnhandledEvent(event="UtteranceUserActionFinished") as $event
      await UtteranceBotAction(script="Unexpected user utterance: {$event.final_transcript}")

    flow main
      activate undefined flows
      activate unexpected user utterance
      start test
      match NeverComingEvent()
    """

    config = _init_state(content)
    state = run_to_completion(config, start_main_flow_event)
    assert is_data_in_events(
        state.outgoing_events,
        [
            {
                "type": "StartUtteranceBotAction",
                "script": "Undefined flow: test",
            }
        ],
    )
    state = run_to_completion(
        state,
        {
            "type": "UtteranceUserActionFinished",
            "final_transcript": "blabla",
        },
    )
    assert is_data_in_events(
        state.outgoing_events,
        [
            {
                "type": "StartUtteranceBotAction",
                "script": "Unexpected user utterance: blabla",
            }
        ],
    )


def test_references_in_groups():
    """Test use of references in groups in child flow."""

    content = """
    flow bot say $script
      await UtteranceBotAction(script=$script) as $action

    flow bot greets
      bot say "Hello" as $ref
        or bot say "Hi" as $ref

    flow main
      bot greets as $ref
      bot say $ref.ref.context.action.context.script
      match NeverComingEvent()
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
    answer = state.outgoing_events[0]["script"]
    state = run_to_completion(
        state,
        {
            "type": "UtteranceBotActionFinished",
            "final_script": answer,
            "action_uid": state.outgoing_events[0]["action_uid"],
        },
    )
    assert is_data_in_events(
        state.outgoing_events,
        [
            {
                "type": "StartUtteranceBotAction",
                "script": answer,
            }
        ],
    )


def test_regular_expressions_action_parameters():
    """Test use of regex in action parameters."""

    content = """
    flow main
      match UtteranceUserAction.Finished(final_transcript=regex("\\\\bmatch\\\\b"))
      send StartUtteranceBotAction(script="Success")
    """

    config = _init_state(content)
    state = run_to_completion(config, start_main_flow_event)
    assert is_data_in_events(
        state.outgoing_events,
        [],
    )
    state = run_to_completion(
        state,
        {
            "type": "UtteranceUserActionFinished",
            "final_transcript": "Hi! This is a match !",
        },
    )
    assert is_data_in_events(
        state.outgoing_events,
        [
            {
                "type": "StartUtteranceBotAction",
                "script": "Success",
            }
        ],
    )


def test_regular_expressions_flow_parameters():
    """Test use of regex in flow parameters."""

    content = """
    flow user said $transcript
      match UtteranceUserAction.Finished(final_transcript=$transcript)

    flow main
      user said (regex("\\\\bmatch\\\\b"))
      send StartUtteranceBotAction(script="Success")
    """

    config = _init_state(content)
    state = run_to_completion(config, start_main_flow_event)
    assert is_data_in_events(
        state.outgoing_events,
        [],
    )
    state = run_to_completion(
        state,
        {
            "type": "UtteranceUserActionFinished",
            "final_transcript": "Hi! This is a match !",
        },
    )
    assert is_data_in_events(
        state.outgoing_events,
        [
            {
                "type": "StartUtteranceBotAction",
                "script": "Success",
            }
        ],
    )


def test_expr_func_search():
    """Test Colang function search."""

    content = """
    flow main
      if search('\\\\bmatch\\\\b', 'dsfkjdsfhds match sfsd')
        send StartUtteranceBotAction(script="Success")
    """

    config = _init_state(content)
    state = run_to_completion(config, start_main_flow_event)
    assert is_data_in_events(
        state.outgoing_events,
        [
            {
                "type": "StartUtteranceBotAction",
                "script": "Success",
            }
        ],
    )


def test_generate_value_with_NLD():
    """Test NLD instructions."""

    content = """
    flow main
      #$test = await GenerateValueAction(var_name="number", instructions="Extract the number the user guessed.")
      $test = ...'Generate a random number'
      send StartUtteranceBotAction(script="{$test}")
    """

    config = _init_state(content)
    state = run_to_completion(config, start_main_flow_event)
    assert is_data_in_events(
        state.outgoing_events,
        [
            {
                "type": "StartGenerateValueAction",
                "instructions": "Generate a random number",
            }
        ],
    )


def test_flow_states_info():
    """Test the generation of flow states info."""

    content = """
    flow a
        match Event1()

    flow main
      activate a
      $fs = flows_info()
      if $self.uid in $fs
        await UtteranceBotAction(script="Success")
      else
        await UtteranceBotAction(script="Failed")
    """

    config = _init_state(content)
    state = run_to_completion(config, start_main_flow_event)
    assert is_data_in_events(
        state.outgoing_events,
        [
            {
                "type": "StartUtteranceBotAction",
                "script": "Success",
            }
        ],
    )


if __name__ == "__main__":
    test_flow_states_info()
