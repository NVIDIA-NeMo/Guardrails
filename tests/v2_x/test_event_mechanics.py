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

from nemoguardrails import RailsConfig
from nemoguardrails.colang.v2_x.runtime.flows import ActionStatus
from nemoguardrails.colang.v2_x.runtime.statemachine import (
    InternalEvent,
    run_to_completion,
)
from tests.utils import TestChat, _init_state, is_data_in_events

FORMAT = "%(message)s"
logging.basicConfig(
    level=logging.DEBUG,
    format=FORMAT,
    datefmt="[%X,%f]",
    handlers=[RichHandler(markup=True)],
)

start_main_flow_event = InternalEvent(name="StartFlow", arguments={"flow_id": "main"})


def test_send_umim_action_event():
    """Test to send an UMIM event."""

    content = """
    flow main
      send StartUtteranceBotAction(script="Hello world")
    """

    state = run_to_completion(_init_state(content), start_main_flow_event)
    assert is_data_in_events(
        state.outgoing_events,
        [
            {
                "type": "StartUtteranceBotAction",
                "script": "Hello world",
            }
        ],
    )


def test_send_umim_action_event_overwriting_default_parameters():
    """Test to send an UMIM event but overwrite default parameters."""

    content = """
    flow main
      $fixed_timestamp = "2024-10-22T12:08:18.874224"
      $uid = "1234"
      send UtteranceBotActionFinished(final_script="Hello world", action_finished_at=$fixed_timestamp, action_uid=$uid, is_success=True)
    """

    state = run_to_completion(_init_state(content), start_main_flow_event)
    assert is_data_in_events(
        state.outgoing_events,
        [
            {
                "type": "UtteranceBotActionFinished",
                "final_script": "Hello world",
                "action_uid": "1234",
                "action_finished_at": "2024-10-22T12:08:18.874224",
            }
        ],
    )


def test_change_umim_event_source_id():
    """Test to send an UMIM event."""

    content = """
    flow main
      send StartUtteranceBotAction(script="Hello world")
    """

    config = """
    colang_version: "2.x"
    event_source_uid : agent-1
    """

    state = run_to_completion(_init_state(content, config), start_main_flow_event)
    assert is_data_in_events(
        state.outgoing_events,
        [
            {
                "type": "StartUtteranceBotAction",
                "script": "Hello world",
                "source_uid": "agent-1",
            }
        ],
    )


def test_match_umim_action_event():
    """Test to match an UMIM event."""

    content = """
    flow main
      match UtteranceUserAction.Finished(final_transcript="Hi")
      send StartUtteranceBotAction(script="Hello world")
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
            "uid": "d4a265bb-4a27-4d28-8ca5-80cc73dc4707",
            "event_created_at": "2023-09-12T13:01:16.334940+00:00",
            "source_uid": "umim_tui_app",
            "action_uid": "cc63b1a0-5703-4e80-b66b-2734c13abcf3",
            "final_transcript": "Hi",
            "is_success": True,
            "action_info_modality": "user_speech",
            "action_info_modality_policy": "replace",
            "action_finished_at": "2023-09-12T13:01:16.334954+00:00",
        },
    )
    assert is_data_in_events(
        state.outgoing_events,
        [
            {
                "type": "StartUtteranceBotAction",
                "script": "Hello world",
            }
        ],
    )


def test_start_action():
    """Test to start an UMIM action"""

    content = """
    flow main
      start UtteranceBotAction(script="Hello world")
    """
    state = run_to_completion(_init_state(content), start_main_flow_event)
    assert is_data_in_events(
        state.outgoing_events,
        [
            {
                "type": "StartUtteranceBotAction",
                "script": "Hello world",
            }
        ],
    )


def test_start_match_action_on_action_parameter():
    """Test to start and match an UMIM action based on action parameters."""

    content = """
    flow main
      start UtteranceBotAction(script="Hello world")
      match UtteranceBotAction(script="Hello world").Finished()
      start UtteranceBotAction(script="Done")
    """
    state = run_to_completion(_init_state(content), start_main_flow_event)
    assert is_data_in_events(
        state.outgoing_events,
        [
            {
                "type": "StartUtteranceBotAction",
                "script": "Hello world",
            }
        ],
    )
    state = run_to_completion(
        state,
        {
            "type": "UtteranceBotActionFinished",
            "final_script": "Hello world",
            "action_uid": state.outgoing_events[0]["action_uid"],
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


def test_start_mismatch_action_on_action_parameter():
    """Test to start and mismatch an UMIM action based on action parameters."""

    content = """
    flow main
      start UtteranceBotAction(script="Hello world")
      match UtteranceBotAction(script="Hello").Finished()
      start UtteranceBotAction(script="Done")
    """
    state = run_to_completion(_init_state(content), start_main_flow_event)
    assert is_data_in_events(
        state.outgoing_events,
        [
            {
                "type": "StartUtteranceBotAction",
                "script": "Hello world",
            }
        ],
    )
    state = run_to_completion(
        state,
        {
            "type": "UtteranceBotActionFinished",
            "final_script": "Hello world",
            "action_uid": state.outgoing_events[0]["action_uid"],
        },
    )
    assert is_data_in_events(
        state.outgoing_events,
        [],
    )


def test_start_match_action_on_event_parameter():
    """Test to start and match an UMIM action based on action parameters."""

    content = """
    flow main
      start UtteranceBotAction(script="Hello world")
      match UtteranceBotAction.Finished(final_script="Hello world")
      start UtteranceBotAction(script="Done")
    """
    state = run_to_completion(_init_state(content), start_main_flow_event)
    assert is_data_in_events(
        state.outgoing_events,
        [
            {
                "type": "StartUtteranceBotAction",
                "script": "Hello world",
            }
        ],
    )
    state = run_to_completion(
        state,
        {
            "type": "UtteranceBotActionFinished",
            "final_script": "Hello world",
            "action_uid": state.outgoing_events[0]["action_uid"],
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


def test_start_mismatch_action_on_event_parameter():
    """Test to start and mismatch an UMIM action based on event parameters."""

    content = """
    flow main
      start UtteranceBotAction(script="Hello world")
      match UtteranceBotAction.Finished(final_script="Hello")
      start UtteranceBotAction(script="Done")
    """
    state = run_to_completion(_init_state(content), start_main_flow_event)
    assert is_data_in_events(
        state.outgoing_events,
        [
            {
                "type": "StartUtteranceBotAction",
                "script": "Hello world",
            }
        ],
    )
    state = run_to_completion(
        state,
        {
            "type": "UtteranceBotActionFinished",
            "final_script": "Hello world",
            "action_uid": state.outgoing_events[0]["action_uid"],
        },
    )
    assert is_data_in_events(
        state.outgoing_events,
        [],
    )


def test_event_match_group():
    """Test mixed and/or event match groups."""

    content = """
    flow a
      start b as $ref_b and c as $ref_c
      match ($ref_b.Finished() and $ref_c.Finished()) or ($ref_b.Failed() and $ref_c.Failed())
      start UtteranceBotAction(script="Hello world")

    flow b
      match WaitEvent()

    flow c
      match WaitEvent()

    flow main
      activate a
      match Event1()
      send FinishFlow(flow_id="b")
      match Event2()
      send StopFlow(flow_id="c")
      match Event1()
      send StopFlow(flow_id="b")
      match Event2()
      send StopFlow(flow_id="c")
      match WaitEvent()
    """
    state = run_to_completion(_init_state(content), start_main_flow_event)
    assert is_data_in_events(
        state.outgoing_events,
        [],
    )
    state = run_to_completion(
        state,
        {
            "type": "Event1",
        },
    )
    assert is_data_in_events(
        state.outgoing_events,
        [],
    )
    state = run_to_completion(
        state,
        {
            "type": "Event2",
        },
    )
    assert is_data_in_events(
        state.outgoing_events,
        [],
    )
    state = run_to_completion(
        state,
        {
            "type": "Event1",
        },
    )
    assert is_data_in_events(
        state.outgoing_events,
        [],
    )
    state = run_to_completion(
        state,
        {
            "type": "Event2",
        },
    )
    assert is_data_in_events(
        state.outgoing_events,
        [
            {
                "type": "StartUtteranceBotAction",
                "script": "Hello world",
            },
            {
                "type": "StopUtteranceBotAction",
            },
        ],
    )


def test_finished_or_failed_flow_event_match():
    """Test matching to the finished and failed event of a flow in a single or group."""

    content = """
    flow a
      start b as $ref
      match $ref.Finished() or $ref.Failed()
      await UtteranceBotAction(script="Hello world")

    flow b
      match WaitEvent()

    flow main
      start a
      send StopFlow(flow_id="b")
    """
    state = run_to_completion(_init_state(content), start_main_flow_event)
    assert is_data_in_events(
        state.outgoing_events,
        [
            {
                "type": "StartUtteranceBotAction",
                "script": "Hello world",
            }
        ],
    )


def test_start_match_action_with_reference():
    """Test to start and match an UMIM action based on action reference."""

    content = """
    flow main
      start UtteranceBotAction(script="Hello world") as $action_ref
      match $action_ref.Finished()
      start UtteranceBotAction(script="Done")
    """
    state = run_to_completion(_init_state(content), start_main_flow_event)
    assert is_data_in_events(
        state.outgoing_events,
        [
            {
                "type": "StartUtteranceBotAction",
                "script": "Hello world",
            }
        ],
    )
    state = run_to_completion(
        state,
        {
            "type": "UtteranceBotActionFinished",
            "final_script": "Hello world",
            "action_uid": state.outgoing_events[0]["action_uid"],
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


def test_await_action():
    """Test to await an UMIM action."""

    content = """
    flow main
      await UtteranceBotAction(script="Hello world")
      start UtteranceBotAction(script="Done")
    """
    state = run_to_completion(_init_state(content), start_main_flow_event)
    assert is_data_in_events(
        state.outgoing_events,
        [
            {
                "type": "StartUtteranceBotAction",
                "script": "Hello world",
            }
        ],
    )
    state = run_to_completion(
        state,
        {
            "type": "UtteranceBotActionFinished",
            "final_script": "Hello world",
            "action_uid": state.outgoing_events[0]["action_uid"],
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


def test_implicit_action_state_update():
    """Test the implicit action state update."""

    content = """
    flow main
      start UtteranceBotAction(script="Hello world") as $action_ref1
      start UtteranceBotAction(script="Hi") as $action_ref2
      match $action_ref1.Finished()
    """
    state = run_to_completion(_init_state(content), start_main_flow_event)
    assert is_data_in_events(
        state.outgoing_events,
        [
            {
                "type": "StartUtteranceBotAction",
                "script": "Hello world",
            },
            {
                "type": "StartUtteranceBotAction",
                "script": "Hi",
            },
        ],
    )
    action_uid = state.outgoing_events[1]["action_uid"]
    state = run_to_completion(
        state,
        {
            "type": "UtteranceBotActionFinished",
            "final_script": "Hi",
            "action_uid": action_uid,
        },
    )
    assert state.actions[action_uid].status == ActionStatus.FINISHED


def test_start_a_flow_expanded_notation():
    """Test the start of a child flow with expanded notation."""

    content = """
    flow a
      start UtteranceBotAction(script="Hello world")

    flow main
      # start a
      send StartFlow(flow_id="a", flow_instance_uid="01234")
      match FlowStarted(flow_id="a", flow_instance_uid="01234")
    """

    state = run_to_completion(_init_state(content), start_main_flow_event)
    assert is_data_in_events(
        state.outgoing_events,
        [
            {
                "type": "StartUtteranceBotAction",
                "script": "Hello world",
            },
            {
                "type": "StopUtteranceBotAction",
            },
        ],
    )


def test_start_a_flow_short_notation():
    """Test the start of a child flow using short notation."""

    content = """
    flow a
      start UtteranceBotAction(script="Hello world")

    flow main
      start a
    """

    state = run_to_completion(_init_state(content), start_main_flow_event)
    assert is_data_in_events(
        state.outgoing_events,
        [
            {
                "type": "StartUtteranceBotAction",
                "script": "Hello world",
            },
            {
                "type": "StopUtteranceBotAction",
            },
        ],
    )


def test_start_match_flow_with_reference():
    """Test to start and match end of a flow based on flow reference."""

    content = """
    flow bot say hello
      await UtteranceBotAction(script="Hello") as $action_ref

    flow main
      start bot say hello as $flow_ref
      match $flow_ref.Finished()
      start UtteranceBotAction(script="Done")
    """
    state = run_to_completion(_init_state(content), start_main_flow_event)
    assert is_data_in_events(
        state.outgoing_events,
        [
            {
                "type": "StartUtteranceBotAction",
                "script": "Hello",
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


def test_user_action_reference():
    """Test user action event matching based on references."""

    content = """
    flow main
      match UtteranceUserAction.Started() as $event_ref
      start UtteranceBotAction(script="Started user action: {$event_ref.action.name}")
      match $event_ref.action.Finished(final_transcript="End")
      start UtteranceBotAction(script="Success")
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
            "type": "UtteranceUserActionStarted",
            "uid": "d4a265bb-4a27-4d28-8ca5-80cc73dc4707",
            "event_created_at": "2023-09-12T13:01:16.334940+00:00",
            "source_uid": "umim_tui_app",
            "action_uid": "cc63b1a0-5703-4e80-b66b-2734c13abcf3",
            "action_info_modality": "user_speech",
        },
    )
    assert is_data_in_events(
        state.outgoing_events,
        [
            {
                "type": "StartUtteranceBotAction",
                "script": "Started user action: UtteranceUserAction",
            },
        ],
    )
    state = run_to_completion(
        state,
        {
            "type": "UtteranceUserActionFinished",
            "uid": "d4a265bb-4a27-4d28-8ca5-80cc73dc4707",
            "event_created_at": "2023-09-12T13:01:16.334940+00:00",
            "source_uid": "umim_tui_app",
            "action_uid": "cc63b1a0-5703-4e80-b66b-2734c13abcf3",
            "final_transcript": "End",
            "is_success": True,
            "action_info_modality": "user_speech",
            "action_info_modality_policy": "replace",
            "action_finished_at": "2023-09-12T13:01:16.334954+00:00",
        },
    )
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
                "type": "StopUtteranceBotAction",
            },
        ],
    )


def test_await_a_flow():
    """Test await a child flow."""

    content = """
    flow a
      start UtteranceBotAction(script="Flow a started")

    flow main
      # await a
      send StartFlow(flow_id="a", flow_instance_uid="01234")
      match FlowStarted(flow_id="a", flow_instance_uid="01234")
      match FlowFinished(flow_id="a", flow_instance_uid="01234")
      start UtteranceBotAction(script="Flow a finished")
    """

    state = run_to_completion(_init_state(content), start_main_flow_event)
    assert is_data_in_events(
        state.outgoing_events,
        [
            {
                "type": "StartUtteranceBotAction",
                "script": "Flow a started",
            },
            {
                "type": "StopUtteranceBotAction",
            },
            {
                "type": "StartUtteranceBotAction",
                "script": "Flow a finished",
            },
            {
                "type": "StopUtteranceBotAction",
            },
        ],
    )


def test_await_a_flow_short_notation():
    """Test await a child flow with short notation."""

    content = """
    flow a
      start UtteranceBotAction(script="Flow a started")

    flow main
      await a
      start UtteranceBotAction(script="Flow a finished")
    """

    state = run_to_completion(_init_state(content), start_main_flow_event)
    assert is_data_in_events(
        state.outgoing_events,
        [
            {
                "type": "StartUtteranceBotAction",
                "script": "Flow a started",
            },
            {
                "type": "StopUtteranceBotAction",
            },
            {
                "type": "StartUtteranceBotAction",
                "script": "Flow a finished",
            },
            {
                "type": "StopUtteranceBotAction",
            },
        ],
    )


def test_await_multimodal_action():
    """Test await statement with multimodal actions."""

    content = """
    flow bot say $text
      await UtteranceBotAction(script=$text) as $action

    flow bot gesture $gesture
      await GestureBotAction(gesture=$gesture) as $action

    flow bot express $text
      await bot say $text

    flow main
        start bot express "Hi"
        start bot gesture "Wave"
        match UtteranceUserAction().Finished()
    """

    state = run_to_completion(_init_state(content), start_main_flow_event)
    assert is_data_in_events(
        state.outgoing_events,
        [
            {
                "type": "StartUtteranceBotAction",
                "script": "Hi",
            },
            {
                "type": "StartGestureBotAction",
                "gesture": "Wave",
            },
        ],
    )


def test_start_child_flow_two_times():
    """Test start a child flow two times."""

    content = """
    flow a
      await UtteranceBotAction(script="Hi")

    flow main
      start a
      await a
    """

    state = run_to_completion(_init_state(content), start_main_flow_event)
    assert is_data_in_events(
        state.outgoing_events,
        [
            {
                "type": "StartUtteranceBotAction",
                "script": "Hi",
            },
            {
                "type": "StartUtteranceBotAction",
                "script": "Hi",
            },
        ],
    )


def test_simple_event_parameter_match():
    """Test a simple event parameter match."""

    content = """
    flow main
      match Event1(a=1)
      start UtteranceBotAction(script="OK1")
      match Event1(a=1)
      await UtteranceBotAction(script="OK2")
    """
    state = run_to_completion(_init_state(content), start_main_flow_event)
    assert is_data_in_events(
        state.outgoing_events,
        [],
    )
    state = run_to_completion(
        state,
        {
            "type": "Event1",
            "a": 1,
        },
    )
    assert is_data_in_events(
        state.outgoing_events,
        [
            {
                "type": "StartUtteranceBotAction",
                "script": "OK1",
            }
        ],
    )
    state = run_to_completion(
        state,
        {
            "type": "Event1",
            "a": 2,
        },
    )
    assert is_data_in_events(
        state.outgoing_events,
        [],
    )
    state = run_to_completion(
        state,
        {
            "type": "Event1",
            "b": 2,
        },
    )
    assert is_data_in_events(
        state.outgoing_events,
        [],
    )
    state = run_to_completion(
        state,
        {
            "type": "Event1",
        },
    )
    assert is_data_in_events(
        state.outgoing_events,
        [],
    )
    state = run_to_completion(
        state,
        {
            "type": "Event1",
            "a": 1,
            "b": 1,
        },
    )
    assert is_data_in_events(
        state.outgoing_events,
        [
            {
                "type": "StartUtteranceBotAction",
                "script": "OK2",
            }
        ],
    )


def test_dict_event_parameter_match():
    """Test event match with dictionary parameters."""

    content = """
    flow main
      match Event1(param={"a":1})
      start UtteranceBotAction(script="OK1")
      match Event1(param={"a":1})
      await UtteranceBotAction(script="OK2")
    """
    state = run_to_completion(_init_state(content), start_main_flow_event)
    assert is_data_in_events(
        state.outgoing_events,
        [],
    )
    state = run_to_completion(
        state,
        {
            "type": "Event1",
            "param": {"a": 1},
        },
    )
    assert is_data_in_events(
        state.outgoing_events,
        [
            {
                "type": "StartUtteranceBotAction",
                "script": "OK1",
            }
        ],
    )
    state = run_to_completion(
        state,
        {
            "type": "Event1",
            "param": {"a": 2},
        },
    )
    assert is_data_in_events(
        state.outgoing_events,
        [],
    )
    state = run_to_completion(
        state,
        {
            "type": "Event1",
            "param": {"b": 1},
        },
    )
    assert is_data_in_events(
        state.outgoing_events,
        [],
    )
    state = run_to_completion(
        state,
        {
            "type": "Event1",
            "param": {},
        },
    )
    assert is_data_in_events(
        state.outgoing_events,
        [],
    )
    state = run_to_completion(
        state,
        {
            "type": "Event1",
            "param": {"a": 1, "b": 1},
        },
    )
    assert is_data_in_events(
        state.outgoing_events,
        [
            {
                "type": "StartUtteranceBotAction",
                "script": "OK2",
            }
        ],
    )


def test_set_event_parameter_match():
    """Test event match with set parameters."""

    content = """
    flow main
      match Event1(param={"a"})
      start UtteranceBotAction(script="OK1")
      match Event1(param={regex(".*")})
      start UtteranceBotAction(script="OK2")
      match Event1(param={"c","a"})
      await UtteranceBotAction(script="OK3")
    """
    state = run_to_completion(_init_state(content), start_main_flow_event)
    assert is_data_in_events(
        state.outgoing_events,
        [],
    )
    state = run_to_completion(
        state,
        {
            "type": "Event1",
            "param": {"c"},
        },
    )
    assert is_data_in_events(
        state.outgoing_events,
        [],
    )
    state = run_to_completion(
        state,
        {
            "type": "Event1",
            "param": {"a", "b"},
        },
    )
    assert is_data_in_events(
        state.outgoing_events,
        [
            {
                "type": "StartUtteranceBotAction",
                "script": "OK1",
            }
        ],
    )
    state = run_to_completion(
        state,
        {
            "type": "Event1",
            "param": {},
        },
    )
    assert is_data_in_events(
        state.outgoing_events,
        [],
    )
    state = run_to_completion(
        state,
        {
            "type": "Event1",
            "param": {"sdfsd"},
        },
    )
    assert is_data_in_events(
        state.outgoing_events,
        [
            {
                "type": "StartUtteranceBotAction",
                "script": "OK2",
            }
        ],
    )
    state = run_to_completion(
        state,
        {
            "type": "Event1",
            "param": {"a", "b"},
        },
    )
    assert is_data_in_events(
        state.outgoing_events,
        [],
    )
    state = run_to_completion(
        state,
        {
            "type": "Event1",
            "param": {"a", "c"},
        },
    )
    assert is_data_in_events(
        state.outgoing_events,
        [
            {
                "type": "StartUtteranceBotAction",
                "script": "OK3",
            }
        ],
    )


def test_list_event_parameter_match():
    """Test event match with list parameters."""

    content = """
    flow main
      match Event1(param=[1,2])
      start UtteranceBotAction(script="OK1")
      match Event1(param=[1,2])
      start UtteranceBotAction(script="OK2")
      match Event1(param=[regex(".*"),2])
      await UtteranceBotAction(script="OK3")
    """
    state = run_to_completion(_init_state(content), start_main_flow_event)
    assert is_data_in_events(
        state.outgoing_events,
        [],
    )
    state = run_to_completion(
        state,
        {
            "type": "Event1",
            "param": [1],
        },
    )
    assert is_data_in_events(
        state.outgoing_events,
        [],
    )
    state = run_to_completion(
        state,
        {
            "type": "Event1",
            "param": [2, 1],
        },
    )
    assert is_data_in_events(
        state.outgoing_events,
        [],
    )
    state = run_to_completion(
        state,
        {
            "type": "Event1",
            "param": [0, 1, 5, 2, 0],
        },
    )
    assert is_data_in_events(
        state.outgoing_events,
        [
            {
                "type": "StartUtteranceBotAction",
                "script": "OK1",
            }
        ],
    )
    state = run_to_completion(
        state,
        {
            "type": "Event1",
            "param": [2],
        },
    )
    assert is_data_in_events(
        state.outgoing_events,
        [],
    )
    state = run_to_completion(
        state,
        {
            "type": "Event1",
            "param": [],
        },
    )
    assert is_data_in_events(
        state.outgoing_events,
        [],
    )
    state = run_to_completion(
        state,
        {
            "type": "Event1",
            "param": [2, 1],
        },
    )
    assert is_data_in_events(
        state.outgoing_events,
        [],
    )
    state = run_to_completion(
        state,
        {
            "type": "Event1",
            "param": [1, 2, 3],
        },
    )
    assert is_data_in_events(
        state.outgoing_events,
        [
            {
                "type": "StartUtteranceBotAction",
                "script": "OK2",
            }
        ],
    )
    state = run_to_completion(
        state,
        {
            "type": "Event1",
            "param": [5, 2],
        },
    )
    assert is_data_in_events(
        state.outgoing_events,
        [
            {
                "type": "StartUtteranceBotAction",
                "script": "OK3",
            }
        ],
    )


def test_custom_regex_event_parameter_match():
    """Test more complex regex event parameters."""

    content = """
    flow main
      while True
        when VisualFormSceneAction.InputUpdated(interim_inputs=[{"id": regex("\\\\bemail\\\\b"), "value": regex(".*")}]) as $e
          start UtteranceBotAction(script="Success")
    """
    state = run_to_completion(_init_state(content), start_main_flow_event)
    assert is_data_in_events(
        state.outgoing_events,
        [],
    )
    state = run_to_completion(
        state,
        {
            "type": "VisualFormSceneActionInputUpdated",
            "interim_inputs": [{"id": "email", "value": "test"}],
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
    state = run_to_completion(
        state,
        {
            "type": "VisualFormSceneActionInputUpdated",
            "interim_inputs": [{"id": "sdfsdf", "value": "test"}],
        },
    )
    assert is_data_in_events(
        state.outgoing_events,
        [],
    )
    state = run_to_completion(
        state,
        {
            "type": "VisualFormSceneActionInputUpdated",
            "interim_inputs": [{"id": "sdfsdf email sdf", "value": ""}],
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


def test_corner_cases_regex_event_parameter_matching():
    """Test corner cases for regex matches."""

    content = """
    flow main
      while True
        when VisualFormSceneAction.InputUpdated(interim_inputs=[{"id" : "ter", "r": regex("[ab]+"), "value": regex("^[r'a]+$")}]) as $e
          start UtteranceBotAction(script="Success")
    """

    state = run_to_completion(_init_state(content), start_main_flow_event)
    assert is_data_in_events(
        state.outgoing_events,
        [],
    )
    state = run_to_completion(
        state,
        {
            "type": "VisualFormSceneActionInputUpdated",
            "interim_inputs": [{"id": "ter", "r": "abbbaa", "value": "rar'araaaaarr"}],
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
    state = run_to_completion(
        state,
        {
            "type": "VisualFormSceneActionInputUpdated",
            "interim_inputs": [{"id": "ter", "r": "abbbaa", "value": 'rar"araaaaarr'}],
        },
    )
    assert is_data_in_events(
        state.outgoing_events,
        [],
    )


def test_action_event_requeuing():
    """Test queuing of action events."""
    config = RailsConfig.from_content(
        colang_content="""
        flow main
          match UtteranceUserAction.Finished(final_transcript="start")
          start UtteranceBotAction(script="started")
          match UtteranceBotAction.Started()
          start UtteranceBotAction(script="success")
        """,
        yaml_content="""
        colang_version: "2.x"
        """,
    )

    chat = TestChat(
        config,
        llm_completions=[],
    )

    chat >> "start"
    chat << "started\nsuccess"


def test_context_update():
    """Test to update the context through events."""

    content = """
    flow main
      global $test
      match UtteranceUserAction.Finished(final_transcript="step1")
      start UtteranceBotAction(script="{$test}")
      match UtteranceUserAction.Finished(final_transcript="step2")
      start UtteranceBotAction(script="{$test}")
    """
    state = run_to_completion(_init_state(content), start_main_flow_event)
    assert is_data_in_events(
        state.outgoing_events,
        [],
    )
    state = run_to_completion(
        state,
        {
            "type": "ContextUpdate",
            "data": {
                "test": 13,
            },
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
            "final_transcript": "step1",
        },
    )
    assert is_data_in_events(
        state.outgoing_events,
        [
            {
                "type": "StartUtteranceBotAction",
                "script": "13",
            },
        ],
    )
    state = run_to_completion(
        state,
        {
            "type": "ContextUpdate",
            "data": {
                "test": 5,
            },
        },
    )
    state = run_to_completion(
        state,
        {
            "type": "UtteranceUserActionFinished",
            "final_transcript": "step2",
        },
    )
    assert is_data_in_events(
        state.outgoing_events,
        [
            {
                "type": "StartUtteranceBotAction",
                "script": "5",
            },
            {
                "type": "StopUtteranceBotAction",
            },
            {
                "type": "StopUtteranceBotAction",
            },
        ],
    )


def test_match_hierarchy_of_internal_events():
    """Test internal event matching for a typical flow hierarchy."""

    content = """
    flow _bot_say $text
      await UtteranceBotAction(script=$text) as $action

    flow bot say $text
      await _bot_say $text

    flow bot started saying something
      match FlowStarted(flow_id="_bot_say") as $event

    flow bot express $text
      await _bot_say $text

    flow bot express greeting
      await bot express "hi"
        or bot express "hello"

    flow a
      await bot started saying something
      bot say "test"

    flow main
      activate a
      bot express greeting
    """
    state = run_to_completion(_init_state(content), start_main_flow_event)
    assert is_data_in_events(
        state.outgoing_events,
        [
            {
                "type": "StartUtteranceBotAction",
            },
            {
                "type": "StartUtteranceBotAction",
                "script": "test",
            },
        ],
    )


def test_event_number_parameter_comparison():
    """Test to match events based on numeric parameters."""

    content = """
    flow main
      match Event(p=less_than(5))
      start UtteranceBotAction(script="success")
      match Event(p=equal_less_than(5))
      start UtteranceBotAction(script="success")
      match Event(p=greater_than(10.1))
      start UtteranceBotAction(script="success")
      match Event(p=equal_greater_than(10.1))
      start UtteranceBotAction(script="success")
      match Event(p=not_equal_to(100))
      start UtteranceBotAction(script="success")
      match Event()
    """
    state = run_to_completion(_init_state(content), start_main_flow_event)
    assert is_data_in_events(
        state.outgoing_events,
        [],
    )
    state = run_to_completion(
        state,
        {
            "type": "Event",
            "p": 6,
        },
    )
    assert is_data_in_events(
        state.outgoing_events,
        [],
    )
    state = run_to_completion(
        state,
        {
            "type": "Event",
            "p": 3,
        },
    )
    assert is_data_in_events(
        state.outgoing_events,
        [
            {
                "type": "StartUtteranceBotAction",
                "script": "success",
            },
        ],
    )
    state = run_to_completion(
        state,
        {
            "type": "Event",
            "p": 5,
        },
    )
    assert is_data_in_events(
        state.outgoing_events,
        [
            {
                "type": "StartUtteranceBotAction",
                "script": "success",
            },
        ],
    )
    state = run_to_completion(
        state,
        {
            "type": "Event",
            "p": 10.0,
        },
    )
    assert is_data_in_events(
        state.outgoing_events,
        [],
    )
    state = run_to_completion(
        state,
        {
            "type": "Event",
            "p": 10.2,
        },
    )
    assert is_data_in_events(
        state.outgoing_events,
        [
            {
                "type": "StartUtteranceBotAction",
                "script": "success",
            },
        ],
    )
    state = run_to_completion(
        state,
        {
            "type": "Event",
            "p": 10.1,
        },
    )
    assert is_data_in_events(
        state.outgoing_events,
        [
            {
                "type": "StartUtteranceBotAction",
                "script": "success",
            },
        ],
    )
    state = run_to_completion(
        state,
        {
            "type": "Event",
            "p": 100,
        },
    )
    assert is_data_in_events(
        state.outgoing_events,
        [],
    )
    state = run_to_completion(
        state,
        {
            "type": "Event",
            "p": 99,
        },
    )
    assert is_data_in_events(
        state.outgoing_events,
        [
            {
                "type": "StartUtteranceBotAction",
                "script": "success",
            },
        ],
    )


def test_runtime_exception_handling_1():
    """Test to match events based on numeric parameters."""

    content = """
    flow a
      $var = "test" + 3
      match WaitEvent()

    flow main
      match StartEvent()
      when a
        pass
      else
        start UtteranceBotAction(script="success")
      match Event()
    """
    state = run_to_completion(_init_state(content), start_main_flow_event)
    assert is_data_in_events(
        state.outgoing_events,
        [],
    )
    state = run_to_completion(
        state,
        {
            "type": "StartEvent",
        },
    )
    assert is_data_in_events(
        state.outgoing_events,
        [
            {
                "type": "StartUtteranceBotAction",
                "script": "success",
            },
        ],
    )


def test_runtime_exception_handling_2():
    """Test to match events based on numeric parameters."""

    content = """
    flow a
      start UtteranceBotAction(script="test")
      match StartEvent()
      $var = "test" + 3

    flow main
      activate a
      match Event()
    """
    state = run_to_completion(_init_state(content), start_main_flow_event)
    assert is_data_in_events(
        state.outgoing_events,
        [
            {
                "type": "StartUtteranceBotAction",
                "script": "test",
            },
        ],
    )
    state = run_to_completion(
        state,
        {
            "type": "StartEvent",
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
                "script": "test",
            },
        ],
    )


def test_user_message_generates_started_and_finished():
    """Test queuing of action events."""
    config = RailsConfig.from_content(
        colang_content="""
        flow main
          match UtteranceUserActionStarted()
          match UtteranceUserActionFinished(final_transcript="yes")
          start UtteranceBotAction(script="ok")
        """,
        yaml_content="""
        colang_version: "2.x"
        """,
    )

    chat = TestChat(
        config,
        llm_completions=[],
    )

    chat >> "yes"
    chat << "ok"


def test_handling_arbitrary_events_through_test_chat():
    """Test queuing of action events."""
    config = RailsConfig.from_content(
        colang_content="""
        flow main
          match CustomEvent(name="test")
          match EventA()
          start UtteranceBotAction(script="started")
        """,
        yaml_content="""
        colang_version: "2.x"
        """,
    )

    chat = TestChat(
        config,
        llm_completions=[],
    )

    chat >> {"type": "CustomEvent", "name": "test"}
    chat >> {"type": "EventA"}
    chat << "started"


if __name__ == "__main__":
    test_event_match_group()
