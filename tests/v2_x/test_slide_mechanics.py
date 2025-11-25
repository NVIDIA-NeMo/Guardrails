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


def test_while_loop_mechanic():
    """Test the while loop statement mechanic."""

    content = """
    flow main

      while $ref is None
        match UtteranceUserAction().Finished(final_transcript="End") as $ref
        start UtteranceBotAction(script="Test")

      start UtteranceBotAction(script="Done")
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
            "final_transcript": "End",
        },
    )
    assert is_data_in_events(
        state.outgoing_events,
        [
            {
                "type": "StartUtteranceBotAction",
                "script": "Test",
            },
            {
                "type": "StartUtteranceBotAction",
                "script": "Done",
            },
            {
                "type": "StopUtteranceBotAction",
            },
            {
                "type": "StopUtteranceBotAction",
            },
        ],
    )


def test_if_branching_mechanic():
    """Test if branching statement mechanism."""

    content = """
    flow main
      while $action_ref_3 is None:
        if $event_ref_1 is None and True:
          start UtteranceBotAction(script="Action1") as $event_ref_1
        else if $event_ref_2 is None or False:
          start UtteranceBotAction(script="Action2") as $event_ref_2
        else:
          start UtteranceBotAction(script="ActionElse") as $action_ref_3
        start UtteranceBotAction(script="Next")
    """

    config = _init_state(content)
    state = run_to_completion(config, start_main_flow_event)
    assert is_data_in_events(
        state.outgoing_events,
        [
            {
                "type": "StartUtteranceBotAction",
                "script": "Action1",
            },
            {
                "type": "StartUtteranceBotAction",
                "script": "Next",
            },
            {
                "type": "StartUtteranceBotAction",
                "script": "Action2",
            },
            {
                "type": "StartUtteranceBotAction",
                "script": "Next",
            },
            {
                "type": "StartUtteranceBotAction",
                "script": "ActionElse",
            },
            {
                "type": "StartUtteranceBotAction",
                "script": "Next",
            },
        ],
    )


def test_event_reference_member_access():
    """Test accessing a event reference member."""

    content = """
    flow main
      match UtteranceUserAction().Finished() as $ref
      start UtteranceBotAction(script=$ref.arguments.final_transcript)
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
            "final_transcript": "Hi there!",
        },
    )
    assert is_data_in_events(
        state.outgoing_events,
        [
            {
                "type": "StartUtteranceBotAction",
                "script": "Hi there!",
            },
            {
                "type": "StopUtteranceBotAction",
            },
        ],
    )


def test_action_reference_member_access():
    """Test accessing a action reference member."""

    content = """
    flow main
      start UtteranceBotAction(script="Hello") as $ref
      start UtteranceBotAction(script=$ref.start_event_arguments.script)
    """

    config = _init_state(content)
    state = run_to_completion(config, start_main_flow_event)
    assert is_data_in_events(
        state.outgoing_events,
        [
            {
                "type": "StartUtteranceBotAction",
                "script": "Hello",
            },
            {
                "type": "StartUtteranceBotAction",
                "script": "Hello",
            },
        ],
    )


def test_flow_references_member_access():
    """Test accessing a flow reference member."""

    content = """
    flow bot say $text
      start UtteranceBotAction(script=$text) as $action_ref

    flow main
      start bot say "Hello" as $flow_ref
      start UtteranceBotAction(script=$flow_ref.context.action_ref.start_event_arguments.script)
    """

    config = _init_state(content)
    state = run_to_completion(config, start_main_flow_event)
    assert is_data_in_events(
        state.outgoing_events,
        [
            {
                "type": "StartUtteranceBotAction",
                "script": "Hello",
            },
            {
                "type": "StopUtteranceBotAction",
            },
            {
                "type": "StartUtteranceBotAction",
                "script": "Hello",
            },
        ],
    )


def test_expressions_in_strings():
    """Test string expression evaluation."""

    content = """
    flow main
      $xyz = 'x\\'y\\'z\\n'
      $test = '''Roger's
                test:\n {$xyz}'''
      start UtteranceBotAction(script=$test) as $ref
      start UtteranceBotAction(script="It's {{->}}")
      await UtteranceBotAction(script='It"s {{->}} \\'{$ref.start_event_arguments.script}!\\'')
    """

    config = _init_state(content)
    state = run_to_completion(config, start_main_flow_event)
    assert is_data_in_events(
        state.outgoing_events,
        [
            {
                "type": "StartUtteranceBotAction",
                "script": "Roger's\n                test:\n x'y'z\n",
            },
            {
                "type": "StartUtteranceBotAction",
                "script": "It's {->}",
            },
            {
                "type": "StartUtteranceBotAction",
                "script": "It\"s {->} 'Roger's\n                test:\n x'y'z\n!'",
            },
        ],
    )


def test_multiline_string():
    """Test string expression evaluation."""

    content = '''
    flow main
      $var = "Test"
      $string = """This is a multiline
      string! '{$var}' "hi" """
      start UtteranceBotAction(script=$string)
    '''

    config = _init_state(content)
    state = run_to_completion(config, start_main_flow_event)
    assert is_data_in_events(
        state.outgoing_events,
        [
            {
                "type": "StartUtteranceBotAction",
                "script": "This is a multiline\n      string! 'Test' \"hi\" ",
            }
        ],
    )


def test_flow_return_values():
    """Test flow return value handling."""

    content = """
    flow a
      return "success"

    flow b
      return 100

    flow c
      $result = "failed"
      return $result

    flow main
      $result_a = await a
      $result_b = await b
      $result_c = await c
      start UtteranceBotAction(script="{$result_a} {$result_b} {$result_c}")
    """

    config = _init_state(content)
    state = run_to_completion(config, start_main_flow_event)
    assert is_data_in_events(
        state.outgoing_events,
        [
            {
                "type": "StartUtteranceBotAction",
                "script": "success 100 failed",
            },
            {
                "type": "StopUtteranceBotAction",
            },
        ],
    )


def test_break_continue_statement_a():
    """Test break and continue statements within while loop."""

    content = """
    flow main
      $count = -1
      while True
        $count = $count + 1
        start UtteranceBotAction(script="S:{$count}")
        if $count < 1
          $count = $count
        elif $count < 3
          continue
        elif $count == 3
          break
        start UtteranceBotAction(script="E:{$count}")
      start UtteranceBotAction(script="Done")
    """

    config = _init_state(content)
    state = run_to_completion(config, start_main_flow_event)
    assert is_data_in_events(
        state.outgoing_events,
        [
            {
                "type": "StartUtteranceBotAction",
                "script": "S:0",
            },
            {
                "type": "StartUtteranceBotAction",
                "script": "E:0",
            },
            {
                "type": "StartUtteranceBotAction",
                "script": "S:1",
            },
            {
                "type": "StartUtteranceBotAction",
                "script": "S:2",
            },
            {
                "type": "StartUtteranceBotAction",
                "script": "S:3",
            },
            {
                "type": "StartUtteranceBotAction",
                "script": "Done",
            },
        ],
    )


def test_break_continue_statement_b():
    """Test break and continue statements within while loop."""

    content = """
    flow main
      while True
        start UtteranceBotAction(script="A")
        while True
          break
          start UtteranceBotAction(script="E1")
        start UtteranceBotAction(script="B")
        break
        start UtteranceBotAction(script="E2")
      start UtteranceBotAction(script="C")
    """

    config = _init_state(content)
    state = run_to_completion(config, start_main_flow_event)
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


def test_when_or_core_mechanics():
    """Test when / or when statement mechanics."""

    content = """
    flow user said $transcript
      match UtteranceUserAction.Finished(final_transcript=$transcript)

    flow main
      while True:
        when UtteranceUserActionFinished(final_transcript="A"):
          start UtteranceBotAction(script="A")
        or when UtteranceUserAction().Finished(final_transcript="B"):
          start UtteranceBotAction(script="B")
        or when user said "C":
          start UtteranceBotAction(script="C")
          break
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
            "final_transcript": "A",
        },
    )
    assert is_data_in_events(
        state.outgoing_events,
        [
            {
                "type": "StartUtteranceBotAction",
                "script": "A",
            },
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
            },
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
                "type": "StartUtteranceBotAction",
                "script": "C",
            },
            {
                "type": "StopUtteranceBotAction",
            },
            {
                "type": "StopUtteranceBotAction",
            },
            {
                "type": "StopUtteranceBotAction",
            },
        ],
    )


def test_when_or_bot_action_mechanics():
    """Test when / or when statement mechanics with actions."""

    content = """
    flow main
      while True
        when UtteranceBotAction(script="Happens immediately")
          start UtteranceBotAction(script="A")
        or when UtteranceUserActionFinished(final_transcript="B")
          start UtteranceBotAction(script="B")
          break
    """

    config = _init_state(content)
    state = run_to_completion(config, start_main_flow_event)
    assert is_data_in_events(
        state.outgoing_events,
        [
            {
                "type": "StartUtteranceBotAction",
                "script": "Happens immediately",
            },
        ],
    )
    state = run_to_completion(
        state,
        {
            "type": "UtteranceBotActionFinished",
            "final_script": "Happens immediately",
            "action_uid": state.outgoing_events[0]["action_uid"],
        },
    )
    assert is_data_in_events(
        state.outgoing_events,
        [
            {
                "type": "StartUtteranceBotAction",
                "script": "A",
            },
            {
                "type": "StartUtteranceBotAction",
                "script": "Happens immediately",
            },
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
                "type": "StopUtteranceBotAction",
            },
            {
                "type": "StartUtteranceBotAction",
                "script": "B",
            },
            {
                "type": "StopUtteranceBotAction",
            },
            {
                "type": "StopUtteranceBotAction",
            },
        ],
    )


def test_when_or_group_mechanics():
    """Test when / or when statement mechanics with or-grouping."""

    content = """
    flow user said $transcript
      match UtteranceUserAction.Finished(final_transcript=$transcript)

    flow main
      while True
        when UtteranceUserActionFinished(final_transcript="A")
          start UtteranceBotAction(script="A")
        or when (user said "B" and user said "C")
          start UtteranceBotAction(script="BC")
        or when (user said "D" or user said "E")
          start UtteranceBotAction(script="DE")
          break
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
            "final_transcript": "A",
        },
    )
    assert is_data_in_events(
        state.outgoing_events,
        [
            {
                "type": "StartUtteranceBotAction",
                "script": "A",
            },
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
        [
            {
                "type": "StartUtteranceBotAction",
                "script": "BC",
            },
        ],
    )
    state = run_to_completion(
        state,
        {
            "type": "UtteranceUserActionFinished",
            "final_transcript": "E",
        },
    )
    assert is_data_in_events(
        state.outgoing_events,
        [
            {
                "type": "StartUtteranceBotAction",
                "script": "DE",
            },
            {
                "type": "StopUtteranceBotAction",
            },
            {
                "type": "StopUtteranceBotAction",
            },
            {
                "type": "StopUtteranceBotAction",
            },
        ],
    )


def test_when_or_competing_events_mechanics():
    """Test when / or when statement mechanics with events."""

    content = """
    flow user said something
      match UtteranceUserAction.Finished()

    flow user said $transcript
      match UtteranceUserAction.Finished(final_transcript=$transcript)

    flow main
      while True
        when user said "hello"
          start UtteranceBotAction(script="A")
        or when user said something
          start UtteranceBotAction(script="B")
        or when user said "hi"
          start UtteranceBotAction(script="C")
          break
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
            "final_transcript": "hello",
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
            "final_transcript": "something 123",
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
                "script": "C",
            },
            {
                "type": "StopUtteranceBotAction",
            },
            {
                "type": "StopUtteranceBotAction",
            },
            {
                "type": "StopUtteranceBotAction",
            },
        ],
    )


def test_when_or_with_references():
    """Test when / or when statement mechanics with references."""

    content = """
    flow user said something
      match UtteranceUserAction.Finished() as $event

    flow main
      when user said something as $ref_1
        start UtteranceBotAction(script=$ref_1.context.event.arguments.final_transcript)
      match WaitEvent()
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
            "final_transcript": "hello",
        },
    )
    assert is_data_in_events(
        state.outgoing_events,
        [
            {
                "type": "StartUtteranceBotAction",
                "script": "hello",
            },
        ],
    )


def test_inside_when_failure_handling():
    """Test when / or when statement failure handling mechanics."""

    content = """
    flow a
      start UtteranceBotAction(script="Start")
      when UtteranceUserAction.Finished(final_transcript="A")
        await b
      or when UtteranceUserAction.Finished(final_transcript="B")
        await b

    flow b
      abort

    flow main
      activate a
      match WaitEvent()
    """

    config = _init_state(content)
    state = run_to_completion(config, start_main_flow_event)
    assert is_data_in_events(
        state.outgoing_events,
        [
            {
                "type": "StartUtteranceBotAction",
                "script": "Start",
            },
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
                "script": "Start",
            },
        ],
    )


def test_abort_flow():
    """Test abort keyword mechanics."""

    content = """
    flow a
      match UtteranceUserAction.Finished(final_transcript="go")
      abort
      start UtteranceBotAction(script="Error")

    flow main
      start a
      match FlowFailed(flow_id="a")
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
            "type": "UtteranceUserActionFinished",
            "final_transcript": "go",
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
        ],
    )


def test_global_statement():
    """Test global variables."""

    content = """
    flow a
      $var = "Check1"
      start UtteranceBotAction(script=$var)

    flow b
      global $var
      $var = "Check2"

    flow main
      global $var
      $var = "Start"
      start a
      start UtteranceBotAction(script=$var)
      start b
      start UtteranceBotAction(script=$var)
      match WaitEvent()
    """

    config = _init_state(content)
    state = run_to_completion(config, start_main_flow_event)
    assert is_data_in_events(
        state.outgoing_events,
        [
            {
                "type": "StartUtteranceBotAction",
                "script": "Check1",
            },
            {
                "type": "StopUtteranceBotAction",
            },
            {
                "type": "StartUtteranceBotAction",
                "script": "Start",
            },
            {
                "type": "StartUtteranceBotAction",
                "script": "Check2",
            },
        ],
    )


def test_out_flow_variables():
    """Test the out variables flow mechanics."""

    content = """
    flow a -> $result_1 = 1, $result_2 = 2
      match WaitEvent()

    flow main
      start a as $ref
      start UtteranceBotAction(script="{$ref.result_1} {$ref.result_2}")
      match WaitEvent()
    """

    config = _init_state(content)
    state = run_to_completion(config, start_main_flow_event)
    assert is_data_in_events(
        state.outgoing_events,
        [
            {
                "type": "StartUtteranceBotAction",
                "script": "1 2",
            }
        ],
    )


def test_expression_evaluation():
    """Test the different ways of expression evaluations."""

    content = """
    flow main
      $dict = {"val": 2 + 3}
      start bot say number ($dict["val"])
      ($dict.update({"val":10}))
      bot say number ($dict["val"] + 1)

    flow bot say number $number
      await UtteranceBotAction(script="{$number}")
    """

    config = _init_state(content)
    state = run_to_completion(config, start_main_flow_event)
    assert is_data_in_events(
        state.outgoing_events,
        [
            {
                "type": "StartUtteranceBotAction",
                "script": "5",
            },
            {
                "type": "StartUtteranceBotAction",
                "script": "11",
            },
        ],
    )


if __name__ == "__main__":
    test_multiline_string()
