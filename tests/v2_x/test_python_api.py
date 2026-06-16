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

from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.exceptions import InvalidStateError
from nemoguardrails.utils import new_event_dict
from tests.utils import TestChat

config = RailsConfig.from_content(
    """
    import core

    flow math question
      $text = user said "what is 2+2"

      bot say "Let me think."
      $result = await WolframAlphaAction(query=$text)
      bot say $result

    flow greeting
      user said "hi"
      bot say "Hello!"
      user said "hi"
      bot say "Hello again!"

    flow main
      activate greeting
      activate math question
    """,
    """
    colang_version: "2.x"
    """,
)


def test_1():
    rails = LLMRails(config=config)
    messages = [{"role": "user", "content": "hi"}]

    response = rails.generate(messages=messages)

    # We should only get the input rail here.
    assert response == {"role": "assistant", "content": "Hello!"}


def test_exception_1():
    rails = LLMRails(config=config)
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "Hello!"},
        {"role": "user", "content": "How are you?"},
    ]

    with pytest.raises(ValueError) as exc_info:
        rails.generate(messages=messages)

        assert "not supported for Colang 2.0" in str(exc_info.value)


def test_state_multi_turn_via_process_events():
    """Colang 2.0 in-process multi-turn keeps a live State across calls.

    Mirrors the pattern at nemoguardrails/cli/chat.py: the live Python State
    object is held in memory between calls and handed back to process_events
    each turn. The second `user said "hi"` only matches the second branch of
    the greeting flow because the State carries the flow position forward.
    """
    chat = TestChat(config, llm_completions=[])

    # The State exists from TestChat's initial process_events([], None) bootstrap.
    assert chat.state is not None

    # First turn: "hi" → "Hello!"
    chat >> "hi"
    chat << "Hello!"

    # Second turn relies on the State surviving across the call; without it
    # the greeting flow would emit "Hello!" again instead of advancing.
    chat >> "hi"
    chat << "Hello again!"


@pytest.mark.parametrize(
    "state",
    [
        {"version": "2.x", "state": "{}"},
        {"events": []},
        {"unexpected": "value"},
    ],
)
def test_dict_state_rejected_2_x(state):
    """Caller-supplied dict state for Colang 2.0 must be refused at the API boundary.

    The serialized State contains trusted control-plane fields (flow_configs,
    rails_config) and cannot come from an untrusted caller.
    """
    rails = LLMRails(config=config)
    messages = [{"role": "user", "content": "hi"}]

    with pytest.raises(InvalidStateError) as exc_info:
        rails.generate(
            messages=messages,
            state=state,
        )

    assert "process_events_async" in str(exc_info.value)


def test_state_not_returned_for_2_x():
    """generate_async no longer returns a continuation state for Colang 2.0.

    Stateful 2.x execution moved to process_events_async (see test above).
    """
    rails = LLMRails(config=config)
    messages = [{"role": "user", "content": "hi"}]

    res = rails.generate(messages=messages, state={})

    assert res.state is None


def test_stream_async_dict_state_rejected_2_x():
    rails = LLMRails(config=config)

    with pytest.raises(InvalidStateError):
        rails.stream_async(
            messages=[{"role": "user", "content": "hi"}],
            state={"events": []},
        )


def test_actions_continue_with_live_state_via_process_events():
    """Tool/action pause and resume still works with trusted in-process State."""
    rails = LLMRails(config=config)

    output_events, state = rails.process_events(
        [{"type": "UtteranceUserActionFinished", "final_transcript": "what is 2+2"}],
        state={},
        blocking=True,
    )
    assert output_events[0]["type"] == "StartUtteranceBotAction"
    assert output_events[0]["script"] == "Let me think."

    output_events, state = rails.process_events(
        [
            new_event_dict(
                "UtteranceBotActionFinished",
                action_uid=output_events[0]["action_uid"],
                is_success=True,
                final_script="Let me think.",
            )
        ],
        state=state,
        blocking=True,
    )
    assert output_events[0]["type"] == "StartWolframAlphaAction"
    assert output_events[0]["query"] == "what is 2+2"

    output_events, state = rails.process_events(
        [
            {
                "type": "WolframAlphaActionFinished",
                "action_uid": output_events[0]["action_uid"],
                "action_name": "WolframAlphaAction",
                "status": "success",
                "is_success": True,
                "return_value": "The result is 4.",
                "events": [],
            }
        ],
        state=state,
        blocking=True,
    )
    assert output_events[0]["type"] == "StartUtteranceBotAction"
    assert output_events[0]["script"] == "The result is 4."


@pytest.fixture
def config_2():
    return RailsConfig.from_content(
        colang_content="""
        import core

        flow main
            user said "hi"
            $datetime = await GetCurrentDateTimeAction()
            user said "there"
            bot say "hello"


        """,
        yaml_content="""
        colang_version: "2.x"
        """,
    )


def test_pattern_matching_with_python_actions(config_2):
    chat = TestChat(
        config_2,
        llm_completions=[],
    )

    chat >> "hi"
    chat >> "there"
    chat << "hello"
