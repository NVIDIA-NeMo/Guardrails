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

"""Regression for #2067: output rails must keep firing after an abort."""

from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.utils import new_event_dict, new_uuid

colang_content = """
import core
import guardrails

flow output rails $output_text
  if $output_text == "BLOCKME"
    bot say "Rejected."
    abort

@active
flow handle user messages
  while True
    user said something
    bot say "BLOCKME"

flow main
  activate handle user messages
  wait indefinitely
"""

yaml_content = """
colang_version: "2.x"
models: []
"""


def _setup_rails() -> LLMRails:
    config = RailsConfig.from_content(colang_content, yaml_content)
    app = LLMRails(config)
    app.runtime.disable_async_execution = True
    return app


def _finish_bot_actions(app: LLMRails, state, events: list):
    for event in events:
        if event["type"] == "StartUtteranceBotAction":
            state = app.process_events(
                [
                    new_event_dict(
                        "UtteranceBotActionFinished",
                        action_uid=event["action_uid"],
                        is_success=True,
                        final_script=event["script"],
                    )
                ],
                state,
            )[1]
    return state


def _user_said(text: str) -> list:
    uid = new_uuid()
    return [
        new_event_dict("UtteranceUserActionStarted", action_uid=uid),
        new_event_dict(
            "UtteranceUserActionFinished",
            final_transcript=text,
            action_uid=uid,
            is_success=True,
        ),
    ]


def _bot_scripts(events: list) -> list[str]:
    return [event["script"] for event in events if event["type"] == "StartUtteranceBotAction"]


def test_output_rails_fire_on_every_turn_after_abort():
    """After an output-rail abort, later turns must still run output rails."""
    app = _setup_rails()
    state = None

    for turn in ("hello", "hi again", "one more"):
        out, state = app.process_events(_user_said(turn), state)
        state = _finish_bot_actions(app, state, out)
        scripts = _bot_scripts(out)
        assert scripts == ["Rejected."], f"turn {turn!r}: expected output rail to reject BLOCKME, got {scripts!r}"


def test_output_rails_pass_through_still_clears_flag():
    """Successful output rails must clear the flag so later bot says still work."""
    colang = """
import core
import guardrails

flow output rails $output_text
  global $bot_message
  $bot_message = "ok:{$output_text}"

@active
flow handle user messages
  while True
    user said something
    bot say "passthrough"

flow main
  activate handle user messages
  wait indefinitely
"""
    config = RailsConfig.from_content(colang, yaml_content)
    app = LLMRails(config)
    app.runtime.disable_async_execution = True
    state = None

    for turn in ("alpha", "beta"):
        out, state = app.process_events(_user_said(turn), state)
        state = _finish_bot_actions(app, state, out)
        assert _bot_scripts(out) == ["ok:passthrough"], (
            f"turn {turn!r}: expected transformed output, got {_bot_scripts(out)!r}"
        )
