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

"""Multi-turn human approval: output rails must re-fire after a rejection.

Rejection completes without abort so $output_rails_in_progress clears (see #2067).
"""

from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.utils import new_event_dict, new_uuid

DANGEROUS_BOT_OUTPUT = "DROP TABLE users"

colang_content = """
import core
import guardrails
import nemoguardrails.library.human_approval

flow output rails $output_text
  await human approval on bot output $output_text

@active
flow handle user messages
  while True
    user said something
    bot say "{$dangerous_output}"

flow main
  activate handle user messages
  wait indefinitely
"""

yaml_content = """
colang_version: "2.x"
models: []
rails:
  config:
    human_approval:
      patterns:
        - "delete|drop"
      approval_keywords:
        - "approve"
        - "yes"
      approval_message: "Human approval required for this action."
      rejection_message: "Action rejected."
"""


def _setup_rails() -> LLMRails:
    config = RailsConfig.from_content(
        colang_content.replace("{$dangerous_output}", DANGEROUS_BOT_OUTPUT),
        yaml_content,
    )
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


def _last_bot_script(events: list) -> str:
    for event in reversed(events):
        if event["type"] == "StartUtteranceBotAction":
            return event["script"]
    raise AssertionError("No StartUtteranceBotAction in events")


def test_human_approval_prompts_again_after_reject_on_new_turn():
    """After reject, a new turn should re-run output rails."""
    app = _setup_rails()
    state = None

    # Turn 1: user message → dangerous bot output → approval prompt
    out, state = app.process_events(_user_said("hello"), state)
    state = _finish_bot_actions(app, state, out)
    assert "requires approval" in _last_bot_script(out).lower()

    out, state = app.process_events(_user_said("no"), state)
    state = _finish_bot_actions(app, state, out)
    assert "rejected" in _last_bot_script(out).lower()

    # Turn 2: output rails fire again
    out, state = app.process_events(_user_said("hello again"), state)
    state = _finish_bot_actions(app, state, out)
    assert "requires approval" in _last_bot_script(out).lower()
