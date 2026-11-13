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

from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.utils import new_event_dict, new_uuid

DANGEROUS_BOT_OUTPUT = "DROP TABLE users"

colang_content = """
import guardrails
import nemoguardrails.library.human_approval

flow output rails $output_text
  await human approval on bot output $output_text

flow main
  bot say "{$output_text}"
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
        colang_content.replace("{$output_text}", DANGEROUS_BOT_OUTPUT),
        yaml_content,
    )
    app = LLMRails(config)
    app.runtime.disable_async_execution = True
    return app


def _finish_bot_action(app: LLMRails, state, start_event: dict):
    return app.process_events(
        [
            new_event_dict(
                "UtteranceBotActionFinished",
                action_uid=start_event["action_uid"],
                is_success=True,
                final_script=start_event["script"],
            )
        ],
        state,
    )


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


def test_human_approval_v2_prompts_then_approves():
    app = _setup_rails()
    state = None

    out, state = app.process_events([], state)
    assert len(out) == 1
    assert out[0]["type"] == "StartUtteranceBotAction"
    assert "requires approval" in out[0]["script"]

    out, state = _finish_bot_action(app, state, out[0])
    out, state = app.process_events(_user_said("approve"), state)

    assert out[-1]["type"] == "StartUtteranceBotAction"
    assert out[-1]["script"] == DANGEROUS_BOT_OUTPUT


def test_human_approval_v2_rejects_non_approval_response():
    app = _setup_rails()
    state = None

    out, state = app.process_events([], state)
    out, state = _finish_bot_action(app, state, out[0])
    out, state = app.process_events(_user_said("no"), state)

    assert out[-1]["type"] == "StartUtteranceBotAction"
    assert "rejected" in out[-1]["script"].lower()
