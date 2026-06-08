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

from nemoguardrails import RailsConfig
from nemoguardrails.actions.rail_outcome import RailOutcome, TransformTarget
from nemoguardrails.library.prompt_security.actions import _protect_text_outcome
from tests.utils import TestChat


def mock_protect_text(return_value):
    def mock_request(*args, **kwargs):
        if isinstance(return_value, RailOutcome):
            return return_value
        target = TransformTarget.BOT_MESSAGE if kwargs.get("bot_response") else TransformTarget.USER_MESSAGE
        if return_value.get("is_blocked"):
            return RailOutcome.block(**return_value)
        if return_value.get("is_modified"):
            return RailOutcome.transform([(target, return_value.get("modified_text") or "")], **return_value)
        return RailOutcome.allow(**return_value)

    return mock_request


@pytest.mark.unit
def test_prompt_security_protection_disabled():
    config = RailsConfig.from_content(
        colang_content="""
            define user express greeting
              "hi"

            define flow
              user express greeting
              bot express greeting

            define bot inform answer unknown
              "I can't answer that."
        """,
    )

    chat = TestChat(
        config,
        llm_completions=[
            "  express greeting",
            '  "Hi! My name is John as well."',
        ],
    )

    chat.app.register_action(mock_protect_text({"is_blocked": True, "is_modified": False}), "protect_text")
    chat >> "Hi! I am Mr. John! And my email is test@gmail.com"
    chat << "Hi! My name is John as well."


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        (
            {"is_blocked": False, "is_modified": False, "modified_text": None},
            RailOutcome.allow(is_blocked=False, is_modified=False, modified_text=None),
        ),
        (
            {"is_blocked": True, "is_modified": False, "modified_text": None},
            RailOutcome.block(is_blocked=True, is_modified=False, modified_text=None),
        ),
        (
            {"is_blocked": False, "is_modified": True, "modified_text": "masked"},
            RailOutcome.transform(
                [(TransformTarget.USER_MESSAGE, "masked")],
                is_blocked=False,
                is_modified=True,
                modified_text="masked",
            ),
        ),
    ],
)
def test_protect_text_outcome(result, expected):
    assert _protect_text_outcome(result, TransformTarget.USER_MESSAGE) == expected


@pytest.mark.unit
def test_prompt_security_protection_input():
    config = RailsConfig.from_content(
        yaml_content="""
            models: []
            rails:
              input:
                flows:
                  - protect prompt
        """,
        colang_content="""
            define user express greeting
              "hi"

            define flow
              user express greeting
              bot express greeting

            define bot inform answer unknown
              "I can't answer that."
        """,
    )

    chat = TestChat(
        config,
        llm_completions=[
            "  express greeting",
            '  "Hi! My name is John as well."',
        ],
    )

    chat.app.register_action(mock_protect_text({"is_blocked": True, "is_modified": False}), "protect_text")
    chat >> "Hi! I am Mr. John! And my email is test@gmail.com"
    chat << "I can't answer that."


@pytest.mark.unit
def test_prompt_security_protection_output():
    config = RailsConfig.from_content(
        yaml_content="""
            models: []
            rails:
              output:
                flows:
                  - protect response
        """,
        colang_content="""
            define user express greeting
              "hi"

            define flow
              user express greeting
              bot express greeting

            define bot inform answer unknown
              "I can't answer that."
        """,
    )

    chat = TestChat(
        config,
        llm_completions=[
            "  express greeting",
            '  "Hi! My name is John as well."',
        ],
    )

    chat.app.register_action(mock_protect_text({"is_blocked": True, "is_modified": False}), "protect_text")
    chat >> "Hi!"
    chat << "I can't answer that."
