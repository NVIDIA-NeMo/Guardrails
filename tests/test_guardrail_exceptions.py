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

from nemoguardrails import RailsConfig
from tests.utils import TestChat

config = RailsConfig.from_content(
    """
    define user express greeting
        "hello"
        "hi"
        "how are you"

    define bot express greeting
        "Hey!"

    define flow greeting
        user express greeting
        bot express greeting
""",
    yaml_content="""
    models: []
    rails:
        input:
            flows:
                - self check input
        output:
            flows:
                - self check output
    prompts:
        # We need to have some prompt placeholders, otherwise the validation fails.
        - task: self_check_input
          content: ...
        - task: self_check_output
          content: ...

    enable_rails_exceptions: True
    """,
)


def test_self_check_input_exception():
    chat = TestChat(
        config,
        llm_completions=[
            "Yes",
        ],
    )

    rails = chat.app
    messages = [
        {"role": "user", "content": "Hi 1!"},
    ]
    new_message = rails.generate(messages=messages)

    assert new_message["role"] == "exception"
    assert new_message["content"]["type"] == "InputRailException"


def test_self_check_output_exception():
    chat = TestChat(
        config,
        llm_completions=[
            "No",
            "  ask general question",
            "  respond",
            '  "Something that should be blocked"',
            "Yes",
        ],
    )

    rails = chat.app
    messages = [
        {"role": "user", "content": "Something that generates bad output"},
    ]
    new_message = rails.generate(messages=messages)

    assert new_message["role"] == "exception"
    assert new_message["content"]["type"] == "OutputRailException"
