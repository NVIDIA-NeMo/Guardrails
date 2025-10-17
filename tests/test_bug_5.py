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
import pytest

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
    models:
      - type: main
        engine: openai
        model: gpt-3.5-instruct-turbo
    """,
)


# TODO: reactivate this test
@pytest.mark.skip(reason="To investigate what changed.")
def test_1():
    chat = TestChat(
        config,
        llm_completions=[
            "  express greeting",
        ],
    )

    rails = chat.app

    messages = [
        {"role": "user", "content": "Hi 1!"},
        {"role": "assistant", "content": "Hi 2!"},
        {"role": "user", "content": "Hi 3!"},
        {"role": "assistant", "content": "Hi 4!"},
        {"role": "user", "content": "Hi!"},
    ]
    new_message = rails.generate(messages=messages)

    assert new_message == {"role": "assistant", "content": "Hey!"}

    info = rails.explain()
    assert len(info.llm_calls) == 1
    assert "Hi 1!" in info.llm_calls[0].prompt
    assert "Hi 3!" in info.llm_calls[0].prompt
