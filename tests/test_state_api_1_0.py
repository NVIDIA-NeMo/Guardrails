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

    define bot express greeting again
        "Hey again!"

    define flow greeting
        user express greeting
        bot express greeting
        user express greeting
        bot express greeting again
    """,
    yaml_content="""
    models: []
    """,
)


def test_1():
    chat = TestChat(
        config,
        llm_completions=[
            "  express greeting",
            "  express greeting",
        ],
    )

    rails = chat.app

    messages = [
        {"role": "user", "content": "Hi 1!"},
    ]
    new_message = rails.generate(messages=messages)

    assert new_message == {"role": "assistant", "content": "Hey!"}

    messages = [
        {"role": "user", "content": "Hi 1!"},
        {"role": "assistant", "content": "Hey!"},
        {"role": "user", "content": "Hi 1!"},
    ]
    new_message = rails.generate(messages=messages)

    assert new_message == {"role": "assistant", "content": "Hey again!"}


def test_2():
    chat = TestChat(
        config,
        llm_completions=[
            "  express greeting",
            "  express greeting",
        ],
    )

    rails = chat.app

    messages = [
        {"role": "user", "content": "Hi 1!"},
    ]
    res = rails.generate(messages=messages, state={})

    assert res.response == [{"role": "assistant", "content": "Hey!"}]

    messages = [
        {"role": "user", "content": "Hi 1!"},
    ]
    res = rails.generate(messages=messages, state=res.state)

    assert res.response == [{"role": "assistant", "content": "Hey again!"}]
