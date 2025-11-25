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

colang_content = '''
    import core
    import llm

    flow main
        activate llm continuation
        activate greeting
        activate other reactions

    flow greeting
        user expressed greeting
        bot say "Hello world!"

    flow other reactions
        user expressed to be bored
        bot say "No problem!"

    flow user expressed greeting
        """"User expressed greeting in any way or form."""
        user said "hi"

    flow user expressed to be bored
        """"User expressed to be bored."""
        user said "This is boring"
    '''

yaml_content = """
colang_version: "2.x"
models:
  - type: main
    engine: openai
    model: gpt-3.5-turbo-instruct

    """


def test_1():
    config = RailsConfig.from_content(colang_content, yaml_content)

    chat = TestChat(
        config,
        llm_completions=["user expressed greeting"],
    )

    chat >> "hi"
    chat << "Hello world!"

    chat >> "hello"
    chat << "Hello world!"


def test_2():
    config = RailsConfig.from_content(colang_content, yaml_content)

    chat = TestChat(
        config,
        llm_completions=["user expressed to be bored"],
    )

    chat >> "You are boring me!"
    chat << "No problem!"


if __name__ == "__main__":
    test_2()
