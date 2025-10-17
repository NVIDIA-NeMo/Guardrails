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


@pytest.fixture
def config_1():
    return RailsConfig.from_content(
        colang_content="""
            flow user said $text
              match UtteranceUserActionFinished(final_transcript=$text)

            flow bot say $text
              await UtteranceBotAction(script=$text)

            flow wait
              await SomeRandomAction()

            # ---

            flow greeting_1
              user said "hi"
              $value = "1"

              if $value == "1"
                bot say "Hello 1"
              else
                bot say "Hello 2"

            flow greeting_2
              user said "hi_2"
              $value = "2"

              if $value == "1"
                bot say "Hello 21"
              else
                bot say "Hello 22"

            flow main
              activate greeting_1
              activate greeting_2
              wait
            """,
        yaml_content="""
            colang_version: "2.x"
            """,
    )


def test_2_x(config_1):
    chat = TestChat(
        config_1,
        llm_completions=[],
    )

    chat >> "hi"
    chat << "Hello 1"


def test_1_2(config_1):
    chat = TestChat(
        config_1,
        llm_completions=[],
    )

    chat >> "hi_2"
    chat << "Hello 22"


@pytest.fixture
def config_2():
    return RailsConfig.from_content(
        colang_content="""
            flow user said $text
              match UtteranceUserActionFinished(final_transcript=$text)

            flow bot say $text
              await UtteranceBotAction(script=$text)

            flow wait
              await SomeRandomAction()

            # ---

            flow bot greet based on value $value
              if $value == "1"
                bot say "Hello 1"
              elif $value == "2"
                bot say "Hello 2"
              elif $value == "3"
                bot say "Hello 3"
              else
                bot say "Hello!"

            flow greeting_1
              user said "hi_1"
              $value = "1"
              bot greet based on value $value

            flow greeting_2
              user said "hi_2"
              $value = "2"
              bot greet based on value $value

            flow greeting_3
              user said "hi_3"
              $value = "3"
              bot greet based on value $value

            flow greeting_4
              user said "hi_4"
              $value = "4"
              bot greet based on value $value

            flow main
              activate greeting_1
              activate greeting_2
              activate greeting_3
              activate greeting_4
              wait
            """,
        yaml_content="""
            colang_version: "2.x"
            """,
    )


def test_2_1(config_2):
    chat = TestChat(
        config_2,
        llm_completions=[],
    )

    chat >> "hi_1"
    chat << "Hello 1"


def test_2_2(config_2):
    chat = TestChat(
        config_2,
        llm_completions=[],
    )

    chat >> "hi_2"
    chat << "Hello 2"


def test_2_3(config_2):
    chat = TestChat(
        config_2,
        llm_completions=[],
    )

    chat >> "hi_3"
    chat << "Hello 3"


def test_2_4(config_2):
    chat = TestChat(
        config_2,
        llm_completions=[],
    )

    chat >> "hi_4"
    chat << "Hello!"
