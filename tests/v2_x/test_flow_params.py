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


def test_1():
    config = RailsConfig.from_content(
        colang_content="""
        flow user said $text
          match UtteranceUserActionFinished(final_transcript=$text)

        flow bot say $text
          await UtteranceBotAction(script=$text)

        flow user express greeting
          user said "hi"

        flow bot express greeting
          bot say "Hello world!"

        flow main
          user express greeting
          bot express greeting
        """,
        yaml_content="""
        colang_version: "2.x"
        """,
    )

    chat = TestChat(
        config,
        llm_completions=[],
    )

    chat >> "hi"
    chat << "Hello world!"


@pytest.fixture
def config_2():
    return RailsConfig.from_content(
        colang_content="""
            # Wrappers

            flow user said something
              match UtteranceUserActionFinished()

            flow user said $text
              match UtteranceUserActionFinished(final_transcript=$text)

            flow bot say $text
              await UtteranceBotAction(script=$text)

            flow wait
              await SomeRandomAction()

            # ---

            flow user express greeting
              user said "hi"

            flow bot express greeting
              bot say "Hello world!"

            flow bot comment unknown next step
              bot say "I'm not sure what to do next."

            flow greeting
              user express greeting
              bot express greeting

            flow fallback
              user said something
              bot comment unknown next step

            flow main
              start greeting
              start fallback
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

    chat >> "hi"
    chat << "Hello world!"


def test_2_2(config_2):
    chat = TestChat(
        config_2,
        llm_completions=[],
    )

    chat >> "I need help"
    chat << "I'm not sure what to do next."


def test_3():
    config = RailsConfig.from_content(
        colang_content="""
        flow user said $text
          match UtteranceUserActionFinished(final_transcript=$text)

        flow bot say $text
          await UtteranceBotAction(script=$text)

        flow user express greeting
          user said "hi"

        flow bot express greeting
          bot say "Hello world!"

        flow main
          $a = [1,2,3]
          user express greeting

          bot say $text = str(len($a))
          bot say (str(len($a)))
        """,
        yaml_content="""
        colang_version: "2.x"
        """,
    )

    chat = TestChat(
        config,
        llm_completions=[],
    )

    chat >> "hi"
    chat << "3\n3"
