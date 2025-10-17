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

import logging

from rich.logging import RichHandler

from nemoguardrails import RailsConfig
from tests.utils import TestChat

FORMAT = "%(message)s"
logging.basicConfig(
    level=logging.DEBUG,
    format=FORMAT,
    datefmt="[%X,%f]",
    handlers=[RichHandler(markup=True)],
)


def test_1():
    config = RailsConfig.from_content(
        colang_content="""
        flow user express greeting
          match UtteranceUserActionFinished(final_transcript="hi")

        flow bot express greeting
          await UtteranceBotAction(script="Hello world!")

        flow main
          user express greeting
          await FetchNameAction()
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

    async def fetch_name():
        return "John"

    chat.app.register_action(fetch_name, "FetchNameAction")

    chat >> "hi"
    chat << "Hello world!"


def test_2():
    config = RailsConfig.from_content(
        colang_content="""
        flow user express greeting
          match UtteranceUserActionFinished(final_transcript="hi")

        flow bot say $text
          await UtteranceBotAction(script=$text)

        flow main
          user express greeting
          $name = FetchNameAction
          bot say $name
        """,
        yaml_content="""
        colang_version: "2.x"
        """,
    )

    chat = TestChat(
        config,
        llm_completions=[],
    )

    async def fetch_name():
        return "John"

    chat.app.register_action(fetch_name, "FetchNameAction")

    chat >> "hi"
    chat << "John"


def test_3():
    config = RailsConfig.from_content(
        colang_content="""
        flow bot say $text
          await UtteranceBotAction(script=$text)

        flow main
          match UtteranceUserActionFinished(final_transcript="hi")
          $information = await FetchDictionaryAction()
          $response_to_user = ..."Summarize the result from the AddItemAction call to the user: {$information}"
          bot say $response_to_user
        """,
        yaml_content="""
        colang_version: "2.x"
        """,
    )

    chat = TestChat(
        config,
        llm_completions=['"I couldn\'t find any items matching your request!"'],
    )

    async def fetch_dictionary():
        return {
            "isSuccess": False,
            "response": "I couldn't find any items matching your request. Would you like to try again, or browse the available options?",
        }

    chat.app.register_action(fetch_dictionary, "FetchDictionaryAction")

    chat >> "hi"
    chat << "I couldn't find any items matching your request!"


if __name__ == "__main__":
    test_3()
