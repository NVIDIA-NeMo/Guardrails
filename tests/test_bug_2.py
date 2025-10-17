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
    define user ask capabilities
      "What can you do?"
      "What can you help me with?"
      "tell me what you can do"
      "tell me about you"

    define user report bug
      "report bug"
      "report error"
      "this is an error"
      "this is a bug"

    define bot inform capabilities
      "I am an AI assistant that helps answer mathematical questions. My core mathematical skills are powered by wolfram alpha."

    define bot ask what is wrong
      "Sorry about that! Could you please tell me what is wrong?"

    define bot ask what is expected
      "Could you please let me know how I should have responded?"

    define bot thank user
      "Thank you!"

    define user report bug expectation
      "You should have told me about the api key"
      "The jailbreak should have captured it"
      "The moderation should have filtered the response"
      "The flow is not activated"

    define flow
      user ask capabilities
      bot inform capabilities

    define flow
      user ask math question
      execute wolfram alpha request
      bot respond to math question

    define user ask math question
      "What is the square root of 53?"
      "What is Pythagoras' theorem?"
      "What is the integral of sin(x) with respect to x"
      "Solve the following equation: x^2 + 2*x + 1 = 0"

    define user inform bug details
      "There was no fact checking done"
      "There was no jail break rail activated"
      "The API key did not work"
      "The system did not respond"
      "test"

    define flow log bugs
        user report bug
        bot ask what is wrong
        user inform bug details
        bot ask what is expected
        user report bug expectation
        bot thank user
        execute log_conversation
    """
)


def test_1():
    chat = TestChat(
        config,
        llm_completions=[
            "  ask math question",
            '  "330 multiplied by 40 is equal to 13200."',
            "  report bug",
            "  inform bug details",
            "  report bug expectation",
            '  "Thank you!"',
        ],
    )

    async def wolfram_alpha_request(context: dict):
        pass

    log = []

    async def log_conversation(context: dict):
        log.append(context.get("last_user_message"))

    chat.app.register_action(wolfram_alpha_request, name="wolfram alpha request")
    chat.app.register_action(log_conversation)

    chat >> "What is 330 * 40?"
    chat << "330 multiplied by 40 is equal to 13200."
    chat >> "report bug"
    chat << "Sorry about that! Could you please tell me what is wrong?"
    chat >> "api is not working"
    chat << "Could you please let me know how I should have responded?"
    chat >> "It should have responded with 202"
    chat << "Thank you!"
