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
from nemoguardrails.actions.actions import ActionResult, action
from tests.utils import FakeLLM, TestChat

COLANG_CONFIG = """
define user express greeting
  "hi"
define bot refuse to respond
  "I'm sorry, I can't respond to that."
"""

YAML_CONFIG = """
models:
  - type: main
    engine: openai
    model: gpt-3.5-turbo-instruct
rails:
  output:
    flows:
      - patronus lynx check output hallucination
prompts:
  - task: patronus_lynx_check_output_hallucination
    content: |
      Given the following QUESTION, DOCUMENT and ANSWER you must analyze the provided answer and determine whether it is faithful to the contents of the DOCUMENT.

      The ANSWER must not offer new information beyond the context provided in the DOCUMENT.

      The ANSWER also must not contradict information provided in the DOCUMENT.

      Output your final score by strictly following this format: "PASS" if the answer is faithful to the DOCUMENT and "FAIL" if the answer is not faithful to the DOCUMENT.

      Show your reasoning.

      --
      QUESTION (THIS DOES NOT COUNT AS BACKGROUND INFORMATION):
      {{ user_input }}

      --
      DOCUMENT:
      {{ provided_context }}

      --
      ANSWER:
      {{ bot_response }}

      --

      Your output should be in JSON FORMAT with the keys "REASONING" and "SCORE".

      Ensure that the JSON is valid and properly formatted.

      {"REASONING": ["<your reasoning as bullet points>"], "SCORE": "<final score>"}
"""


@action()
def retrieve_relevant_chunks():
    context_updates = {"relevant_chunks": "Mock retrieved context."}

    return ActionResult(
        return_value=context_updates["relevant_chunks"],
        context_updates=context_updates,
    )


@pytest.mark.asyncio
def test_patronus_lynx_returns_no_hallucination():
    """
    Test that that chat flow completes successfully when
    Patronus Lynx returns "PASS" for the hallucination check
    """
    config = RailsConfig.from_content(colang_content=COLANG_CONFIG, yaml_content=YAML_CONFIG)
    chat = TestChat(
        config,
        llm_completions=[
            "Mock generated user intent",  # mock response for the generate_user_intent action
            "Mock generated next step",  # mock response for the generate_next_step action
            "  Hi there! How are you doing?",  # mock response for the generate_bot_message action
        ],
    )

    chat.app.register_action(retrieve_relevant_chunks, "retrieve_relevant_chunks")

    patronus_lynx_llm = FakeLLM(
        responses=[
            '{"REASONING": ["There is no hallucination."], "SCORE": "PASS"}',
        ]
    )
    chat.app.register_action_param("patronus_lynx_llm", patronus_lynx_llm)

    chat >> "Hi"
    chat << "Hi there! How are you doing?"


@pytest.mark.asyncio
def test_patronus_lynx_returns_hallucination():
    """
    Test that that bot output is successfully guarded against when
    Patronus Lynx returns "FAIL" for the hallucination check
    """
    config = RailsConfig.from_content(colang_content=COLANG_CONFIG, yaml_content=YAML_CONFIG)
    chat = TestChat(
        config,
        llm_completions=[
            "Mock generated user intent",  # mock response for the generate_user_intent action
            "Mock generated next step",  # mock response for the generate_next_step action
            "  Hi there! How are you doing?",  # mock response for the generate_bot_message action
        ],
    )

    chat.app.register_action(retrieve_relevant_chunks, "retrieve_relevant_chunks")

    patronus_lynx_llm = FakeLLM(
        responses=[
            '{"REASONING": ["There is a hallucination."], "SCORE": "FAIL"}',
        ]
    )
    chat.app.register_action_param("patronus_lynx_llm", patronus_lynx_llm)

    chat >> "Hi"
    chat << "I don't know the answer to that."


@pytest.mark.asyncio
def test_patronus_lynx_parses_score_when_no_double_quote():
    """
    Test that that chat flow completes successfully when
    Patronus Lynx returns "PASS" for the hallucination check
    """
    config = RailsConfig.from_content(colang_content=COLANG_CONFIG, yaml_content=YAML_CONFIG)
    chat = TestChat(
        config,
        llm_completions=[
            "Mock generated user intent",  # mock response for the generate_user_intent action
            "Mock generated next step",  # mock response for the generate_next_step action
            "  Hi there! How are you doing?",  # mock response for the generate_bot_message action
        ],
    )

    chat.app.register_action(retrieve_relevant_chunks, "retrieve_relevant_chunks")

    patronus_lynx_llm = FakeLLM(
        responses=[
            '{"REASONING": ["There is no hallucination."], "SCORE": PASS}',
        ]
    )
    chat.app.register_action_param("patronus_lynx_llm", patronus_lynx_llm)

    chat >> "Hi"
    chat << "Hi there! How are you doing?"


@pytest.mark.asyncio
def test_patronus_lynx_returns_no_hallucination_when_no_retrieved_context():
    """
    Test that that Patronus Lynx does not block the bot output
    when no relevant context is given
    """
    config = RailsConfig.from_content(colang_content=COLANG_CONFIG, yaml_content=YAML_CONFIG)
    chat = TestChat(
        config,
        llm_completions=[
            "Mock generated user intent",  # mock response for the generate_user_intent action
            "Mock generated next step",  # mock response for the generate_next_step action
            "  Hi there! How are you doing?",  # mock response for the generate_bot_message action
        ],
    )

    patronus_lynx_llm = FakeLLM(
        responses=[
            '{"REASONING": ["There is a hallucination."], "SCORE": "FAIL"}',
        ]
    )
    chat.app.register_action_param("patronus_lynx_llm", patronus_lynx_llm)

    chat >> "Hi"
    chat << "Hi there! How are you doing?"


@pytest.mark.asyncio
def test_patronus_lynx_returns_hallucination_when_no_score_in_llm_output():
    """
    Test that that Patronus Lynx defaults to blocking the bot output
    when no score is returned in its response.
    """
    config = RailsConfig.from_content(colang_content=COLANG_CONFIG, yaml_content=YAML_CONFIG)
    chat = TestChat(
        config,
        llm_completions=[
            "Mock generated user intent",  # mock response for the generate_user_intent action
            "Mock generated next step",  # mock response for the generate_next_step action
            "  Hi there! How are you doing?",  # mock response for the generate_bot_message action
        ],
    )

    chat.app.register_action(retrieve_relevant_chunks, "retrieve_relevant_chunks")

    patronus_lynx_llm = FakeLLM(
        responses=[
            '{"REASONING": ["Mock reasoning."]}',
        ]
    )
    chat.app.register_action_param("patronus_lynx_llm", patronus_lynx_llm)

    chat >> "Hi"
    chat << "I don't know the answer to that."


@pytest.mark.asyncio
def test_patronus_lynx_returns_no_hallucination_when_no_reasoning_in_llm_output():
    """
    Test that that Patronus Lynx's hallucination check does not
    depend on the reasoning provided in its response.
    """
    config = RailsConfig.from_content(colang_content=COLANG_CONFIG, yaml_content=YAML_CONFIG)
    chat = TestChat(
        config,
        llm_completions=[
            "Mock generated user intent",  # mock response for the generate_user_intent action
            "Mock generated next step",  # mock response for the generate_next_step action
            "  Hi there! How are you doing?",  # mock response for the generate_bot_message action
        ],
    )

    chat.app.register_action(retrieve_relevant_chunks, "retrieve_relevant_chunks")

    patronus_lynx_llm = FakeLLM(
        responses=[
            '{"SCORE": "PASS"}',
        ]
    )
    chat.app.register_action_param("patronus_lynx_llm", patronus_lynx_llm)

    chat >> "Hi"
    chat << "Hi there! How are you doing?"
