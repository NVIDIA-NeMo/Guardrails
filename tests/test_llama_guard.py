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

from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.actions.actions import ActionResult
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
  input:
    flows:
      - llama guard check input

  output:
    flows:
      - llama guard check output

prompts:
  - task: llama_guard_check_input
    content: |
      <s>[INST] Sample content. Only the entry needs to exist for this test. [/INST]

  - task: llama_guard_check_output
    content: |
      <s>[INST] Sample content. Only the entry needs to exist for this test. [/INST]
"""


def test_llama_guard_check_all_safe():
    """
    Test the chat flow when both llama_guard_check_input and llama_guard_check_output actions return "safe"
    """
    config = RailsConfig.from_content(
        colang_content=COLANG_CONFIG, yaml_content=YAML_CONFIG
    )
    chat = TestChat(
        config,
        llm_completions=[
            "Mock generated user intent",  # mock response for the generate_user_intent action
            "Mock generated next step",  # mock response for the generate_next_step action
            "  Hi there! How are you doing?",  # mock response for the generate_bot_message action
        ],
    )

    llama_guard_llm = FakeLLM(
        responses=[
            "safe",  # llama_guard_check_input
            "safe",  # llama_guard_check_output
        ]
    )
    chat.app.register_action_param("llama_guard_llm", llama_guard_llm)

    chat >> "Hi"
    chat << "Hi there! How are you doing?"


def test_llama_guard_check_input_unsafe():
    """
    Test the chat flow when the llama_guard_check_input action returns "unsafe"
    """
    config = RailsConfig.from_content(
        colang_content=COLANG_CONFIG, yaml_content=YAML_CONFIG
    )
    chat = TestChat(
        config,
        llm_completions=[
            # Since input is unsafe, the main llm doesn't need to perform any of
            # generate_user_intent, generate_next_step, or generate_bot_message
            # Dev note: iff the input was safe, this empty llm_completions list would result in a test failure.
        ],
    )

    llama_guard_llm = FakeLLM(
        responses=[
            "unsafe",  # llama_guard_check_input
        ]
    )
    chat.app.register_action_param("llama_guard_llm", llama_guard_llm)

    chat >> "Unsafe input"
    chat << "I'm sorry, I can't respond to that."


def test_llama_guard_check_input_error():
    """
    Test the chat flow when the llama_guard_check_input action raises an error
    """
    config = RailsConfig.from_content(
        colang_content=COLANG_CONFIG, yaml_content=YAML_CONFIG
    )
    chat = TestChat(
        config,
        llm_completions=[
            # Since input is unsafe, the main llm doesn't need to perform any of
            # generate_user_intent, generate_next_step, or generate_bot_message
            # Dev note: iff the input was safe, this empty llm_completions list would result in a test failure.
        ],
    )

    llama_guard_llm = FakeLLM(
        responses=[
            "error",  # llama_guard_check_input
        ]
    )
    chat.app.register_action_param("llama_guard_llm", llama_guard_llm)

    chat >> "Unsafe input"
    chat << "I'm sorry, I can't respond to that."


def test_llama_guard_check_output_unsafe():
    """
    Test the chat flow when the llama_guard_check_input action raises an error
    """
    config = RailsConfig.from_content(
        colang_content=COLANG_CONFIG, yaml_content=YAML_CONFIG
    )
    chat = TestChat(
        config,
        llm_completions=[
            "Mock generated user intent",  # mock response for the generate_user_intent action
            "Mock generated next step",  # mock response for the generate_next_step action
            "  Hi there! How are you doing?",  # mock response for the generate_bot_message action
        ],
    )

    llama_guard_llm = FakeLLM(
        responses=[
            "safe",  # llama_guard_check_input
            "unsafe",  # llama_guard_check_output
        ]
    )
    chat.app.register_action_param("llama_guard_llm", llama_guard_llm)

    chat >> "Unsafe input"
    chat << "I'm sorry, I can't respond to that."


def test_llama_guard_check_output_error():
    """
    Test the chat flow when the llama_guard_check_input action raises an error
    """
    config = RailsConfig.from_content(
        colang_content=COLANG_CONFIG, yaml_content=YAML_CONFIG
    )
    chat = TestChat(
        config,
        llm_completions=[
            "Mock generated user intent",  # mock response for the generate_user_intent action
            "Mock generated next step",  # mock response for the generate_next_step action
            "  Hi there! How are you doing?",  # mock response for the generate_bot_message action
        ],
    )

    llama_guard_llm = FakeLLM(
        responses=[
            "safe",  # llama_guard_check_input
            "error",  # llama_guard_check_output
        ]
    )
    chat.app.register_action_param("llama_guard_llm", llama_guard_llm)

    chat >> "Unsafe input"
    chat << "I'm sorry, I can't respond to that."
