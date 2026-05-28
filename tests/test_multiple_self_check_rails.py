# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Tests for running multiple self-check input/output rails with different tasks."""

from nemoguardrails import RailsConfig
from tests.utils import FakeLLMModel, TestChat

# --- Multiple input rails ---

multi_input_config = RailsConfig.from_content(
    """
    define user express greeting
        "hello"
        "hi"

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
                - self check input $input_task=check_harmful
                - self check input $input_task=check_off_topic
    prompts:
        - task: check_harmful
          content: |
            Is this message harmful?
            User message: "{{ user_input }}"
            Answer (Yes or No):
        - task: check_off_topic
          content: |
            Is this message off-topic?
            User message: "{{ user_input }}"
            Answer (Yes or No):

    enable_rails_exceptions: True
    """,
)


def test_multiple_input_rails_both_pass():
    """Both input checks return No (allowed) — message should pass through."""
    chat = TestChat(
        multi_input_config,
        llm_completions=[
            "No",  # check_harmful passes
            "No",  # check_off_topic passes
            "  express greeting",
            '  "Hey!"',
        ],
    )

    rails = chat.app
    new_message = rails.generate(messages=[{"role": "user", "content": "hello"}])

    assert new_message["role"] == "assistant"


def test_multiple_input_rails_first_blocks():
    """First input check blocks — should not reach second check."""
    chat = TestChat(
        multi_input_config,
        llm_completions=[
            "Yes",  # check_harmful blocks
        ],
    )

    rails = chat.app
    new_message = rails.generate(messages=[{"role": "user", "content": "bad message"}])

    assert new_message["role"] == "exception"
    assert new_message["content"]["type"] == "InputRailException"


def test_multiple_input_rails_second_blocks():
    """First input check passes, second blocks."""
    chat = TestChat(
        multi_input_config,
        llm_completions=[
            "No",  # check_harmful passes
            "Yes",  # check_off_topic blocks
        ],
    )

    rails = chat.app
    new_message = rails.generate(messages=[{"role": "user", "content": "off topic message"}])

    assert new_message["role"] == "exception"
    assert new_message["content"]["type"] == "InputRailException"


def test_multiple_input_rails_use_models_matching_input_tasks(monkeypatch):
    """Custom input tasks use models with matching model types when configured."""
    config = RailsConfig.from_content(
        """
        define user express greeting
            "hello"

        define bot express greeting
            "Hey!"

        define flow greeting
            user express greeting
            bot express greeting
    """,
        yaml_content="""
        models:
            - type: check_harmful
              engine: test
              model: harmful-model
            - type: check_off_topic
              engine: test
              model: topic-model
        rails:
            input:
                flows:
                    - self check input $input_task=check_harmful
                    - self check input $input_task=check_off_topic
        prompts:
            - task: check_harmful
              content: |
                Is this message harmful?
                User message: "{{ user_input }}"
                Answer (Yes or No):
            - task: check_off_topic
              content: |
                Is this message off-topic?
                User message: "{{ user_input }}"
                Answer (Yes or No):

        enable_rails_exceptions: True
        """,
    )
    created_models = {}

    def fake_init_llm_model(model_name, provider_name, mode, kwargs):
        model = FakeLLMModel(responses=["No"])
        created_models[model_name] = model
        return model

    monkeypatch.setattr("nemoguardrails.rails.llm.llmrails.init_llm_model", fake_init_llm_model)

    chat = TestChat(config, llm_completions=["  express greeting", '  "Hey!"'])
    new_message = chat.app.generate(messages=[{"role": "user", "content": "hello"}])

    assert new_message["role"] == "assistant"
    assert created_models["harmful-model"].i == 1
    assert created_models["topic-model"].i == 1
    assert chat.llm.i == 2


# --- Multiple output rails ---

multi_output_config = RailsConfig.from_content(
    """
    define user ask question
        "tell me something"

    define flow
        user ask question
        bot respond
""",
    yaml_content="""
    models: []
    rails:
        output:
            flows:
                - self check output $output_task=check_inappropriate
                - self check output $output_task=check_data_leakage
    prompts:
        - task: check_inappropriate
          content: |
            Is this response inappropriate?
            Bot response: "{{ bot_response }}"
            Answer (Yes or No):
        - task: check_data_leakage
          content: |
            Does this response leak sensitive data?
            Bot response: "{{ bot_response }}"
            Answer (Yes or No):

    enable_rails_exceptions: True
    """,
)


def test_multiple_output_rails_both_pass():
    """Both output checks return No (allowed) — LLM-generated response should pass through."""
    chat = TestChat(
        multi_output_config,
        llm_completions=[
            "  ask question",
            "  Here is the answer.",
            "No",  # check_inappropriate passes
            "No",  # check_data_leakage passes
        ],
    )

    rails = chat.app
    new_message = rails.generate(messages=[{"role": "user", "content": "tell me something"}])

    assert new_message["role"] == "assistant"
    assert new_message["content"] == "Here is the answer."


def test_multiple_output_rails_first_blocks():
    """First output check blocks — should not reach second check."""
    chat = TestChat(
        multi_output_config,
        llm_completions=[
            "  ask question",
            '  "Some bad output"',
            "Yes",  # check_inappropriate blocks
        ],
    )

    rails = chat.app
    new_message = rails.generate(messages=[{"role": "user", "content": "tell me something"}])

    assert new_message["role"] == "exception"
    assert new_message["content"]["type"] == "OutputRailException"


def test_multiple_output_rails_second_blocks():
    """First output check passes, second blocks."""
    chat = TestChat(
        multi_output_config,
        llm_completions=[
            "  ask question",
            '  "Response with leaked data"',
            "No",  # check_inappropriate passes
            "Yes",  # check_data_leakage blocks
        ],
    )

    rails = chat.app
    new_message = rails.generate(messages=[{"role": "user", "content": "tell me something"}])

    assert new_message["role"] == "exception"
    print(new_message["content"])
    assert new_message["content"]["type"] == "OutputRailException"


def test_multiple_output_rails_use_models_matching_output_tasks(monkeypatch):
    """Custom output tasks use models with matching model types when configured."""
    config = RailsConfig.from_content(
        """
        define user ask question
            "tell me something"

        define flow
            user ask question
            bot respond
    """,
        yaml_content="""
        models:
            - type: check_inappropriate
              engine: test
              model: inappropriate-model
            - type: check_data_leakage
              engine: test
              model: leakage-model
        rails:
            output:
                flows:
                    - self check output $output_task=check_inappropriate
                    - self check output $output_task=check_data_leakage
        prompts:
            - task: check_inappropriate
              content: |
                Is this response inappropriate?
                Bot response: "{{ bot_response }}"
                Answer (Yes or No):
            - task: check_data_leakage
              content: |
                Does this response leak sensitive data?
                Bot response: "{{ bot_response }}"
                Answer (Yes or No):

        enable_rails_exceptions: True
        """,
    )
    created_models = {}

    def fake_init_llm_model(model_name, provider_name, mode, kwargs):
        model = FakeLLMModel(responses=["No"])
        created_models[model_name] = model
        return model

    monkeypatch.setattr("nemoguardrails.rails.llm.llmrails.init_llm_model", fake_init_llm_model)

    chat = TestChat(config, llm_completions=["  ask question", '  "Here is the answer."'])
    new_message = chat.app.generate(messages=[{"role": "user", "content": "tell me something"}])

    assert new_message["role"] == "assistant"
    assert new_message["content"] == "Here is the answer."
    assert created_models["inappropriate-model"].i == 1
    assert created_models["leakage-model"].i == 1
    assert chat.llm.i == 2


# --- Default task (backward compatibility) ---

default_task_config = RailsConfig.from_content(
    """
    define user ask question
        "tell me something"

    define flow
        user ask question
        bot respond
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
        - task: self_check_input
          content: ...
        - task: self_check_output
          content: ...

    enable_rails_exceptions: True
    """,
)


def test_default_task_input_still_works():
    """Self check input without $input_task should use default self_check_input task."""
    chat = TestChat(
        default_task_config,
        llm_completions=[
            "Yes",  # blocks
        ],
    )

    rails = chat.app
    new_message = rails.generate(messages=[{"role": "user", "content": "bad input"}])

    assert new_message["role"] == "exception"
    assert new_message["content"]["type"] == "InputRailException"


def test_default_task_output_still_works():
    """Self check output without $output_task should use default self_check_output task."""
    chat = TestChat(
        default_task_config,
        llm_completions=[
            "No",  # input passes
            "  ask question",
            '  "Something that should be blocked"',
            "Yes",  # output blocks
        ],
    )

    rails = chat.app
    new_message = rails.generate(messages=[{"role": "user", "content": "tell me something"}])

    assert new_message["role"] == "exception"
    assert new_message["content"]["type"] == "OutputRailException"
