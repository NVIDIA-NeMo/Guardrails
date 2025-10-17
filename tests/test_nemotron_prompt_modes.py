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


"""Tests for Nemotron prompt format and structure.

This module verifies that:
1. Nemotron models use message-based prompts from nemotron.yml
2. DeepSeek models use content-based prompts from deepseek.yml
3. Some tasks in nemotron.yml have "detailed thinking" for internal steps
   (generate_bot_message and generate_value)
4. Other tasks (generate_user_intent and generate_next_steps) in nemotron.yml don't have "detailed thinking"
"""

import pytest
import yaml

from nemoguardrails import RailsConfig
from nemoguardrails.llm.prompts import _get_prompt, _prompts, get_prompt
from nemoguardrails.llm.types import Task

NEMOTRON_MODEL = "nvidia/llama-3.1-nemotron-ultra-253b-v1"
DEEPSEEK_MODEL = "deepseek-ai/deepseek-v2"


def colang_config():
    """Basic colang configuration for testing."""
    return """
    define user express greeting
        "hi"
        "hello"

    define flow
        user express greeting
        bot express greeting
    """


def create_config(model=NEMOTRON_MODEL):
    """Create a test config with specified model."""
    config = {"models": [{"type": "main", "engine": "nim", "model": model}]}
    return yaml.dump(config)


def test_nemotron_uses_messages():
    """Verify Nemotron models use message-based format from nemotron.yml."""
    config = RailsConfig.from_content(colang_config(), yaml_content=create_config())
    prompt = get_prompt(config, Task.GENERATE_BOT_MESSAGE)

    assert hasattr(prompt, "messages") and prompt.messages is not None
    assert not hasattr(prompt, "content") or prompt.content is None
    assert "nemotron" in prompt.models


def test_tasks_with_detailed_thinking():
    """Verify tasks that should have detailed thinking in system messages."""
    config = RailsConfig.from_content(colang_config(), yaml_content=create_config())

    for task in [Task.GENERATE_BOT_MESSAGE, Task.GENERATE_VALUE]:
        prompt = get_prompt(config, task)

        assert hasattr(prompt, "messages") and prompt.messages is not None

        # two system messages (one for detailed thinking, one for instructions)
        system_messages = [
            msg
            for msg in prompt.messages
            if hasattr(msg, "type") and msg.type == "system"
        ]
        assert (
            len(system_messages) == 2
        ), f"Task {task} should have exactly two system messages"

        assert (
            "detailed thinking on" in system_messages[0].content
        ), f"Task {task} should have 'detailed thinking on' in first system message"


def test_tasks_without_detailed_thinking():
    """Verify tasks that should have only one system message (no detailed thinking)."""
    config = RailsConfig.from_content(colang_config(), yaml_content=create_config())

    for task in [Task.GENERATE_USER_INTENT, Task.GENERATE_NEXT_STEPS]:
        prompt = get_prompt(config, task)

        assert hasattr(prompt, "messages") and prompt.messages is not None

        # one system message (no detailed thinking)
        system_messages = [
            msg
            for msg in prompt.messages
            if hasattr(msg, "type") and msg.type == "system"
        ]
        assert (
            len(system_messages) == 1
        ), f"Task {task} should have exactly one system message"

        assert (
            "detailed thinking on" not in system_messages[0].content
        ), f"Task {task} should not have 'detailed thinking on' in system message"


def test_deepseek_uses_deepseek_yml():
    """Verify DeepSeek models use deepseek.yml."""
    config = RailsConfig.from_content(
        colang_config(), yaml_content=create_config(DEEPSEEK_MODEL)
    )

    for task in [Task.GENERATE_BOT_MESSAGE, Task.GENERATE_USER_INTENT]:
        prompt = get_prompt(config, task)

        # should use content-based format from deepseek.yml
        assert hasattr(prompt, "content") and prompt.content is not None
        assert not hasattr(prompt, "messages") or prompt.messages is None

        # should have "Use a short thinking process" from deepseek.yml
        assert "IMPORTANT: Use a short thinking process" in prompt.content
        assert "deepseek" in prompt.models
        assert "nemotron" not in prompt.models


def test_prompt_selection_mechanism():
    """Test the core prompt selection mechanism directly."""
    task_name = Task.GENERATE_BOT_MESSAGE.value
    nemotron_model = NEMOTRON_MODEL
    deepseek_model = DEEPSEEK_MODEL

    # Nemotron model -> message-based prompt
    nemotron_prompt = _get_prompt(task_name, nemotron_model, None, _prompts)
    assert hasattr(nemotron_prompt, "messages")
    assert "nemotron" in nemotron_prompt.models

    # DeepSeek model -> content-based prompt
    deepseek_prompt = _get_prompt(task_name, deepseek_model, None, _prompts)
    assert hasattr(deepseek_prompt, "content")
    assert "deepseek" in deepseek_prompt.models
    assert "nemotron" not in deepseek_prompt.models


ACTUAL_NEMOTRON_MODELS_FOR_TEST = [
    "nvidia/llama-3.1-nemotron-51b-instruct",
    "nvidia/llama-3.1-nemotron-70b-instruct",
    "nvidia/llama-3.1-nemotron-nano-8b-v1",
    "nvidia/llama-3.1-nemotron-ultra-253b-v1",
    "nvidia/llama-3.3-nemotron-super-49b-v1",
    "nvidia/nemotron-4-340b-instruct",
    "llama-3.1-nemotron-custom-variant",
    "nemotron-generic-variant",
    "nvidia/nemotron-mini-4b-instruct",
    "nvidia/llama-3.1-nemotron-70b-instruct",
]

ACTUAL_LLAMA3_MODELS_FOR_TEST = [
    "meta/llama-3.1-405b-instruct",
    "meta/llama-3.1-70b-instruct",
    "meta/llama-3.1-8b-instruct",
    "meta/llama3-70b-instruct",
    "meta/llama3-8b-instruct",
    "meta/llama-3.3-70b-instruct",
    "nvidia/usdcode-llama-3.1-70b-instruct",
]

EXPECTED_NEMOTRON_PROMPT_MODELS_FIELD = sorted(["nvidia/nemotron", "nemotron"])
EXPECTED_LLAMA3_PROMPT_MODELS_FIELD = sorted(
    ["meta/llama-3", "meta/llama3", "nvidia/usdcode-llama-3"]
)


@pytest.mark.parametrize("model_name", ACTUAL_NEMOTRON_MODELS_FOR_TEST)
def test_specific_nemotron_model_variants_select_nemotron_prompt(model_name):
    """Verify that specific Nemotron model variants correctly select the Nemotron prompt."""
    config = RailsConfig.from_content(
        colang_config(), yaml_content=create_config(model=model_name)
    )
    prompt = get_prompt(config, Task.GENERATE_BOT_MESSAGE)

    assert (
        hasattr(prompt, "messages") and prompt.messages is not None
    ), f"Prompt for {model_name} should be message-based for Nemotron."
    assert (
        not hasattr(prompt, "content") or prompt.content is None
    ), f"Prompt for {model_name} should not have content for Nemotron."

    # sort because the order within the list in the YAML might not be guaranteed upon loading
    assert (
        sorted(prompt.models) == EXPECTED_NEMOTRON_PROMPT_MODELS_FIELD
    ), f"Prompt for {model_name} selected wrong model identifiers. Expected {EXPECTED_NEMOTRON_PROMPT_MODELS_FIELD}, Got {sorted(prompt.models)}"


@pytest.mark.parametrize("model_name", ACTUAL_LLAMA3_MODELS_FOR_TEST)
def test_specific_llama3_model_variants_select_llama3_prompt(model_name):
    """Verify that specific Llama3 model variants correctly select the Llama3 prompt."""

    config = RailsConfig.from_content(
        colang_config(), yaml_content=create_config(model=model_name)
    )
    prompt = get_prompt(config, Task.GENERATE_BOT_MESSAGE)

    assert (
        hasattr(prompt, "messages") and prompt.messages is not None
    ), f"Prompt for {model_name} should be message-based for Llama3."

    assert (
        sorted(prompt.models) == EXPECTED_LLAMA3_PROMPT_MODELS_FIELD
    ), f"Prompt for {model_name} selected wrong model identifiers. Expected {EXPECTED_LLAMA3_PROMPT_MODELS_FIELD}, Got {sorted(prompt.models)}"
