# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from pydantic import ValidationError

from nemoguardrails.rails.llm.config import TaskPrompt


def test_task_prompt_valid_content():
    prompt = TaskPrompt(task="example_task", content="This is a valid prompt.")
    assert prompt.task == "example_task"
    assert prompt.content == "This is a valid prompt."
    assert prompt.messages is None


def test_task_prompt_valid_messages():
    prompt = TaskPrompt(task="example_task", messages=["Hello", "How can I help you?"])
    assert prompt.task == "example_task"
    assert prompt.messages == ["Hello", "How can I help you?"]
    assert prompt.content is None


def test_task_prompt_missing_content_and_messages():
    with pytest.raises(ValidationError) as excinfo:
        TaskPrompt(task="example_task")
    assert "One of `content` or `messages` must be provided." in str(excinfo.value)


def test_task_prompt_both_content_and_messages():
    with pytest.raises(ValidationError) as excinfo:
        TaskPrompt(
            task="example_task",
            content="This is a prompt.",
            messages=["Hello", "How can I help you?"],
        )
    assert "Only one of `content` or `messages` must be provided." in str(excinfo.value)


def test_task_prompt_models_validation():
    prompt = TaskPrompt(
        task="example_task",
        content="Test prompt",
        models=["openai", "openai/gpt-3.5-turbo"],
    )
    assert prompt.models == ["openai", "openai/gpt-3.5-turbo"]

    prompt = TaskPrompt(task="example_task", content="Test prompt", models=[])
    assert prompt.models == []

    prompt = TaskPrompt(task="example_task", content="Test prompt", models=None)
    assert prompt.models is None


def test_task_prompt_max_length_validation():
    prompt = TaskPrompt(task="example_task", content="Test prompt")
    assert prompt.max_length == 16000

    prompt = TaskPrompt(task="example_task", content="Test prompt", max_length=1000)
    assert prompt.max_length == 1000

    with pytest.raises(ValidationError) as excinfo:
        TaskPrompt(task="example_task", content="Test prompt", max_length=0)
    assert "Input should be greater than or equal to 1" in str(excinfo.value)

    with pytest.raises(ValidationError) as excinfo:
        TaskPrompt(task="example_task", content="Test prompt", max_length=-1)
    assert "Input should be greater than or equal to 1" in str(excinfo.value)


def test_task_prompt_mode_validation():
    prompt = TaskPrompt(task="example_task", content="Test prompt")
    # default mode is "standard"
    assert prompt.mode == "standard"

    prompt = TaskPrompt(task="example_task", content="Test prompt", mode="chat")
    assert prompt.mode == "chat"

    prompt = TaskPrompt(task="example_task", content="Test prompt", mode=None)
    assert prompt.mode is None


def test_task_prompt_stop_tokens_validation():
    prompt = TaskPrompt(
        task="example_task", content="Test prompt", stop=["\n", "Human:", "Assistant:"]
    )
    assert prompt.stop == ["\n", "Human:", "Assistant:"]

    prompt = TaskPrompt(task="example_task", content="Test prompt", stop=[])
    assert prompt.stop == []

    prompt = TaskPrompt(task="example_task", content="Test prompt", stop=None)
    assert prompt.stop is None

    with pytest.raises(ValidationError) as excinfo:
        TaskPrompt(task="example_task", content="Test prompt", stop=[1, 2, 3])
    assert "Input should be a valid string" in str(excinfo.value)


def test_task_prompt_max_tokens_validation():
    prompt = TaskPrompt(task="example_task", content="Test prompt")
    assert prompt.max_tokens is None

    prompt = TaskPrompt(task="example_task", content="Test prompt", max_tokens=1000)
    assert prompt.max_tokens == 1000

    with pytest.raises(ValidationError) as excinfo:
        TaskPrompt(task="example_task", content="Test prompt", max_tokens=0)
    assert "Input should be greater than or equal to 1" in str(excinfo.value)

    with pytest.raises(ValidationError) as excinfo:
        TaskPrompt(task="example_task", content="Test prompt", max_tokens=-1)
    assert "Input should be greater than or equal to 1" in str(excinfo.value)
