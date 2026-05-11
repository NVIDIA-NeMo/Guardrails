# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from nemoguardrails.llm.taskmanager import LLMTaskManager


@pytest.fixture(scope="module")
def fake_base64():
    return "iVBORw0KGgoAAAANSUhEUg" * 5000


def _make_vision_config():
    return RailsConfig.from_path("./examples/configs/content_safety_vision")


def _make_multimodal_input(base64_data):
    return [
        {"type": "text", "text": "What is this?"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_data}"}},
    ]


def test_render_preserves_multimodal_list(fake_base64):
    config = _make_vision_config()
    tm = LLMTaskManager(config)
    user_input = _make_multimodal_input(fake_base64)

    prompt = tm.render_task_prompt(
        task="content_safety_check_input $model=vision_rails",
        context={"user_input": user_input, "reasoning_enabled": False},
    )

    user_msg = prompt[-1]
    assert user_msg["type"] == "user"
    assert isinstance(user_msg["content"], list)
    assert user_msg["content"] == user_input


def test_render_text_only_unchanged():
    config = _make_vision_config()
    tm = LLMTaskManager(config)

    prompt = tm.render_task_prompt(
        task="content_safety_check_input $model=vision_rails",
        context={"user_input": "Is this safe?", "reasoning_enabled": False},
    )

    user_msg = prompt[-1]
    assert user_msg["type"] == "user"
    assert isinstance(user_msg["content"], str)
    assert "Is this safe?" in user_msg["content"]


def test_render_multimodal_no_base64_in_string_content(fake_base64):
    config = _make_vision_config()
    tm = LLMTaskManager(config)
    user_input = _make_multimodal_input(fake_base64)

    prompt = tm.render_task_prompt(
        task="content_safety_check_input $model=vision_rails",
        context={"user_input": user_input, "reasoning_enabled": False},
    )

    for msg in prompt:
        if isinstance(msg["content"], str):
            assert "iVBORw0KGgoAAAANSUhEUg" not in msg["content"]


def test_render_mixed_template_stringifies():
    config = RailsConfig.from_content(
        config={
            "models": [{"type": "main", "engine": "openai", "model": "gpt-4o"}],
            "prompts": [
                {
                    "task": "test_mixed",
                    "messages": [
                        {"type": "user", "content": "hello {{ var }}"},
                    ],
                }
            ],
        },
    )
    tm = LLMTaskManager(config)
    list_val = [{"type": "text", "text": "world"}]

    prompt = tm.render_task_prompt(
        task="test_mixed",
        context={"var": list_val},
    )

    user_msg = prompt[-1]
    assert isinstance(user_msg["content"], str)


def test_prompt_context_list_overrides_context(fake_base64):
    """A list registered via ``prompt_context`` for a single-variable template
    must override a scalar value supplied through ``context`` for the same
    variable. Covers the ``value = candidate`` branch in
    ``_resolve_message_content``.
    """
    config = _make_vision_config()
    tm = LLMTaskManager(config)
    list_value = _make_multimodal_input(fake_base64)
    tm.register_prompt_context("user_input", list_value)

    prompt = tm.render_task_prompt(
        task="content_safety_check_input $model=vision_rails",
        context={"user_input": "ignored scalar fallback", "reasoning_enabled": False},
    )

    user_msg = prompt[-1]
    assert user_msg["type"] == "user"
    assert isinstance(user_msg["content"], list)
    assert user_msg["content"] == list_value


def test_prompt_context_callable_invoked_once():
    config = RailsConfig.from_content(
        config={
            "models": [{"type": "main", "engine": "openai", "model": "gpt-4o"}],
            "prompts": [
                {
                    "task": "test_single_var",
                    "messages": [
                        {"type": "user", "content": "{{ user_input }}"},
                    ],
                }
            ],
        },
    )
    tm = LLMTaskManager(config)
    call_count = {"n": 0}

    def side_effect_callable():
        call_count["n"] += 1
        return "hello"

    tm.register_prompt_context("user_input", side_effect_callable)

    tm.render_task_prompt(task="test_single_var", context={})

    assert call_count["n"] == 1


def test_render_empty_list_is_dropped():
    config = _make_vision_config()
    tm = LLMTaskManager(config)

    prompt = tm.render_task_prompt(
        task="content_safety_check_input $model=vision_rails",
        context={"user_input": [], "reasoning_enabled": False},
    )

    for msg in prompt:
        assert msg["content"] != []


def test_rendered_prompt_length_reasonable():
    big_base64 = "A" * 100_000
    config = _make_vision_config()
    tm = LLMTaskManager(config)
    user_input = _make_multimodal_input(base64_data=big_base64)

    prompt = tm.render_task_prompt(
        task="content_safety_check_input $model=vision_rails",
        context={"user_input": user_input, "reasoning_enabled": False},
    )

    total_text = 0
    for msg in prompt:
        content = msg["content"]
        if isinstance(content, str):
            total_text += len(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    total_text += len(item.get("text", ""))

    assert total_text < 1000
