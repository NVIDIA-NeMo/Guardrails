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

from typing import Any, cast

import pytest

from nemoguardrails import RailsConfig
from nemoguardrails.actions.rail_outcome import RailOutcome
from nemoguardrails.library.llama_guard.actions import (
    llama_guard_check_input,
    llama_guard_check_output,
)
from nemoguardrails.llm.taskmanager import LLMTaskManager
from tests.llama_guard_fixtures import (
    LLAMA_GUARD_SAFE_POLICY_VIOLATIONS,
    LLAMA_GUARD_UNPARSEABLE_POLICY_VIOLATIONS,
    LLAMA_GUARD_UNSAFE_POLICY_VIOLATIONS,
)
from tests.utils import FakeLLMModel, TestChat

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


class _LlamaGuardTaskManager:
    def render_task_prompt(self, task: Any, context: dict[str, Any]) -> str:
        return "prompt"

    def get_stop_tokens(self, task: Any) -> list[str]:
        return []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("llm_response", "expected"),
    [
        ("safe", RailOutcome.allow(metadata={"policy_violations": LLAMA_GUARD_SAFE_POLICY_VIOLATIONS})),
        ("unsafe s1", RailOutcome.block(metadata={"policy_violations": LLAMA_GUARD_UNSAFE_POLICY_VIOLATIONS})),
        ("error", RailOutcome.block(metadata={"policy_violations": LLAMA_GUARD_UNPARSEABLE_POLICY_VIOLATIONS})),
    ],
)
async def test_llama_guard_actions_return_rail_outcome(llm_response, expected):
    task_manager = cast(LLMTaskManager, _LlamaGuardTaskManager())
    context = {"user_message": "hello", "bot_message": "hello"}

    for action_func in (llama_guard_check_input, llama_guard_check_output):
        outcome = await action_func(
            llm_task_manager=task_manager,
            context=context,
            llms={"llama_guard": FakeLLMModel(responses=[llm_response])},
            model_name="llama_guard",
        )

        assert outcome == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("action_func", [llama_guard_check_input, llama_guard_check_output])
async def test_llama_guard_actions_select_named_model_from_registry(action_func):
    task_manager = cast(LLMTaskManager, _LlamaGuardTaskManager())
    selected_llm = FakeLLMModel(responses=["safe"])
    other_llm = FakeLLMModel(responses=["unsafe"])

    outcome = await action_func(
        llm_task_manager=task_manager,
        context={"user_message": "hello", "bot_message": "hello"},
        llms={"llama_guard": selected_llm, "other": other_llm},
        model_name="llama_guard",
    )

    assert outcome.is_blocked is False
    assert selected_llm.inference_count == 1
    assert other_llm.inference_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("action_func", [llama_guard_check_input, llama_guard_check_output])
async def test_llama_guard_actions_fail_before_call_without_model(action_func):
    task_manager = cast(LLMTaskManager, _LlamaGuardTaskManager())

    with pytest.raises(ValueError, match="Llama Guard model"):
        await action_func(
            llm_task_manager=task_manager,
            context={"user_message": "hello", "bot_message": "hello"},
            llms={"llama_guard": None},
            model_name="llama_guard",
        )


def test_llama_guard_output_action_metadata_is_registration_only():
    assert set(getattr(llama_guard_check_output, "action_meta")) == {
        "name",
        "is_system_action",
        "execute_async",
    }


def test_llama_guard_check_all_safe():
    """
    Test the chat flow when both llama_guard_check_input and llama_guard_check_output actions return "safe"
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

    llama_guard_model = FakeLLMModel(
        responses=[
            "safe",  # llama_guard_check_input
            "safe",  # llama_guard_check_output
        ]
    )
    chat.app.register_action_param("llms", {"llama_guard": llama_guard_model})

    _ = chat >> "Hi"
    _ = chat << "Hi there! How are you doing?"


def test_llama_guard_check_input_unsafe():
    """
    Test the chat flow when the llama_guard_check_input action returns "unsafe"
    """
    config = RailsConfig.from_content(colang_content=COLANG_CONFIG, yaml_content=YAML_CONFIG)
    chat = TestChat(
        config,
        llm_completions=[
            # Since input is unsafe, the main llm doesn't need to perform any of
            # generate_user_intent, generate_next_step, or generate_bot_message
            # Dev note: iff the input was safe, this empty llm_completions list would result in a test failure.
        ],
    )

    llama_guard_model = FakeLLMModel(
        responses=[
            "unsafe",  # llama_guard_check_input
        ]
    )
    chat.app.register_action_param("llms", {"llama_guard": llama_guard_model})

    _ = chat >> "Unsafe input"
    _ = chat << "I'm sorry, I can't respond to that."


def test_llama_guard_check_input_unparseable_fail_closed():
    """
    Test the chat flow when llama_guard_check_input returns an unparseable response
    """
    config = RailsConfig.from_content(colang_content=COLANG_CONFIG, yaml_content=YAML_CONFIG)
    chat = TestChat(
        config,
        llm_completions=[
            # Since input is unsafe, the main llm doesn't need to perform any of
            # generate_user_intent, generate_next_step, or generate_bot_message
            # Dev note: iff the input was safe, this empty llm_completions list would result in a test failure.
        ],
    )

    llama_guard_model = FakeLLMModel(
        responses=[
            "error",  # unparseable llama_guard_check_input response
        ]
    )
    chat.app.register_action_param("llms", {"llama_guard": llama_guard_model})

    _ = chat >> "Unsafe input"
    _ = chat << "I'm sorry, I can't respond to that."


def test_llama_guard_check_output_unsafe():
    """
    Test the chat flow when the llama_guard_check_input action raises an error
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

    llama_guard_model = FakeLLMModel(
        responses=[
            "safe",  # llama_guard_check_input
            "unsafe",  # llama_guard_check_output
        ]
    )
    chat.app.register_action_param("llms", {"llama_guard": llama_guard_model})

    _ = chat >> "Unsafe input"
    _ = chat << "I'm sorry, I can't respond to that."


def test_llama_guard_check_output_unparseable_fail_closed():
    """
    Test the chat flow when llama_guard_check_output returns an unparseable response
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

    llama_guard_model = FakeLLMModel(
        responses=[
            "safe",  # llama_guard_check_input
            "error",  # unparseable llama_guard_check_output response
        ]
    )
    chat.app.register_action_param("llms", {"llama_guard": llama_guard_model})

    _ = chat >> "Unsafe input"
    _ = chat << "I'm sorry, I can't respond to that."
