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

import pytest

from nemoguardrails import RailsConfig
from nemoguardrails.library.self_check.utils import (
    SELF_CHECK_INPUT_DEFAULT_TASK,
    SELF_CHECK_INPUT_FLOW,
    SELF_CHECK_INPUT_TASK_PARAM,
    get_self_check_llm,
    get_self_check_task_from_rail,
    resolve_self_check_task,
    run_self_check_task,
)
from nemoguardrails.llm.taskmanager import LLMTaskManager
from nemoguardrails.testing.fake_model import FakeLLMModel
from tests.utils import TestChat

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
                - self check input $task=check_harmful
                - self check input $task=check_off_topic
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
            "No",
            "No",
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
            "Yes",
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
            "No",
            "Yes",
        ],
    )

    rails = chat.app
    new_message = rails.generate(messages=[{"role": "user", "content": "off topic message"}])

    assert new_message["role"] == "exception"
    assert new_message["content"]["type"] == "InputRailException"


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
                - self check output $task=check_inappropriate
                - self check output $task=check_data_leakage
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
            "No",
            "No",
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
            "Yes",
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
            "No",
            "Yes",
        ],
    )

    rails = chat.app
    new_message = rails.generate(messages=[{"role": "user", "content": "tell me something"}])

    assert new_message["role"] == "exception"
    assert new_message["content"]["type"] == "OutputRailException"


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
    """Self check input without $task should use default self_check_input task."""
    chat = TestChat(
        default_task_config,
        llm_completions=[
            "Yes",
        ],
    )

    rails = chat.app
    new_message = rails.generate(messages=[{"role": "user", "content": "bad input"}])

    assert new_message["role"] == "exception"
    assert new_message["content"]["type"] == "InputRailException"


def test_mixed_input_rails_run_custom_and_default_tasks():
    config = RailsConfig.from_content(
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
                    - self check input $task=check_harmful
                    - self check input
        prompts:
            - task: check_harmful
              content: |
                Is this message harmful?
                User message: "{{ user_input }}"
                Answer (Yes or No):
            - task: self_check_input
              content: |
                Is this message safe?
                User message: "{{ user_input }}"
                Answer (Yes or No):

        enable_rails_exceptions: True
        """,
    )
    custom_llm = FakeLLMModel(responses=["No", "No"])
    default_llm = FakeLLMModel(responses=["Yes"])

    chat = TestChat(
        config,
        llm_completions=[
            "  express greeting",
            '  "Hey!"',
        ],
    )
    rails = chat.app
    rails.runtime.registered_action_params["llms"]["check_harmful"] = custom_llm
    rails.runtime.registered_action_params["llms"]["self_check_input"] = default_llm

    new_message = rails.generate(messages=[{"role": "user", "content": "hello"}])

    assert custom_llm.inference_count == 1
    assert default_llm.inference_count == 1
    assert new_message["role"] == "exception"
    assert new_message["content"]["type"] == "InputRailException"


def test_default_task_output_still_works():
    """Self check output without $task should use default self_check_output task."""
    chat = TestChat(
        default_task_config,
        llm_completions=[
            "No",
            "  ask question",
            '  "Something that should be blocked"',
            "Yes",
        ],
    )

    rails = chat.app
    new_message = rails.generate(messages=[{"role": "user", "content": "tell me something"}])

    assert new_message["role"] == "exception"
    assert new_message["content"]["type"] == "OutputRailException"


def test_mixed_output_rails_run_custom_and_default_tasks():
    config = RailsConfig.from_content(
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
                    - self check output $task=check_inappropriate
                    - self check output
        prompts:
            - task: check_inappropriate
              content: |
                Is this response inappropriate?
                Bot response: "{{ bot_response }}"
                Answer (Yes or No):
            - task: self_check_output
              content: |
                Is this response safe?
                Bot response: "{{ bot_response }}"
                Answer (Yes or No):

        enable_rails_exceptions: True
        """,
    )
    custom_llm = FakeLLMModel(responses=["No", "No"])
    default_llm = FakeLLMModel(responses=["Yes"])

    chat = TestChat(
        config,
        llm_completions=[
            "  ask question",
            '  "Response that the default rail blocks"',
        ],
    )
    rails = chat.app
    rails.runtime.registered_action_params["llms"]["check_inappropriate"] = custom_llm
    rails.runtime.registered_action_params["llms"]["self_check_output"] = default_llm

    new_message = rails.generate(messages=[{"role": "user", "content": "tell me something"}])

    assert custom_llm.inference_count == 1
    assert default_llm.inference_count == 1
    assert new_message["role"] == "exception"
    assert new_message["content"]["type"] == "OutputRailException"


per_task_input_config = RailsConfig.from_content(
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
                - self check input $task=check_harmful
                - self check input $task=check_off_topic
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


def test_per_task_llm_input_uses_task_specific_model():
    """Each input check task should use its own LLM when configured in the llms dict."""
    harmful_llm = FakeLLMModel(responses=["No"])
    off_topic_llm = FakeLLMModel(responses=["No"])

    chat = TestChat(
        per_task_input_config,
        llm_completions=[
            "  express greeting",
            '  "Hey!"',
        ],
    )

    rails = chat.app
    rails.runtime.registered_action_params["llms"]["check_harmful"] = harmful_llm
    rails.runtime.registered_action_params["llms"]["check_off_topic"] = off_topic_llm

    new_message = rails.generate(messages=[{"role": "user", "content": "hello"}])

    assert new_message["role"] == "assistant"
    assert harmful_llm.inference_count == 1
    assert off_topic_llm.inference_count == 1


def test_per_task_llm_input_first_blocks_skips_second():
    """When the first per-task LLM blocks, the second task-specific LLM should not be called."""
    harmful_llm = FakeLLMModel(responses=["Yes"])
    off_topic_llm = FakeLLMModel(responses=["No"])

    chat = TestChat(
        per_task_input_config,
        llm_completions=[],
    )

    rails = chat.app
    rails.runtime.registered_action_params["llms"]["check_harmful"] = harmful_llm
    rails.runtime.registered_action_params["llms"]["check_off_topic"] = off_topic_llm

    new_message = rails.generate(messages=[{"role": "user", "content": "bad message"}])

    assert new_message["role"] == "exception"
    assert new_message["content"]["type"] == "InputRailException"
    assert harmful_llm.inference_count == 1
    assert off_topic_llm.inference_count == 0


per_task_output_config = RailsConfig.from_content(
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
                - self check output $task=check_inappropriate
                - self check output $task=check_data_leakage
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


def test_per_task_llm_output_uses_task_specific_model():
    """Each output check task should use its own LLM when configured in the llms dict."""
    inappropriate_llm = FakeLLMModel(responses=["No"])
    data_leakage_llm = FakeLLMModel(responses=["No"])

    chat = TestChat(
        per_task_output_config,
        llm_completions=[
            "  ask question",
            "  Here is the answer.",
        ],
    )

    rails = chat.app
    rails.runtime.registered_action_params["llms"]["check_inappropriate"] = inappropriate_llm
    rails.runtime.registered_action_params["llms"]["check_data_leakage"] = data_leakage_llm

    new_message = rails.generate(messages=[{"role": "user", "content": "tell me something"}])

    assert new_message["role"] == "assistant"
    assert new_message["content"] == "Here is the answer."
    assert inappropriate_llm.inference_count == 1
    assert data_leakage_llm.inference_count == 1


def test_per_task_llm_output_first_blocks_skips_second():
    """When the first per-task output LLM blocks, the second should not be called."""
    inappropriate_llm = FakeLLMModel(responses=["Yes"])
    data_leakage_llm = FakeLLMModel(responses=["No"])

    chat = TestChat(
        per_task_output_config,
        llm_completions=[
            "  ask question",
            '  "Some bad output"',
        ],
    )

    rails = chat.app
    rails.runtime.registered_action_params["llms"]["check_inappropriate"] = inappropriate_llm
    rails.runtime.registered_action_params["llms"]["check_data_leakage"] = data_leakage_llm

    new_message = rails.generate(messages=[{"role": "user", "content": "tell me something"}])

    assert new_message["role"] == "exception"
    assert new_message["content"]["type"] == "OutputRailException"
    assert inappropriate_llm.inference_count == 1
    assert data_leakage_llm.inference_count == 0


def test_parallel_input_rail_uses_custom_task():
    config = RailsConfig.from_content(
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
                parallel: true
                flows:
                    - self check input $task=check_harmful
        prompts:
            - task: check_harmful
              content: |
                Is this message harmful?
                User message: "{{ user_input }}"
                Answer (Yes or No):
            - task: self_check_input
              content: |
                Is this message safe?
                User message: "{{ user_input }}"
                Answer (Yes or No):

        enable_rails_exceptions: True
        """,
    )
    custom_llm = FakeLLMModel(responses=["No"])
    default_llm = FakeLLMModel(responses=["Yes"])

    chat = TestChat(
        config,
        llm_completions=[
            "  express greeting",
            '  "Hey!"',
        ],
    )
    rails = chat.app
    rails.runtime.registered_action_params["llms"]["check_harmful"] = custom_llm
    rails.runtime.registered_action_params["llms"]["self_check_input"] = default_llm

    new_message = rails.generate(messages=[{"role": "user", "content": "hello"}])

    assert new_message["role"] == "assistant"
    assert custom_llm.inference_count == 1
    assert default_llm.inference_count == 0


def test_parallel_output_rail_uses_custom_task():
    config = RailsConfig.from_content(
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
                parallel: true
                flows:
                    - self check output $task=check_inappropriate
        prompts:
            - task: check_inappropriate
              content: |
                Is this response inappropriate?
                Bot response: "{{ bot_response }}"
                Answer (Yes or No):
            - task: self_check_output
              content: |
                Is this response safe?
                Bot response: "{{ bot_response }}"
                Answer (Yes or No):

        enable_rails_exceptions: True
        """,
    )
    custom_llm = FakeLLMModel(responses=["No"])
    default_llm = FakeLLMModel(responses=["Yes"])

    chat = TestChat(
        config,
        llm_completions=[
            "  ask question",
            "  Here is the answer.",
        ],
    )
    rails = chat.app
    rails.runtime.registered_action_params["llms"]["check_inappropriate"] = custom_llm
    rails.runtime.registered_action_params["llms"]["self_check_output"] = default_llm

    new_message = rails.generate(messages=[{"role": "user", "content": "tell me something"}])

    assert new_message["role"] == "assistant"
    assert new_message["content"] == "Here is the answer."
    assert custom_llm.inference_count == 1
    assert default_llm.inference_count == 0


def test_per_task_llm_falls_back_to_main_when_not_configured():
    """When no task-specific LLM is in the llms dict, it should fall back to the main LLM."""
    chat = TestChat(
        per_task_input_config,
        llm_completions=[
            "No",
            "No",
            "  express greeting",
        ],
    )

    rails = chat.app
    new_message = rails.generate(messages=[{"role": "user", "content": "hello"}])

    assert new_message["role"] == "assistant"
    assert chat.llm.inference_count == 3


def test_input_fallback_to_default_task_model():
    """When a custom task has no model, fall back to the self_check_input model."""
    default_llm = FakeLLMModel(responses=["No", "No"])

    chat = TestChat(
        per_task_input_config,
        llm_completions=[
            "  express greeting",
        ],
    )

    rails = chat.app
    rails.runtime.registered_action_params["llms"]["self_check_input"] = default_llm

    new_message = rails.generate(messages=[{"role": "user", "content": "hello"}])

    assert new_message["role"] == "assistant"
    assert default_llm.inference_count == 2
    assert chat.llm.inference_count == 1


def test_output_fallback_to_default_task_model():
    """When a custom task has no model, fall back to the self_check_output model."""
    default_llm = FakeLLMModel(responses=["No", "No"])

    chat = TestChat(
        per_task_output_config,
        llm_completions=[
            "  ask question",
            "  Here is the answer.",
        ],
    )

    rails = chat.app
    rails.runtime.registered_action_params["llms"]["self_check_output"] = default_llm

    new_message = rails.generate(messages=[{"role": "user", "content": "tell me something"}])

    assert new_message["role"] == "assistant"
    assert new_message["content"] == "Here is the answer."
    assert default_llm.inference_count == 2
    assert chat.llm.inference_count == 2


def test_input_fallback_chain_prefers_task_over_default():
    """Task-specific model takes priority over default self_check_input model."""
    task_llm = FakeLLMModel(responses=["No"])
    default_llm = FakeLLMModel(responses=["No"])

    chat = TestChat(
        per_task_input_config,
        llm_completions=[
            "  express greeting",
        ],
    )

    rails = chat.app
    rails.runtime.registered_action_params["llms"]["self_check_input"] = default_llm
    rails.runtime.registered_action_params["llms"]["check_harmful"] = task_llm

    new_message = rails.generate(messages=[{"role": "user", "content": "hello"}])

    assert new_message["role"] == "assistant"
    assert task_llm.inference_count == 1
    assert default_llm.inference_count == 1
    assert chat.llm.inference_count == 1


def test_output_fallback_chain_prefers_task_over_default():
    """Task-specific model takes priority over default self_check_output model."""
    task_llm = FakeLLMModel(responses=["No"])
    default_llm = FakeLLMModel(responses=["No"])

    chat = TestChat(
        per_task_output_config,
        llm_completions=[
            "  ask question",
            "  Here is the answer.",
        ],
    )

    rails = chat.app
    rails.runtime.registered_action_params["llms"]["self_check_output"] = default_llm
    rails.runtime.registered_action_params["llms"]["check_inappropriate"] = task_llm

    new_message = rails.generate(messages=[{"role": "user", "content": "tell me something"}])

    assert new_message["role"] == "assistant"
    assert task_llm.inference_count == 1
    assert default_llm.inference_count == 1
    assert chat.llm.inference_count == 2


def _resolve_input_task(task=None, context=None, events=None):
    return resolve_self_check_task(
        task,
        context,
        events,
        triggered_rail_key="triggered_input_rail",
        start_rail_event_type="StartInputRail",
        flow_id=SELF_CHECK_INPUT_FLOW,
        task_param=SELF_CHECK_INPUT_TASK_PARAM,
        default_task=SELF_CHECK_INPUT_DEFAULT_TASK,
    )


@pytest.mark.asyncio
async def test_run_self_check_task_uses_concrete_task_without_runtime_context():
    task_llm = FakeLLMModel(responses=["No"])
    default_llm = FakeLLMModel(responses=["Yes"])

    is_safe, response = await run_self_check_task(
        task="check_harmful",
        prompt_context={"user_input": "hello"},
        llms={
            "check_harmful": task_llm,
            "self_check_input": default_llm,
        },
        default_task=SELF_CHECK_INPUT_DEFAULT_TASK,
        main_llm=None,
        llm_task_manager=LLMTaskManager(per_task_input_config),
        lowest_temperature=per_task_input_config.lowest_temperature,
    )

    assert is_safe
    assert response == "No"
    assert task_llm.inference_count == 1
    assert default_llm.inference_count == 0


def test_get_self_check_task_from_rail_resolves_custom_and_default_tasks():
    assert (
        get_self_check_task_from_rail(
            "self check input $task=check_harmful",
            flow_id=SELF_CHECK_INPUT_FLOW,
            task_param=SELF_CHECK_INPUT_TASK_PARAM,
            default_task=SELF_CHECK_INPUT_DEFAULT_TASK,
        )
        == "check_harmful"
    )

    assert (
        get_self_check_task_from_rail(
            "self check input",
            flow_id=SELF_CHECK_INPUT_FLOW,
            task_param=SELF_CHECK_INPUT_TASK_PARAM,
            default_task=SELF_CHECK_INPUT_DEFAULT_TASK,
        )
        == SELF_CHECK_INPUT_DEFAULT_TASK
    )

    assert (
        get_self_check_task_from_rail(
            "self check output $task=check_inappropriate",
            flow_id=SELF_CHECK_INPUT_FLOW,
            task_param=SELF_CHECK_INPUT_TASK_PARAM,
            default_task=SELF_CHECK_INPUT_DEFAULT_TASK,
        )
        is None
    )


def test_resolve_self_check_task_prefers_explicit_task():
    task = _resolve_input_task(
        task="check_harmful",
        context={"triggered_input_rail": "self check input $task=check_off_topic"},
    )

    assert task == "check_harmful"


def test_resolve_self_check_task_uses_triggered_rail_context():
    task = _resolve_input_task(
        task="$task",
        context={"triggered_input_rail": "self check input $task=check_harmful"},
    )

    assert task == "check_harmful"


def test_resolve_self_check_task_uses_latest_start_rail_event():
    task = _resolve_input_task(
        task="$task",
        events=[
            {"type": "StartInputRail", "flow_id": "self check input $task=check_off_topic"},
            {"type": "SomeOtherEvent", "flow_id": "self check input $task=ignored"},
            {"type": "StartInputRail", "flow_id": "self check input $task=check_harmful"},
        ],
    )

    assert task == "check_harmful"


def test_resolve_self_check_task_uses_start_flow_params():
    task = _resolve_input_task(
        events=[
            {"type": "start_flow", "flow_id": SELF_CHECK_INPUT_FLOW, "params": {"task": "check_harmful"}},
        ],
    )

    assert task == "check_harmful"


def test_resolve_self_check_task_defaults_unresolved_placeholders():
    assert _resolve_input_task(context={"triggered_input_rail": "self check input"}) == SELF_CHECK_INPUT_DEFAULT_TASK
    assert _resolve_input_task(task="$task") == SELF_CHECK_INPUT_DEFAULT_TASK
    assert _resolve_input_task() == SELF_CHECK_INPUT_DEFAULT_TASK


def test_input_no_model_raises_error():
    """When no model is available at any fallback level, get_self_check_llm raises ValueError."""
    with pytest.raises(ValueError, match="No matching model"):
        get_self_check_llm({}, "check_harmful", "self_check_input", main_llm=None)


def test_get_llm_fallback_chain():
    """get_self_check_llm resolves models in order: task -> default_task -> main -> ValueError."""
    task_llm = FakeLLMModel(responses=[])
    default_llm = FakeLLMModel(responses=[])
    main_llm = FakeLLMModel(responses=[])

    all_llms = {
        "check_harmful": task_llm,
        "self_check_input": default_llm,
    }

    # Level 1: exact task match
    assert get_self_check_llm(all_llms, "check_harmful", "self_check_input", main_llm=main_llm) is task_llm

    # Level 2: fall back to default task
    assert get_self_check_llm(all_llms, "check_off_topic", "self_check_input", main_llm=main_llm) is default_llm

    # Level 3: fall back to default_llm (the llm action param)
    assert get_self_check_llm({}, "check_harmful", "self_check_input", main_llm=main_llm) is main_llm

    # Level 4: no model raises ValueError
    with pytest.raises(ValueError, match="No matching model"):
        get_self_check_llm({}, "check_harmful", "self_check_input", main_llm=None)

    with pytest.raises(ValueError, match="No matching model"):
        get_self_check_llm({}, "check_inappropriate", "self_check_output", main_llm=None)
