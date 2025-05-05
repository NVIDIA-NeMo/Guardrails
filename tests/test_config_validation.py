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

from nemoguardrails import LLMRails, RailsConfig


def test_input_rail_exists_check():
    with pytest.raises(ValueError) as exc_info:
        config = RailsConfig.from_content(
            yaml_content="""
            rails:
                input:
                    flows:
                        - example input rail
            """,
        )
        LLMRails(config=config)

    assert "`example input rail` does not exist" in str(exc_info.value)


def test_output_rail_exists_check():
    with pytest.raises(ValueError) as exc_info:
        config = RailsConfig.from_content(
            yaml_content="""
            rails:
                output:
                    flows:
                        - example output rail
            """,
        )
        LLMRails(config=config)

    assert "`example output rail` does not exist" in str(exc_info.value)


def test_retrieval_rail_exists_check():
    with pytest.raises(ValueError) as exc_info:
        config = RailsConfig.from_content(
            yaml_content="""
            rails:
                retrieval:
                    flows:
                        - example retrieval rail
            """,
        )
        LLMRails(config=config)

    assert "`example retrieval rail` does not exist" in str(exc_info.value)


def test_self_check_input_prompt_exception():
    with pytest.raises(ValueError) as exc_info:
        config = RailsConfig.from_content(
            yaml_content="""
            rails:
                input:
                    flows:
                        - self check input
            """,
        )
        LLMRails(config=config)

    assert "You must provide a `self_check_input` prompt" in str(exc_info.value)


def test_self_check_output_prompt_exception():
    with pytest.raises(ValueError) as exc_info:
        config = RailsConfig.from_content(
            yaml_content="""
            rails:
                output:
                    flows:
                        - self check output
            """,
        )
        LLMRails(config=config)

    assert "You must provide a `self_check_output` prompt" in str(exc_info.value)


def test_passthrough_and_single_call_incompatibility():
    with pytest.raises(ValueError) as exc_info:
        config = RailsConfig.from_content(
            yaml_content="""
            rails:
                dialog:
                    single_call:
                        enabled: True
            passthrough: True
            """,
        )
        LLMRails(config=config)

    assert "The passthrough mode and the single call dialog" in str(exc_info.value)


# def test_self_check_facts_prompt_exception():
#     with pytest.raises(ValueError) as exc_info:
#         config = RailsConfig.from_content(
#             yaml_content="""
#             rails:
#                 output:
#                     flows:
#                         - self check facts
#             """,
#         )
#         LLMRails(config=config)
#
#     assert "You must provide a `self_check_facts` prompt" in str(exc_info.value)


def test_reasoning_traces_with_explicit_dialog_rails():
    """Test that reasoning traces cannot be enabled when dialog rails are explicitly configured."""

    with pytest.raises(ValueError) as exc_info:
        _ = RailsConfig.from_content(
            yaml_content="""
            models:
              - type: main
                engine: openai
                model: gpt-3.5-turbo-instruct
                reasoning_config:
                  remove_reasoning_traces: false
            rails:
              dialog:
                single_call:
                  enabled: true
            """,
        )

    assert "Main model has reasoning traces enabled in config.yml" in str(
        exc_info.value
    )
    assert "Reasoning traces must be disabled when dialog rails are present" in str(
        exc_info.value
    )
    assert (
        "Please update your config.yml to set 'remove_reasoning_traces: true' under reasoning_config"
        in str(exc_info.value)
    )


def test_reasoning_traces_without_dialog_rails():
    """Test that reasoning traces can be enabled when no dialog rails are present."""

    _ = RailsConfig.from_content(
        yaml_content="""
        models:
          - type: main
            engine: openai
            model: gpt-3.5-turbo-instruct
            reasoning_config:
              remove_reasoning_traces: false
        """,
    )


def test_dialog_rails_without_reasoning_traces():
    """Test that dialog rails can be enabled when reasoning traces are not enabled."""

    _ = RailsConfig.from_content(
        yaml_content="""
        models:
          - type: main
            engine: openai
            model: gpt-3.5-turbo-instruct
        rails:
          dialog:
            single_call:
              enabled: true
        """,
    )


def test_input_rails_only_no_dialog_rails():
    config = RailsConfig.from_content(
        yaml_content="""
        models:
          - type: main
            engine: openai
            model: gpt-3.5-turbo-instruct
            reasoning_config:
                remove_reasoning_traces: false
        rails:
          input:
            flows:
              - self check input
        prompts:
          - task: self_check_input
            content: "Check if input is safe"
        """,
    )

    assert not config.user_messages
    assert not config.bot_messages
    assert not config.flows
    assert not config.rails.dialog.single_call.enabled


def test_no_dialog_tasks_with_only_output_rails():
    """Test that dialog tasks are not used when only output rails are present."""

    config = RailsConfig.from_content(
        yaml_content="""
        models:
          - type: main
            engine: openai
            model: gpt-3.5-turbo-instruct
            reasoning_config:
                remove_reasoning_traces: false
        rails:
          output:
            flows:
              - self check output
        prompts:
          - task: self_check_output
            content: "Check if output is safe"
        """,
    )

    assert not config.user_messages
    assert not config.bot_messages
    assert not config.flows
    assert not config.rails.dialog.single_call.enabled


def test_reasoning_traces_with_implicit_dialog_rails_user_bot_messages():
    """Test that reasoning traces cannot be enabled when dialog rails are implicitly enabled thru user/bot messages."""

    with pytest.raises(ValueError) as exc_info:
        _ = RailsConfig.from_content(
            yaml_content="""
            models:
              - type: main
                engine: openai
                model: gpt-3.5-turbo-instruct
                reasoning_config:
                  remove_reasoning_traces: false
            """,
            colang_content="""
            define user express greeting
              "hello"
              "hi"

            define bot express greeting
              "Hello there!"

            define flow
              user express greeting
              bot express greeting
            """,
        )

    assert "Main model has reasoning traces enabled in config.yml" in str(
        exc_info.value
    )
    assert "Reasoning traces must be disabled when dialog rails are present" in str(
        exc_info.value
    )
    assert (
        "Please update your config.yml to set 'remove_reasoning_traces: true' under reasoning_config"
        in str(exc_info.value)
    )


def test_reasoning_traces_with_implicit_dialog_rails_flows_only():
    """Test that reasoning traces cannot be enabled when dialog rails are implicitly enabled thru flows only."""

    with pytest.raises(ValueError) as exc_info:
        _ = RailsConfig.from_content(
            yaml_content="""
            models:
              - type: main
                engine: openai
                model: gpt-3.5-turbo-instruct
                reasoning_config:
                  remove_reasoning_traces: False
            """,
            colang_content="""
            define flow
              user express greeting
              bot express greeting
            define user express greeting
                "hi"
            define bot express greeting
                "HI HI"
            """,
        )

    assert "Main model has reasoning traces enabled in config.yml" in str(
        exc_info.value
    )
    assert "Reasoning traces must be disabled when dialog rails are present" in str(
        exc_info.value
    )
    assert (
        "Please update your config.yml to set 'remove_reasoning_traces: true' under reasoning_config"
        in str(exc_info.value)
    )


def test_reasoning_traces_with_implicit_dialog_rails_user_messages_only():
    """Test that reasoning traces cannot be enabled when dialog rails are implicitly enabled through user messages (user canonical forms) only."""

    with pytest.raises(ValueError) as exc_info:
        _ = RailsConfig.from_content(
            yaml_content="""
            models:
              - type: main
                engine: openai
                model: gpt-3.5-turbo-instruct
                reasoning_config:
                  remove_reasoning_traces: false
            """,
            colang_content="""
            define user express greeting
                "hello"
                "hi"
            """,
        )

    assert "Reasoning traces must be disabled when dialog rails are present" in str(
        exc_info.value
    )


def test_reasoning_traces_with_bot_messages_only():
    """Test that reasoning traces cannot be enabled when bot messages are present."""

    with pytest.raises(ValueError) as exc_info:
        _ = RailsConfig.from_content(
            yaml_content="""
            models:
            - type: main
              engine: openai
              model: gpt-3.5-turbo-instruct
              reasoning_config:
                  remove_reasoning_traces: False
            """,
            colang_content="""
            define bot express greeting
                "Hello there!"
            """,
        )

    assert "Reasoning traces must be disabled when dialog rails are present" in str(
        exc_info.value
    )


def test_reasoning_traces_with_dedicated_task_models():
    """Test that reasoning traces cannot be enabled for dedicated task models when dialog rails are present."""

    with pytest.raises(ValueError) as exc_info:
        _ = RailsConfig.from_content(
            yaml_content="""
            models:
              - type: main
                engine: openai
                model: gpt-3.5-turbo-instruct
              - type: generate_bot_message
                engine: openai
                model: gpt-3.5-turbo-instruct
                reasoning_config:
                  remove_reasoning_traces: false
              - type: generate_user_intent
                engine: openai
                model: gpt-3.5-turbo-instruct
                reasoning_config:
                  remove_reasoning_traces: false
            rails:
              dialog:
                single_call:
                  enabled: true
            """,
        )

    assert (
        "Model 'generate_user_intent' has reasoning traces enabled in config.yml"
        in str(exc_info.value)
    )
    assert "Reasoning traces must be disabled for dialog rail tasks" in str(
        exc_info.value
    )
    assert (
        "Please update your config.yml to set 'remove_reasoning_traces: true' under reasoning_config"
        in str(exc_info.value)
    )


def test_reasoning_traces_with_mixed_task_models():
    """Test that reasoning traces cannot be enabled for any task model when dialog rails are present."""

    with pytest.raises(ValueError) as exc_info:
        _ = RailsConfig.from_content(
            yaml_content="""
            models:
              - type: main
                engine: openai
                model: gpt-3.5-turbo-instruct
                reasoning_config:
                  remove_reasoning_traces: false
              - type: generate_bot_message
                engine: openai
                model: gpt-3.5-turbo-instruct
              - type: generate_user_intent
                engine: openai
                model: gpt-3.5-turbo-instruct
                reasoning_config:
                  remove_reasoning_traces: false
            rails:
              dialog:
                single_call:
                  enabled: true
            """,
        )

    assert (
        "Model 'generate_user_intent' has reasoning traces enabled in config.yml"
        in str(exc_info.value)
    )
    assert "Reasoning traces must be disabled for dialog rail tasks" in str(
        exc_info.value
    )
    assert (
        "Please update your config.yml to set 'remove_reasoning_traces: true' under reasoning_config"
        in str(exc_info.value)
    )


def test_reasoning_traces_with_all_dialog_tasks():
    """Test that reasoning traces cannot be enabled for any dialog rail task."""

    with pytest.raises(ValueError) as exc_info:
        _ = RailsConfig.from_content(
            yaml_content="""
            models:
              - type: main
                engine: openai
                model: gpt-3.5-turbo-instruct
              - type: generate_bot_message
                engine: openai
                model: gpt-3.5-turbo-instruct
                reasoning_config:
                  remove_reasoning_traces: false
              - type: generate_user_intent
                engine: openai
                model: gpt-3.5-turbo-instruct
              - type: generate_next_steps
                engine: openai
                model: gpt-3.5-turbo-instruct
                reasoning_config:
                  remove_reasoning_traces: false
              - type: generate_intent_steps_message
                engine: openai
                model: gpt-3.5-turbo-instruct
            rails:
              dialog:
                single_call:
                  enabled: true
            """,
        )

    error_message = str(exc_info.value)
    assert (
        "Model 'generate_bot_message' has reasoning traces enabled in config.yml"
        not in error_message
    )
    assert (
        "Model 'generate_next_steps' has reasoning traces enabled in config.yml"
        in error_message
    )
    assert "Reasoning traces must be disabled for dialog rail tasks" in error_message
    assert (
        "Please update your config.yml to set 'remove_reasoning_traces: true' under reasoning_config"
        in error_message
    )


def test_reasoning_traces_with_dedicated_models_no_dialog_rails():
    """Test that reasoning traces can be enabled for dedicated models when no dialog rails are present."""

    _ = RailsConfig.from_content(
        yaml_content="""
        models:
          - type: main
            engine: openai
            model: gpt-3.5-turbo-instruct
          - type: generate_bot_message
            engine: openai
            model: gpt-3.5-turbo-instruct
            reasoning_config:
              remove_reasoning_traces: false
          - type: generate_user_intent
            engine: openai
            model: gpt-3.5-turbo-instruct
            reasoning_config:
              remove_reasoning_traces: false
        """,
    )


def test_reasoning_traces_with_implicit_dialog_rails_and_dedicated_models():
    """Test that reasoning traces cannot be enabled for dedicated models when dialog rails are implicitly enabled."""

    with pytest.raises(ValueError) as exc_info:
        _ = RailsConfig.from_content(
            yaml_content="""
            models:
              - type: main
                engine: openai
                model: gpt-3.5-turbo-instruct
              - type: generate_user_intent
                engine: openai
                model: gpt-3.5-turbo-instruct
                reasoning_config:
                  remove_reasoning_traces: false
            """,
            colang_content="""
            define user express greeting
              "hello"
              "hi"

            define bot express greeting
              "Hello there!"

            define flow
              user express greeting
              bot express greeting
            """,
        )

    assert (
        "Model 'generate_user_intent' has reasoning traces enabled in config.yml"
        in str(exc_info.value)
    )
    assert "Reasoning traces must be disabled for dialog rail tasks" in str(
        exc_info.value
    )
    assert (
        "Please update your config.yml to set 'remove_reasoning_traces: true' under reasoning_config"
        in str(exc_info.value)
    )


def test_reasoning_traces_with_partial_dedicated_models():
    """Test that reasoning traces cannot be enabled for any model when some tasks use dedicated models and others fall back to main."""

    with pytest.raises(ValueError) as exc_info:
        _ = RailsConfig.from_content(
            yaml_content="""
            models:
              - type: main
                engine: openai
                model: gpt-3.5-turbo-instruct
                reasoning_config:
                  remove_reasoning_traces: false
              - type: generate_bot_message
                engine: openai
                model: gpt-3.5-turbo-instruct
            rails:
              dialog:
                single_call:
                  enabled: true
            """,
        )

    assert "Main model has reasoning traces enabled in config.yml" in str(
        exc_info.value
    )
    assert "Reasoning traces must be disabled when dialog rails are present" in str(
        exc_info.value
    )
    assert (
        "Please update your config.yml to set 'remove_reasoning_traces: true' under reasoning_config"
        in str(exc_info.value)
    )


def test_reasoning_traces_with_implicit_dialog_rails_embeddings_only():
    """Test that reasoning traces can be enabled when embeddings_only is True, even with user messages."""

    _ = RailsConfig.from_content(
        yaml_content="""
        models:
        - type: main
          engine: openai
          model: gpt-3.5-turbo-instruct
          reasoning_config:
              remove_reasoning_traces: False
        rails:
          dialog:
            user_messages:
              embeddings_only: True
        """,
        colang_content="""
        define user express greeting
            "hello"
            "hi"
        """,
    )


def test_reasoning_traces_with_bot_messages_embeddings_only():
    """Test that reasoning traces can be enabled when embeddings_only is True, even with bot messages."""

    _ = RailsConfig.from_content(
        yaml_content="""
        models:
        - type: main
          engine: openai
          model: gpt-3.5-turbo-instruct
          reasoning_config:
              remove_reasoning_traces: False
        rails:
          dialog:
            user_messages:
              embeddings_only: True
        """,
        colang_content="""
        define bot express greeting
            "Hello there!"
        """,
    )
