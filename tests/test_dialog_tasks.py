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

import os

import pytest

from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.imports import check_optional_dependency

has_langchain_openai = check_optional_dependency("langchain_openai")

has_openai_key = bool(os.getenv("OPENAI_API_KEY"))

skip_if_no_openai = pytest.mark.skipif(
    not (has_langchain_openai and has_openai_key),
    reason="Requires langchain_openai and OPENAI_API_KEY environment variable",
)


@skip_if_no_openai
def test_dialog_tasks_with_only_input_rails():
    """Test that dialog tasks are not used when only input rails are present."""

    config = RailsConfig.from_content(
        yaml_content="""
        models:
          - type: main
            engine: openai
            model: gpt-3.5-turbo-instruct
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

    rails = LLMRails(config=config)

    assert not rails.config.rails.dialog.single_call.enabled
    # with just input rails, some basic flows and messages are created
    # but they are not for actual dialog processing
    assert rails.config.bot_messages
    assert rails.config.flows
    # even there should be no user messages defined
    assert not rails.config.user_messages


@skip_if_no_openai
def test_dialog_tasks_with_only_output_rails():
    """Test that dialog tasks are not used when only output rails are present."""

    config = RailsConfig.from_content(
        yaml_content="""
        models:
          - type: main
            engine: openai
            model: gpt-3.5-turbo-instruct
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

    rails = LLMRails(config=config)

    assert not rails.config.rails.dialog.single_call.enabled
    assert rails.config.bot_messages
    assert rails.config.flows
    assert not rails.config.user_messages


@skip_if_no_openai
def test_dialog_tasks_with_dialog_rails():
    """Test that dialog tasks are used when dialog rails are present."""

    config = RailsConfig.from_content(
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

    assert config.user_messages
    assert config.bot_messages
    assert config.flows
    assert config.rails.dialog.single_call.enabled

    rails = LLMRails(config=config)

    assert rails.config.rails.dialog.single_call.enabled
    assert rails.config.user_messages
    assert rails.config.bot_messages
    assert rails.config.flows


@skip_if_no_openai
def test_dialog_tasks_with_implicit_dialog_rails():
    """Test that dialog tasks are used when dialog rails are implicitly present through user/bot messages."""

    config = RailsConfig.from_content(
        yaml_content="""
        models:
          - type: main
            engine: openai
            model: gpt-3.5-turbo-instruct
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

    assert config.user_messages
    assert config.bot_messages
    assert config.flows

    assert config.user_messages == {"express greeting": ["hello", "hi"]}
    assert config.bot_messages == {"express greeting": ["Hello there!"]}
    assert len(config.bot_messages) == 1
    assert len(config.flows) == 1

    assert not config.rails.dialog.single_call.enabled

    rails = LLMRails(config=config)

    assert rails.config.user_messages
    assert len(rails.config.user_messages) == 1
    assert rails.config.bot_messages
    assert len(rails.config.bot_messages) > 1
    assert rails.config.flows


@skip_if_no_openai
def test_dialog_tasks_with_mixed_rails():
    """Test that dialog tasks are used when dialog rails are present along with other rails."""

    config = RailsConfig.from_content(
        yaml_content="""
        models:
          - type: main
            engine: openai
            model: gpt-3.5-turbo-instruct
        rails:
          input:
            flows:
              - self check input
          output:
            flows:
              - self check output
          dialog:
            single_call:
              enabled: true
        prompts:
          - task: self_check_input
            content: "Check if input is safe"
          - task: self_check_output
            content: "Check if output is safe"
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
    assert config.rails.dialog.single_call.enabled
    assert config.user_messages
    assert config.bot_messages
    assert config.flows
    assert config.rails.input.flows
    assert config.rails.output.flows

    rails = LLMRails(config=config)

    assert rails.config.rails.dialog.single_call.enabled
    assert rails.config.user_messages
    assert rails.config.bot_messages
    assert rails.config.flows
    assert rails.config.rails.input.flows
    assert rails.config.rails.output.flows
