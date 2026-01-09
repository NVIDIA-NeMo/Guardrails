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

import textwrap

import pytest

from nemoguardrails import RailsConfig
from tests.utils import TestChat

BASE_CONFIG = textwrap.dedent("""
    models:
      - type: main
        engine: openai
        model: gpt-4o
""")

INPUT_RAILS_CONFIG = textwrap.dedent("""
    rails:
      input:
        flows:
          - self check input

    prompts:
      - task: self_check_input
        content: |
          Instruction: Check if the user input is safe.
          User input: {{ user_input }}
          Answer [yes/no]:
""")

OUTPUT_RAILS_CONFIG = textwrap.dedent("""
    rails:
      output:
        flows:
          - self check output

    prompts:
      - task: self_check_output
        content: |
          Instruction: Check if the bot output is safe.
          Bot output: {{ bot_response }}
          Answer [yes/no]:
""")

INPUT_OUTPUT_RAILS_CONFIG = textwrap.dedent("""
    rails:
      input:
        flows:
          - self check input
      output:
        flows:
          - self check output

    prompts:
      - task: self_check_input
        content: |
          Instruction: Check if the user input is safe.
          User input: {{ user_input }}
          Answer [yes/no]:
      - task: self_check_output
        content: |
          Instruction: Check if the bot output is safe.
          Bot output: {{ bot_response }}
          Answer [yes/no]:
""")

PASSTHROUGH_CONFIG = textwrap.dedent("""
    models:
      - type: main
        engine: openai
        model: gpt-4o

    passthrough: true
""")

USER_DEFINITIONS = textwrap.dedent("""
    define user express greeting
      "hello"
      "hi there"

    define user ask about weather
      "what is the weather"
      "how is the weather today"
""")

BOT_DEFINITIONS = textwrap.dedent("""
    define bot express greeting
      "Hello! How can I help you?"

    define bot inform weather
      "The weather is nice today."
""")

FLOW_DEFINITIONS = textwrap.dedent("""
    define flow greeting
      user express greeting
      bot express greeting

    define flow weather
      user ask about weather
      bot inform weather
""")


def _create_test_chat(yaml_content: str, colang_content: str = "", llm_completions=None):
    if llm_completions is None:
        llm_completions = ["Hello!"]
    config = RailsConfig.from_content(
        yaml_content=yaml_content,
        colang_content=colang_content if colang_content else None,
    )
    return TestChat(config, llm_completions=llm_completions)


class TestEmbeddingIndexesNotCreatedAtInit:
    def test_main_model_only(self):
        chat = _create_test_chat(BASE_CONFIG)
        actions = chat.app.llm_generation_actions

        assert actions.user_message_index is None
        assert actions.bot_message_index is None
        assert actions.flows_index is None

    def test_input_rails_only(self):
        chat = _create_test_chat(BASE_CONFIG + INPUT_RAILS_CONFIG, llm_completions=["yes", "Hello!"])
        actions = chat.app.llm_generation_actions

        assert actions.user_message_index is None
        assert actions.bot_message_index is None
        assert actions.flows_index is None

    def test_output_rails_only(self):
        chat = _create_test_chat(BASE_CONFIG + OUTPUT_RAILS_CONFIG, llm_completions=["Hello!", "yes"])
        actions = chat.app.llm_generation_actions

        assert actions.user_message_index is None
        assert actions.bot_message_index is None
        assert actions.flows_index is None

    def test_input_output_rails(self):
        chat = _create_test_chat(BASE_CONFIG + INPUT_OUTPUT_RAILS_CONFIG, llm_completions=["yes", "Hello!", "yes"])
        actions = chat.app.llm_generation_actions

        assert actions.user_message_index is None
        assert actions.bot_message_index is None
        assert actions.flows_index is None

    def test_passthrough(self):
        chat = _create_test_chat(PASSTHROUGH_CONFIG)
        actions = chat.app.llm_generation_actions

        assert actions.user_message_index is None
        assert actions.bot_message_index is None
        assert actions.flows_index is None

    def test_user_definitions_only(self):
        chat = _create_test_chat(BASE_CONFIG, USER_DEFINITIONS)
        actions = chat.app.llm_generation_actions

        assert actions.user_message_index is None
        assert actions.bot_message_index is None
        assert actions.flows_index is None

    def test_bot_definitions_only(self):
        chat = _create_test_chat(BASE_CONFIG, BOT_DEFINITIONS)
        actions = chat.app.llm_generation_actions

        assert actions.user_message_index is None
        assert actions.bot_message_index is None
        assert actions.flows_index is None

    def test_user_and_bot_definitions(self):
        chat = _create_test_chat(BASE_CONFIG, USER_DEFINITIONS + BOT_DEFINITIONS)
        actions = chat.app.llm_generation_actions

        assert actions.user_message_index is None
        assert actions.bot_message_index is None
        assert actions.flows_index is None

    def test_flow_definitions_only(self):
        chat = _create_test_chat(BASE_CONFIG, FLOW_DEFINITIONS)
        actions = chat.app.llm_generation_actions

        assert actions.user_message_index is None
        assert actions.bot_message_index is None
        assert actions.flows_index is None

    def test_full_dialog_rails(self):
        chat = _create_test_chat(BASE_CONFIG, USER_DEFINITIONS + BOT_DEFINITIONS + FLOW_DEFINITIONS)
        actions = chat.app.llm_generation_actions

        assert actions.user_message_index is None
        assert actions.bot_message_index is None
        assert actions.flows_index is None

    def test_input_rails_with_user_definitions(self):
        chat = _create_test_chat(BASE_CONFIG + INPUT_RAILS_CONFIG, USER_DEFINITIONS, llm_completions=["yes", "Hello!"])
        actions = chat.app.llm_generation_actions

        assert actions.user_message_index is None
        assert actions.bot_message_index is None
        assert actions.flows_index is None


class TestConfigDataPresent:
    def test_user_messages_present_in_config(self):
        chat = _create_test_chat(BASE_CONFIG, USER_DEFINITIONS)
        assert len(chat.app.config.user_messages) == 2

    def test_bot_messages_include_library_defaults(self):
        chat = _create_test_chat(BASE_CONFIG)
        assert len(chat.app.config.bot_messages) >= 9

    def test_non_system_flows_counted_correctly(self):
        chat = _create_test_chat(BASE_CONFIG, FLOW_DEFINITIONS)
        non_system = [f for f in chat.app.config.flows if not f.get("is_system_flow", False)]
        assert len(non_system) == 2


class TestFastEmbedNotDownloadedForSimpleRails:
    def test_input_rails_no_cache_created(self, tmp_path):
        import os

        cache_dir = tmp_path / "fastembed_cache"
        cache_dir.mkdir()
        os.environ["FASTEMBED_CACHE_PATH"] = str(cache_dir)

        try:
            config = RailsConfig.from_content(yaml_content=BASE_CONFIG + INPUT_RAILS_CONFIG)
            chat = TestChat(config, llm_completions=["yes", "Hello!"])

            response = chat.app.generate(messages=[{"role": "user", "content": "Hello"}])

            assert response is not None

            cache_contents = list(cache_dir.iterdir())
            assert len(cache_contents) == 0, f"FastEmbed cache should be empty but found: {cache_contents}"
        finally:
            if "FASTEMBED_CACHE_PATH" in os.environ:
                del os.environ["FASTEMBED_CACHE_PATH"]

    def test_output_rails_no_cache_created(self, tmp_path):
        import os

        cache_dir = tmp_path / "fastembed_cache"
        cache_dir.mkdir()
        os.environ["FASTEMBED_CACHE_PATH"] = str(cache_dir)

        try:
            config = RailsConfig.from_content(yaml_content=BASE_CONFIG + OUTPUT_RAILS_CONFIG)
            chat = TestChat(config, llm_completions=["Hello!", "yes"])

            response = chat.app.generate(messages=[{"role": "user", "content": "Hello"}])

            assert response is not None

            cache_contents = list(cache_dir.iterdir())
            assert len(cache_contents) == 0, f"FastEmbed cache should be empty but found: {cache_contents}"
        finally:
            if "FASTEMBED_CACHE_PATH" in os.environ:
                del os.environ["FASTEMBED_CACHE_PATH"]

    def test_passthrough_no_cache_created(self, tmp_path):
        import os

        cache_dir = tmp_path / "fastembed_cache"
        cache_dir.mkdir()
        os.environ["FASTEMBED_CACHE_PATH"] = str(cache_dir)

        try:
            config = RailsConfig.from_content(yaml_content=PASSTHROUGH_CONFIG)
            chat = TestChat(config, llm_completions=["Hello!"])

            response = chat.app.generate(messages=[{"role": "user", "content": "Hello"}])

            assert response is not None
            assert response["content"] == "Hello!"

            cache_contents = list(cache_dir.iterdir())
            assert len(cache_contents) == 0, f"FastEmbed cache should be empty but found: {cache_contents}"
        finally:
            if "FASTEMBED_CACHE_PATH" in os.environ:
                del os.environ["FASTEMBED_CACHE_PATH"]


class TestIndexInitializedAfterGenerate:
    def test_user_message_index_initialized_after_generate(self):
        config = RailsConfig.from_content(
            yaml_content=BASE_CONFIG,
            colang_content=USER_DEFINITIONS + BOT_DEFINITIONS + FLOW_DEFINITIONS,
        )
        chat = TestChat(
            config,
            llm_completions=["express greeting", "Hello! How can I help you?"],
        )
        actions = chat.app.llm_generation_actions

        assert actions.user_message_index is None, "Index should be None before generate"

        chat.app.generate(messages=[{"role": "user", "content": "hello"}])

        assert actions.user_message_index is not None, "Index should be initialized after generate"


class TestConcurrentInitialization:
    @pytest.mark.asyncio
    async def test_concurrent_ensure_user_message_index_calls_init_once(self):
        import asyncio

        config = RailsConfig.from_content(
            yaml_content=BASE_CONFIG,
            colang_content=USER_DEFINITIONS,
        )
        chat = TestChat(config, llm_completions=["Hello!"])
        actions = chat.app.llm_generation_actions

        init_call_count = 0

        async def counting_init():
            nonlocal init_call_count
            init_call_count += 1
            await asyncio.sleep(0.05)
            actions.user_message_index = "initialized"

        actions._init_user_message_index = counting_init

        tasks = [actions._ensure_user_message_index() for _ in range(10)]
        await asyncio.gather(*tasks)

        assert init_call_count == 1, f"Expected 1 init call, got {init_call_count}"

    @pytest.mark.asyncio
    async def test_concurrent_ensure_flows_index_calls_init_once(self):
        import asyncio

        config = RailsConfig.from_content(
            yaml_content=BASE_CONFIG,
            colang_content=USER_DEFINITIONS + BOT_DEFINITIONS + FLOW_DEFINITIONS,
        )
        chat = TestChat(config, llm_completions=["Hello!"])
        actions = chat.app.llm_generation_actions

        init_call_count = 0

        async def counting_init():
            nonlocal init_call_count
            init_call_count += 1
            await asyncio.sleep(0.05)
            actions.flows_index = "initialized"

        actions._init_flows_index = counting_init

        tasks = [actions._ensure_flows_index() for _ in range(10)]
        await asyncio.gather(*tasks)

        assert init_call_count == 1, f"Expected 1 init call, got {init_call_count}"
