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

from nemoguardrails import LLMRails, RailsConfig
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


def _create_rails(yaml_content: str, colang_content: str = ""):
    config = RailsConfig.from_content(
        yaml_content=yaml_content,
        colang_content=colang_content if colang_content else None,
    )
    return LLMRails(config)


class TestEmbeddingIndexesNotCreatedAtInit:
    def test_main_model_only(self):
        rails = _create_rails(BASE_CONFIG)
        actions = rails.llm_generation_actions

        assert actions.user_message_index is None
        assert actions.bot_message_index is None
        assert actions.flows_index is None

    def test_input_rails_only(self):
        rails = _create_rails(BASE_CONFIG + INPUT_RAILS_CONFIG)
        actions = rails.llm_generation_actions

        assert actions.user_message_index is None
        assert actions.bot_message_index is None
        assert actions.flows_index is None

    def test_output_rails_only(self):
        rails = _create_rails(BASE_CONFIG + OUTPUT_RAILS_CONFIG)
        actions = rails.llm_generation_actions

        assert actions.user_message_index is None
        assert actions.bot_message_index is None
        assert actions.flows_index is None

    def test_input_output_rails(self):
        rails = _create_rails(BASE_CONFIG + INPUT_OUTPUT_RAILS_CONFIG)
        actions = rails.llm_generation_actions

        assert actions.user_message_index is None
        assert actions.bot_message_index is None
        assert actions.flows_index is None

    def test_passthrough(self):
        rails = _create_rails(PASSTHROUGH_CONFIG)
        actions = rails.llm_generation_actions

        assert actions.user_message_index is None
        assert actions.bot_message_index is None
        assert actions.flows_index is None

    def test_user_definitions_only(self):
        rails = _create_rails(BASE_CONFIG, USER_DEFINITIONS)
        actions = rails.llm_generation_actions

        assert actions.user_message_index is None
        assert actions.bot_message_index is None
        assert actions.flows_index is None

    def test_bot_definitions_only(self):
        rails = _create_rails(BASE_CONFIG, BOT_DEFINITIONS)
        actions = rails.llm_generation_actions

        assert actions.user_message_index is None
        assert actions.bot_message_index is None
        assert actions.flows_index is None

    def test_user_and_bot_definitions(self):
        rails = _create_rails(BASE_CONFIG, USER_DEFINITIONS + BOT_DEFINITIONS)
        actions = rails.llm_generation_actions

        assert actions.user_message_index is None
        assert actions.bot_message_index is None
        assert actions.flows_index is None

    def test_flow_definitions_only(self):
        rails = _create_rails(BASE_CONFIG, FLOW_DEFINITIONS)
        actions = rails.llm_generation_actions

        assert actions.user_message_index is None
        assert actions.bot_message_index is None
        assert actions.flows_index is None

    def test_full_dialog_rails(self):
        rails = _create_rails(BASE_CONFIG, USER_DEFINITIONS + BOT_DEFINITIONS + FLOW_DEFINITIONS)
        actions = rails.llm_generation_actions

        assert actions.user_message_index is None
        assert actions.bot_message_index is None
        assert actions.flows_index is None

    def test_input_rails_with_user_definitions(self):
        rails = _create_rails(BASE_CONFIG + INPUT_RAILS_CONFIG, USER_DEFINITIONS)
        actions = rails.llm_generation_actions

        assert actions.user_message_index is None
        assert actions.bot_message_index is None
        assert actions.flows_index is None


class TestConfigDataPresent:
    def test_user_messages_present_in_config(self):
        rails = _create_rails(BASE_CONFIG, USER_DEFINITIONS)
        assert len(rails.config.user_messages) == 2

    def test_bot_messages_include_library_defaults(self):
        rails = _create_rails(BASE_CONFIG)
        assert len(rails.config.bot_messages) >= 9

    def test_non_system_flows_counted_correctly(self):
        rails = _create_rails(BASE_CONFIG, FLOW_DEFINITIONS)
        non_system = [f for f in rails.config.flows if not f.get("is_system_flow", False)]
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


class TestFastEmbedDownloadedForDialogRails:
    def test_dialog_rails_cache_created_on_generate(self, tmp_path):
        import os

        cache_dir = tmp_path / "fastembed_cache"
        cache_dir.mkdir()
        os.environ["FASTEMBED_CACHE_PATH"] = str(cache_dir)

        try:
            config = RailsConfig.from_content(
                yaml_content=BASE_CONFIG,
                colang_content=USER_DEFINITIONS + BOT_DEFINITIONS + FLOW_DEFINITIONS,
            )
            chat = TestChat(
                config,
                llm_completions=["express greeting", "Hello! How can I help you?"],
            )

            cache_before = list(cache_dir.iterdir())
            assert len(cache_before) == 0, "Cache should be empty before generate"

            response = chat.app.generate(messages=[{"role": "user", "content": "hello"}])

            assert response is not None

            cache_after = list(cache_dir.iterdir())
            assert len(cache_after) > 0, "FastEmbed cache should have models after generate with dialog rails"
        finally:
            if "FASTEMBED_CACHE_PATH" in os.environ:
                del os.environ["FASTEMBED_CACHE_PATH"]
