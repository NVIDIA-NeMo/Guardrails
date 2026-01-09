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

from nemoguardrails import RailsConfig
from tests.utils import TestChat

BASE_CONFIG = textwrap.dedent("""
    colang_version: "2.x"
    models:
      - type: main
        engine: openai
        model: gpt-4o
""")

PASSTHROUGH_CONFIG = textwrap.dedent("""
    colang_version: "2.x"
    models:
      - type: main
        engine: openai
        model: gpt-4o

    passthrough: true
""")

PASSTHROUGH_COLANG = textwrap.dedent("""
    import core
    import llm

    flow main
        activate passthrough mode
""")

MINIMAL_COLANG = textwrap.dedent("""
    import core
    import llm

    flow main
        activate llm continuation
""")

DIALOG_COLANG = textwrap.dedent("""
    import core
    import llm

    flow main
        activate llm continuation
        activate greeting

    flow greeting
        user expressed greeting
        bot say "Hello! How can I help you?"

    flow user expressed greeting
        \"\"\"User expressed greeting in any way or form.\"\"\"
        user said "hi"
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
        chat = _create_test_chat(BASE_CONFIG, MINIMAL_COLANG)
        actions = chat.app.llm_generation_actions

        assert actions.user_message_index is None
        assert actions.bot_message_index is None
        assert actions.flows_index is None
        assert not hasattr(actions, "instruction_flows_index") or actions.instruction_flows_index is None

    def test_passthrough(self):
        chat = _create_test_chat(PASSTHROUGH_CONFIG, PASSTHROUGH_COLANG)
        actions = chat.app.llm_generation_actions

        assert actions.user_message_index is None
        assert actions.bot_message_index is None
        assert actions.flows_index is None
        assert not hasattr(actions, "instruction_flows_index") or actions.instruction_flows_index is None

    def test_minimal_colang(self):
        chat = _create_test_chat(BASE_CONFIG, MINIMAL_COLANG)
        actions = chat.app.llm_generation_actions

        assert actions.user_message_index is None
        assert actions.bot_message_index is None
        assert actions.flows_index is None
        assert not hasattr(actions, "instruction_flows_index") or actions.instruction_flows_index is None

    def test_dialog_colang(self):
        chat = _create_test_chat(BASE_CONFIG, DIALOG_COLANG)
        actions = chat.app.llm_generation_actions

        assert actions.user_message_index is None
        assert actions.bot_message_index is None
        assert actions.flows_index is None
        assert not hasattr(actions, "instruction_flows_index") or actions.instruction_flows_index is None


class TestFastEmbedNotDownloadedForSimpleRails:
    def test_passthrough_no_cache_created(self, tmp_path):
        import os

        cache_dir = tmp_path / "fastembed_cache"
        cache_dir.mkdir()
        os.environ["FASTEMBED_CACHE_PATH"] = str(cache_dir)

        try:
            config = RailsConfig.from_content(
                yaml_content=PASSTHROUGH_CONFIG,
                colang_content=PASSTHROUGH_COLANG,
            )
            chat = TestChat(config, llm_completions=["Hello!"])

            response = chat.app.generate(messages=[{"role": "user", "content": "Hello"}])

            assert response is not None

            cache_contents = list(cache_dir.iterdir())
            assert len(cache_contents) == 0, f"FastEmbed cache should be empty but found: {cache_contents}"
        finally:
            if "FASTEMBED_CACHE_PATH" in os.environ:
                del os.environ["FASTEMBED_CACHE_PATH"]


class TestIndexInitializedAfterGenerate:
    def test_user_message_index_initialized_after_generate(self):
        config = RailsConfig.from_content(
            yaml_content=BASE_CONFIG,
            colang_content=DIALOG_COLANG,
        )
        chat = TestChat(
            config,
            llm_completions=["user expressed greeting"],
        )
        actions = chat.app.llm_generation_actions

        assert actions.user_message_index is None, "Index should be None before generate"

        chat.app.generate(messages=[{"role": "user", "content": "hello"}])

        assert actions.user_message_index is not None, "Index should be initialized after generate"
