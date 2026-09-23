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

from tests.utils import TestChat

BASE_CONFIG = textwrap.dedent("""
    colang_version: "2.x"
""")

OR_BOT_FLOW_COLANG = textwrap.dedent("""
    import core

    flow main
        activate greeting

    flow greeting
        user said "hi"
        bot inform about service

    flow bot inform about service
        bot say "You can ask me anything!"
            or bot say "Just ask me something!"
""")

AND_BOT_FLOW_COLANG = textwrap.dedent("""
    import core

    flow main
        activate greeting

    flow greeting
        user said "hi"
        bot express greeting

    flow bot express greeting
        bot say "Hello!"
            and bot say "Welcome!"
""")


def _make_chat(colang: str) -> TestChat:
    from nemoguardrails import RailsConfig

    config = RailsConfig.from_content(yaml_content=BASE_CONFIG, colang_content=colang)
    return TestChat(config, llm_completions=[])


def test_llmrails_init_handles_spec_or_in_bot_flow():
    chat = _make_chat(OR_BOT_FLOW_COLANG)
    assert "bot inform about service" not in chat.app._llm_generation_actions.bot_messages


def test_llmrails_init_handles_spec_and_in_bot_flow():
    chat = _make_chat(AND_BOT_FLOW_COLANG)
    assert "bot express greeting" not in chat.app._llm_generation_actions.bot_messages
