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
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.language_models import BaseChatModel

from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.kb.kb import KnowledgeBase
from tests.utils import TestChat

RAILS_CONFIG = RailsConfig.from_content(
    """
import llm
import core

flow main
    activate llm continuation

flow user express greeting
   user said "hello"
   or user said "hi"
   or user said "how are you"

flow bot express greeting
   bot say "Hey!"

flow greeting
    user express greeting
    bot express greeting
""",
    yaml_content="""
    colang_version: 2.x
    models: []
    """,
)


def test_relevant_chunk_inserted_in_prompt():
    mock_kb = MagicMock(spec=KnowledgeBase)

    mock_kb.search_relevant_chunks.return_value = [
        {"title": "Test Title", "body": "Test Body"}
    ]

    chat = TestChat(
        RAILS_CONFIG,
        llm_completions=[
            " user express greeting",
            ' bot respond to aditional context\nbot action: "Hello is there anything else" ',
        ],
    )

    rails = chat.app

    rails.runtime.register_action_param("kb", mock_kb)

    messages = [
        {"role": "user", "content": "Hi!"},
    ]

    before_llm_calls = len(rails.explain().llm_calls)
    _ = rails.generate(messages=messages)
    after_llm_calls = len(rails.explain().llm_calls)
    llm_call_count = after_llm_calls - before_llm_calls

    info = rails.explain()
    assert llm_call_count == 2
    assert "Test Body" in info.llm_calls[-1].prompt
    assert "markdown" in info.llm_calls[-1].prompt
    assert "context" in info.llm_calls[-1].prompt


def test_relevant_chunk_inserted_in_prompt_no_kb():
    chat = TestChat(
        RAILS_CONFIG,
        llm_completions=[
            " user express greeting",
            ' bot respond to aditional context\nbot action: "Hello is there anything else" ',
        ],
    )
    rails = chat.app
    messages = [
        {"role": "user", "content": "Hi!"},
    ]

    before_llm_calls = len(rails.explain().llm_calls)
    _ = rails.generate(messages=messages)
    after_llm_calls = len(rails.explain().llm_calls)
    llm_call_count = after_llm_calls - before_llm_calls

    info = rails.explain()
    assert llm_call_count == 2
    assert "markdown" not in info.llm_calls[1].prompt
    assert "context" not in info.llm_calls[1].prompt
