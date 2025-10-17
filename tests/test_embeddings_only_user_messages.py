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

from unittest.mock import MagicMock

import pytest

from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.actions.llm.utils import LLMCallException
from nemoguardrails.llm.filters import colang
from tests.utils import TestChat


@pytest.fixture
def colang_1_config():
    return RailsConfig.from_content(
        """
        define user express greeting
          "hi"

        define bot express greeting
          "Hello!"

        define flow
          user express greeting
          bot express greeting
        """,
        """
        rails:
            dialog:
                user_messages:
                    embeddings_only: True
                    embeddings_only_similarity_threshold: 0.8
                    embeddings_only_fallback_intent: "express greeting"
        """,
    )


@pytest.fixture
def colang_2_config():
    return RailsConfig.from_content(
        """
    import core
    import llm

    flow main
        activate greeting
        activate llm continuation

    flow user expressed greeting
        user said "hi"
        or user said "hello"

    flow bot express greeting
        bot say "Hello!"

    flow greeting
        user expressed greeting
        bot express greeting
    """,
        """
    colang_version: "2.x"
    rails:
        dialog:
            user_messages:
                embeddings_only: True
                embeddings_only_similarity_threshold: 0.8
                embeddings_only_fallback_intent: "user expressed greeting"
    """,
    )


def test_greeting_1(colang_1_config):
    """Test that the bot responds with 'Hello!' when the user says 'hello'."""

    chat = TestChat(
        colang_1_config,
        llm_completions=[],
    )

    chat >> "hello"
    chat << "Hello!"


def test_greeting_2(colang_2_config):
    """Test that the bot responds with 'Hello!' when the user says 'hello'."""

    chat = TestChat(
        colang_2_config,
        llm_completions=[],
    )

    chat >> "hi"
    chat << "Hello!"


def test_error_when_embeddings_only_is_false(colang_1_config):
    """Test that an error is raised when the 'embeddings_only' option is False."""

    # Check that if we deactivate the embeddings_only option we get an error
    colang_1_config.rails.dialog.user_messages.embeddings_only = False
    chat = TestChat(
        colang_1_config,
        llm_completions=[],
    )

    with pytest.raises(LLMCallException):
        chat >> "how is your day?"
        chat << "Hello!"


def test_error_when_embeddings_only_is_false_2(colang_2_config):
    """Test that an error is raised when the 'embeddings_only' option is False."""

    # Check that if we deactivate the embeddings_only option we get an error
    colang_2_config.rails.dialog.user_messages.embeddings_only = False
    chat = TestChat(
        colang_2_config,
        llm_completions=[],
    )

    with pytest.raises(LLMCallException):
        chat >> "how is your day?"
        chat << "Hello!"


def test_fallback_intent(colang_1_config):
    """Test that the bot uses the fallback intent when it doesn't recognize the user's message."""

    rails = LLMRails(colang_1_config)
    res = rails.generate(messages=[{"role": "user", "content": "lets use fallback"}])
    assert res["content"] == "Hello!"

    colang_1_config.rails.dialog.user_messages.embeddings_only_fallback_intent = None
    rails = LLMRails(colang_1_config)
    with pytest.raises(LLMCallException):
        rails.generate(messages=[{"role": "user", "content": "lets use fallback"}])


def test_fallback_intent_2(colang_2_config):
    """Test that the bot uses the fallback intent when it doesn't recognize the user's message."""

    rails = LLMRails(colang_2_config)
    res = rails.generate(messages=[{"role": "user", "content": "lets use fallback"}])
    assert res["content"] == "Hello!"

    colang_2_config.rails.dialog.user_messages.embeddings_only_fallback_intent = None
    rails = LLMRails(colang_2_config)
    with pytest.raises(LLMCallException):
        rails.generate(messages=[{"role": "user", "content": "lets use fallback"}])


def test_examples_included_in_prompts(colang_1_config):
    colang_1_config.rails.dialog.user_messages.embeddings_only_fallback_intent = None
    chat = TestChat(
        colang_1_config,
        llm_completions=[
            " user express greeting",
            ' bot respond to aditional context\nbot action: "Hello is there anything else" ',
        ],
    )

    rails = chat.app

    messages = [
        {"role": "user", "content": "Hi!"},
    ]

    rails.generate(messages=messages)

    info = rails.explain()
    assert len(info.llm_calls) == 1
    assert 'user "hi"' in info.llm_calls[0].prompt


def test_examples_included_in_prompts_2(colang_2_config):
    colang_2_config.rails.dialog.user_messages.embeddings_only_fallback_intent = None
    chat = TestChat(
        colang_2_config,
        llm_completions=[
            " user express greeting",
            ' bot respond to uknown intent "Hello is there anything else" ',
        ],
    )

    rails = chat.app

    messages = [
        {"role": "user", "content": "Hi"},
    ]

    rails.generate(messages=messages)

    info = rails.explain()
    assert len(info.llm_calls) == 2
    assert 'user said "Hi"' in info.llm_calls[0].prompt


def test_no_llm_calls_embedding_only(colang_2_config):
    colang_2_config.rails.dialog.user_messages.embeddings_only_fallback_intent = None
    chat = TestChat(
        colang_2_config,
        llm_completions=[
            " user express greeting",
            ' bot respond to uknown intent "Hello is there anything else" ',
        ],
    )

    rails = chat.app

    messages = [
        {"role": "user", "content": "hi"},
    ]

    new_message = rails.generate(messages=messages)

    assert new_message["content"] == "Hello!"

    assert rails.explain_info.llm_calls == []
