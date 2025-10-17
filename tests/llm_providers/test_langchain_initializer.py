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

from unittest.mock import MagicMock, patch

import pytest

from nemoguardrails.llm.models.langchain_initializer import (
    ModelInitializationError,
    _handle_model_special_cases,
    _init_chat_completion_model,
    _init_community_chat_models,
    _init_text_completion_model,
    init_langchain_model,
)


@pytest.fixture
def mock_initializers():
    """Mock all initialization methods for unit tests."""
    with (
        patch(
            "nemoguardrails.llm.models.langchain_initializer._handle_model_special_cases"
        ) as mock_special,
        patch(
            "nemoguardrails.llm.models.langchain_initializer._init_chat_completion_model"
        ) as mock_chat,
        patch(
            "nemoguardrails.llm.models.langchain_initializer._init_community_chat_models"
        ) as mock_community,
        patch(
            "nemoguardrails.llm.models.langchain_initializer._init_text_completion_model"
        ) as mock_text,
    ):
        # Set __name__ attributes for the mocks
        mock_special.__name__ = "_handle_model_special_cases"
        mock_chat.__name__ = "_init_chat_completion_model"
        mock_community.__name__ = "_init_community_chat_models"
        mock_text.__name__ = "_init_text_completion_model"

        yield {
            "special": mock_special,
            "chat": mock_chat,
            "community": mock_community,
            "text": mock_text,
        }


def test_special_case_called_first(mock_initializers):
    mock_initializers["special"].return_value = "special_model"
    result = init_langchain_model("gpt-3.5-turbo-instruct", "provider", "chat", {})
    assert result == "special_model"
    mock_initializers["special"].assert_called_once()
    mock_initializers["chat"].assert_not_called()
    mock_initializers["community"].assert_not_called()
    mock_initializers["text"].assert_not_called()


def test_chat_completion_called(mock_initializers):
    mock_initializers["special"].return_value = None
    mock_initializers["chat"].return_value = "chat_model"
    result = init_langchain_model("chat-model", "provider", "chat", {})
    assert result == "chat_model"
    mock_initializers["special"].assert_called_once()
    mock_initializers["chat"].assert_called_once()
    mock_initializers["community"].assert_not_called()
    mock_initializers["text"].assert_not_called()


def test_community_chat_called(mock_initializers):
    mock_initializers["special"].return_value = None
    mock_initializers["chat"].return_value = None
    mock_initializers["community"].return_value = "community_model"
    result = init_langchain_model("community-chat", "provider", "chat", {})
    assert result == "community_model"
    mock_initializers["special"].assert_called_once()
    mock_initializers["chat"].assert_called_once()
    mock_initializers["community"].assert_called_once()
    mock_initializers["text"].assert_not_called()


def test_text_completion_called(mock_initializers):
    mock_initializers["special"].return_value = None
    mock_initializers["chat"].return_value = None
    mock_initializers["community"].return_value = None
    mock_initializers["text"].return_value = "text_model"
    result = init_langchain_model("text-model", "provider", "text", {})
    assert result == "text_model"
    mock_initializers["special"].assert_called_once()
    mock_initializers["chat"].assert_not_called()
    mock_initializers["community"].assert_not_called()
    mock_initializers["text"].assert_called_once()


def test_all_initializers_fail(mock_initializers):
    mock_initializers["special"].return_value = None
    mock_initializers["chat"].return_value = None
    mock_initializers["community"].return_value = None
    mock_initializers["text"].return_value = None
    with pytest.raises(ModelInitializationError):
        init_langchain_model("unknown-model", "provider", "chat", {})
    mock_initializers["special"].assert_called_once()
    mock_initializers["chat"].assert_called_once()
    mock_initializers["community"].assert_called_once()
    mock_initializers["text"].assert_called_once()


def test_unsupported_mode(mock_initializers):
    with pytest.raises(ValueError, match="Unsupported mode: invalid_mode"):
        init_langchain_model("text-model", "provider", "invalid_mode", {})
    mock_initializers["special"].assert_not_called()
    mock_initializers["chat"].assert_not_called()
    mock_initializers["community"].assert_not_called()
    mock_initializers["text"].assert_not_called()


def test_missing_model_name(mock_initializers):
    with pytest.raises(
        ModelInitializationError, match="Model name is required for provider provider"
    ):
        init_langchain_model(None, "provider", "chat", {})
    mock_initializers["special"].assert_not_called()
    mock_initializers["chat"].assert_not_called()
    mock_initializers["community"].assert_not_called()
    mock_initializers["text"].assert_not_called()


def test_all_initializers_raise_exceptions(mock_initializers):
    mock_initializers["special"].side_effect = RuntimeError("Special case failed")
    mock_initializers["chat"].side_effect = ValueError("Chat model failed")
    mock_initializers["community"].side_effect = ImportError("Community model failed")
    mock_initializers["text"].side_effect = KeyError("Text model failed")
    with pytest.raises(
        ModelInitializationError, match=r"Failed to initialize model 'unknown-model'"
    ):
        init_langchain_model("unknown-model", "provider", "chat", {})
    mock_initializers["special"].assert_called_once()
    mock_initializers["chat"].assert_called_once()
    mock_initializers["community"].assert_called_once()
    mock_initializers["text"].assert_called_once()


def test_duplicate_modes_in_initializer(mock_initializers):
    mock_initializers["special"].return_value = None
    mock_initializers["chat"].return_value = "chat_model"
    result = init_langchain_model("chat-model", "provider", "chat", {})
    assert result == "chat_model"
    mock_initializers["special"].assert_called_once()
    mock_initializers["chat"].assert_called_once()
    mock_initializers["community"].assert_not_called()
    mock_initializers["text"].assert_not_called()


def test_chat_completion_called_when_special_returns_none(mock_initializers):
    mock_initializers["special"].return_value = None
    mock_initializers["chat"].return_value = "chat_model"
    result = init_langchain_model("chat-model", "provider", "chat", {})
    assert result == "chat_model"
    mock_initializers["special"].assert_called_once()
    mock_initializers["chat"].assert_called_once()
    mock_initializers["community"].assert_not_called()
    mock_initializers["text"].assert_not_called()


def test_community_chat_called_when_previous_fail(mock_initializers):
    mock_initializers["special"].return_value = None
    mock_initializers["chat"].return_value = None
    mock_initializers["community"].return_value = "community_model"
    result = init_langchain_model("community-chat", "provider", "chat", {})
    assert result == "community_model"
    mock_initializers["special"].assert_called_once()
    mock_initializers["chat"].assert_called_once()
    mock_initializers["community"].assert_called_once()
    mock_initializers["text"].assert_not_called()


def test_text_completion_called_when_previous_fail(mock_initializers):
    mock_initializers["special"].return_value = None
    mock_initializers["chat"].return_value = None
    mock_initializers["community"].return_value = None
    mock_initializers["text"].return_value = "text_model"
    result = init_langchain_model("text-model", "provider", "text", {})
    assert result == "text_model"
    mock_initializers["special"].assert_called_once()
    mock_initializers["chat"].assert_not_called()
    mock_initializers["community"].assert_not_called()
    mock_initializers["text"].assert_called_once()


def test_text_completion_supports_chat_mode(mock_initializers):
    mock_initializers["special"].return_value = None
    mock_initializers["chat"].return_value = None
    mock_initializers["community"].return_value = None
    mock_initializers["text"].return_value = "text_model"
    result = init_langchain_model("text-model", "provider", "chat", {})
    assert result == "text_model"
    mock_initializers["special"].assert_called_once()
    mock_initializers["chat"].assert_called_once()
    mock_initializers["community"].assert_called_once()
    mock_initializers["text"].assert_called_once()
