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

from unittest.mock import patch

import pytest

from nemoguardrails.integrations.langchain.langchain_initializer import (
    ModelInitializationError,
    init_langchain_model,
)


@pytest.fixture
def mock_initializers():
    """Mock all initialization methods for unit tests."""
    with (
        patch(
            "nemoguardrails.integrations.langchain.langchain_initializer._handle_model_special_cases"
        ) as mock_special,
        patch("nemoguardrails.integrations.langchain.langchain_initializer._init_chat_completion_model") as mock_chat,
        patch(
            "nemoguardrails.integrations.langchain.langchain_initializer._init_community_chat_models"
        ) as mock_community,
    ):
        # Set __name__ attributes for the mocks
        mock_special.__name__ = "_handle_model_special_cases"
        mock_chat.__name__ = "_init_chat_completion_model"
        mock_community.__name__ = "_init_community_chat_models"

        yield {
            "special": mock_special,
            "chat": mock_chat,
            "community": mock_community,
        }


def test_special_case_called_first(mock_initializers):
    mock_initializers["special"].return_value = "special_model"
    result = init_langchain_model("any-model", "provider", {})
    assert result == "special_model"
    mock_initializers["special"].assert_called_once()
    mock_initializers["chat"].assert_not_called()
    mock_initializers["community"].assert_not_called()


def test_chat_completion_called(mock_initializers):
    mock_initializers["special"].return_value = None
    mock_initializers["chat"].return_value = "chat_model"
    result = init_langchain_model("chat-model", "provider", {})
    assert result == "chat_model"
    mock_initializers["special"].assert_called_once()
    mock_initializers["chat"].assert_called_once()
    mock_initializers["community"].assert_not_called()


def test_community_chat_called(mock_initializers):
    mock_initializers["special"].return_value = None
    mock_initializers["chat"].return_value = None
    mock_initializers["community"].return_value = "community_model"
    result = init_langchain_model("community-chat", "provider", {})
    assert result == "community_model"
    mock_initializers["special"].assert_called_once()
    mock_initializers["chat"].assert_called_once()
    mock_initializers["community"].assert_called_once()


def test_all_initializers_fail(mock_initializers):
    mock_initializers["special"].return_value = None
    mock_initializers["chat"].return_value = None
    mock_initializers["community"].return_value = None
    with pytest.raises(ModelInitializationError):
        init_langchain_model("unknown-model", "provider", {})
    mock_initializers["special"].assert_called_once()
    mock_initializers["chat"].assert_called_once()
    mock_initializers["community"].assert_called_once()


def test_missing_model_name(mock_initializers):
    with pytest.raises(ModelInitializationError, match="Model name is required for provider provider"):
        init_langchain_model(None, "provider", {})
    mock_initializers["special"].assert_not_called()
    mock_initializers["chat"].assert_not_called()
    mock_initializers["community"].assert_not_called()


def test_all_initializers_raise_exceptions(mock_initializers):
    mock_initializers["special"].side_effect = RuntimeError("Special case failed")
    mock_initializers["chat"].side_effect = ValueError("Chat model failed")
    mock_initializers["community"].side_effect = ImportError("Community model failed")
    with pytest.raises(ModelInitializationError, match=r"Failed to initialize model 'unknown-model'"):
        init_langchain_model("unknown-model", "provider", {})
    mock_initializers["special"].assert_called_once()
    mock_initializers["chat"].assert_called_once()
    mock_initializers["community"].assert_called_once()


def test_exception_not_masked_by_none_return(mock_initializers):
    """Test that an exception from an initializer is preserved when later ones return None.

    For example: if community chat throws an error (e.g., invalid API key),
    the community error should be raised.
    """
    mock_initializers["special"].return_value = None
    mock_initializers["chat"].return_value = None
    mock_initializers["community"].side_effect = ValueError("Invalid API key for provider")

    with pytest.raises(ModelInitializationError, match="Invalid API key for provider"):
        init_langchain_model("community-model", "provider", {})


def test_import_error_prioritized_over_other_exceptions(mock_initializers):
    """Test that ImportError is surfaced to help users know when packages are missing."""
    mock_initializers["special"].return_value = None
    mock_initializers["chat"].side_effect = ValueError("Some config error")
    mock_initializers["community"].side_effect = ImportError("Missing langchain_community package")

    with pytest.raises(ModelInitializationError, match="Missing langchain_community package"):
        init_langchain_model("model", "provider", {})
