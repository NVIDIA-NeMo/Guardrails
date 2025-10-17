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

"""
Tests for the initialization methods for different model types.

This module contains tests for the initialization methods that are used to initialize
different types of models (chat completion, community chat, text completion).
"""

from unittest.mock import MagicMock, patch

import pytest

from nemoguardrails.llm.models.langchain_initializer import (
    _init_chat_completion_model,
    _init_community_chat_models,
    _init_text_completion_model,
    _update_model_kwargs,
)


class TestChatCompletionInitializer:
    """Tests for the chat completion initializer."""

    def test_init_chat_completion_model_success(self):
        with patch(
            "nemoguardrails.llm.models.langchain_initializer.init_chat_model"
        ) as mock_init:
            mock_init.return_value = "chat_model"
            with patch(
                "nemoguardrails.llm.models.langchain_initializer.version"
            ) as mock_version:
                mock_version.return_value = "0.2.7"
                result = _init_chat_completion_model("gpt-3.5-turbo", "openai", {})
                assert result == "chat_model"
                mock_init.assert_called_once_with(
                    model="gpt-3.5-turbo",
                    model_provider="openai",
                )

    def test_init_chat_completion_model_with_api_key_success(self):
        with patch(
            "nemoguardrails.llm.models.langchain_initializer.init_chat_model"
        ) as mock_init:
            mock_init.return_value = "chat_model"
            with patch(
                "nemoguardrails.llm.models.langchain_initializer.version"
            ) as mock_version:
                mock_version.return_value = "0.2.7"
                # Pass in an API Key for use in LLM calls
                kwargs = {"api_key": "sk-svcacct-abcdef12345"}
                result = _init_chat_completion_model("gpt-3.5-turbo", "openai", kwargs)
                assert result == "chat_model"
                mock_init.assert_called_once_with(
                    model="gpt-3.5-turbo",
                    model_provider="openai",
                    api_key="sk-svcacct-abcdef12345",
                )

    def test_init_chat_completion_model_old_version(self):
        with patch(
            "nemoguardrails.llm.models.langchain_initializer.version"
        ) as mock_version:
            mock_version.return_value = "0.2.6"
            with pytest.raises(
                RuntimeError,
                match="this feature is supported from v0.2.7 of langchain-core",
            ):
                _init_chat_completion_model("gpt-3.5-turbo", "openai", {})

    def test_init_chat_completion_model_error(self):
        with patch(
            "nemoguardrails.llm.models.langchain_initializer.init_chat_model"
        ) as mock_init:
            mock_init.side_effect = ValueError("Chat model failed")
            with patch(
                "nemoguardrails.llm.models.langchain_initializer.version"
            ) as mock_version:
                mock_version.return_value = "0.2.7"
                with pytest.raises(ValueError, match="Chat model failed"):
                    _init_chat_completion_model("gpt-3.5-turbo", "openai", {})


class TestCommunityChatInitializer:
    """Tests for the community chat initializer."""

    def test_init_community_chat_models_success(self):
        with patch(
            "nemoguardrails.llm.models.langchain_initializer._get_chat_completion_provider"
        ) as mock_get_provider:
            mock_provider_cls = MagicMock()
            mock_provider_cls.model_fields = {"model": None}
            mock_provider_cls.return_value = "community_model"
            mock_get_provider.return_value = mock_provider_cls
            result = _init_community_chat_models("community-model", "provider", {})
            assert result == "community_model"
            mock_get_provider.assert_called_once_with("provider")
            mock_provider_cls.assert_called_once_with(model="community-model")

    def test_init_community_chat_models_with_api_key_success(self):
        with patch(
            "nemoguardrails.llm.models.langchain_initializer._get_chat_completion_provider"
        ) as mock_get_provider:
            mock_provider_cls = MagicMock()
            mock_provider_cls.model_fields = {"model": None}
            mock_provider_cls.return_value = "community_model"
            mock_get_provider.return_value = mock_provider_cls
            # Pass in an API Key for use in client creation
            api_key = "abcdef12345"
            result = _init_community_chat_models(
                "community-model", "provider", {"api_key": api_key}
            )
            assert result == "community_model"
            mock_get_provider.assert_called_once_with("provider")
            mock_provider_cls.assert_called_once_with(
                model="community-model", api_key=api_key
            )

    def test_init_community_chat_models_no_provider(self):
        with patch(
            "nemoguardrails.llm.models.langchain_initializer._get_chat_completion_provider"
        ) as mock_get_provider:
            mock_get_provider.return_value = None
            with pytest.raises(ValueError):
                _init_community_chat_models("community-model", "provider", {})


class TestTextCompletionInitializer:
    """Tests for the text completion initializer."""

    def test_init_text_completion_model_success(self):
        with patch(
            "nemoguardrails.llm.models.langchain_initializer._get_text_completion_provider"
        ) as mock_get_provider:
            mock_provider_cls = MagicMock()
            mock_provider_cls.model_fields = {"model": None}
            mock_provider_cls.return_value = "text_model"
            mock_get_provider.return_value = mock_provider_cls
            result = _init_text_completion_model("text-model", "provider", {})
            assert result == "text_model"
            mock_get_provider.assert_called_once_with("provider")
            mock_provider_cls.assert_called_once_with(model="text-model")

    def test_init_text_completion_model_with_api_key_success(self):
        with patch(
            "nemoguardrails.llm.models.langchain_initializer._get_text_completion_provider"
        ) as mock_get_provider:
            mock_provider_cls = MagicMock()
            mock_provider_cls.model_fields = {"model": None}
            mock_provider_cls.return_value = "text_model"
            mock_get_provider.return_value = mock_provider_cls
            # Pass in an API Key for use in client creation
            api_key = "abcdef12345"
            result = _init_text_completion_model(
                "text-model", "provider", {"api_key": api_key}
            )
            assert result == "text_model"
            mock_get_provider.assert_called_once_with("provider")
            mock_provider_cls.assert_called_once_with(
                model="text-model", api_key=api_key
            )

    def test_init_text_completion_model_no_provider(self):
        with patch(
            "nemoguardrails.llm.models.langchain_initializer._get_text_completion_provider"
        ) as mock_get_provider:
            mock_get_provider.return_value = None
            with pytest.raises(ValueError):
                _init_text_completion_model("text-model", "provider", {})


class TestUpdateModelKwargs:
    """Tests for the _update_model_kwargs function."""

    def test_update_model_kwargs_with_model_field(self):
        mock_provider_cls = MagicMock()
        mock_provider_cls.model_fields = {"model": {}}
        kwargs = {}
        updated_kwargs = _update_model_kwargs(mock_provider_cls, "test-model", kwargs)
        assert updated_kwargs == {"model": "test-model"}

    def test_update_model_kwargs_with_model_field_and_api_key(self):
        mock_provider_cls = MagicMock()
        mock_provider_cls.model_fields = {"model": {}}
        api_key = "abcdef12345"
        updated_kwargs = _update_model_kwargs(
            mock_provider_cls, "test-model", {"api_key": api_key}
        )
        assert updated_kwargs == {"model": "test-model", "api_key": api_key}

    def test_update_model_kwargs_with_model_name_field(self):
        """Test that _update_model_kwargs updates kwargs with model name when provider has model_name field."""
        mock_provider_cls = MagicMock()
        mock_provider_cls.model_fields = {"model_name": {}}
        kwargs = {}
        updated_kwargs = _update_model_kwargs(mock_provider_cls, "test-model", kwargs)
        assert updated_kwargs == {"model_name": "test-model"}

    def test_update_model_kwargs_with_model_name_and_api_key_field(self):
        """Test that _update_model_kwargs updates kwargs with model name when provider has model_name field."""
        mock_provider_cls = MagicMock()
        mock_provider_cls.model_fields = {"model_name": {}}
        api_key = "abcdef12345"
        updated_kwargs = _update_model_kwargs(
            mock_provider_cls, "test-model", {"api_key": api_key}
        )
        assert updated_kwargs == {"model_name": "test-model", "api_key": api_key}

    def test_update_model_kwargs_with_both_fields(self):
        """Test _update_model_kwargs updates kwargs with model name when provider has both model and model_name fields."""

        mock_provider_cls = MagicMock()
        mock_provider_cls.model_fields = {"model": {}, "model_name": {}}
        kwargs = {}
        updated_kwargs = _update_model_kwargs(mock_provider_cls, "test-model", kwargs)
        assert updated_kwargs == {"model": "test-model", "model_name": "test-model"}

    def test_update_model_kwargs_with_both_fields_and_api_key(self):
        """Test _update_model_kwargs updates kwargs with model name when provider has both model and model_name fields."""

        mock_provider_cls = MagicMock()
        mock_provider_cls.model_fields = {"model": {}, "model_name": {}}
        api_key = "abcdef12345"
        updated_kwargs = _update_model_kwargs(
            mock_provider_cls, "test-model", {"api_key": api_key}
        )
        assert updated_kwargs == {
            "model": "test-model",
            "model_name": "test-model",
            "api_key": api_key,
        }

    def test_update_model_kwargs_with_existing_kwargs(self):
        """Test _update_model_kwargs preserves existing kwargs."""

        mock_provider_cls = MagicMock()
        mock_provider_cls.model_fields = {"model": {}}
        kwargs = {"temperature": 0.7}
        updated_kwargs = _update_model_kwargs(mock_provider_cls, "test-model", kwargs)
        assert updated_kwargs == {"model": "test-model", "temperature": 0.7}

    def test_update_model_kwargs_and_api_key_with_existing_kwargs(self):
        """Test _update_model_kwargs preserves existing kwargs."""

        mock_provider_cls = MagicMock()
        mock_provider_cls.model_fields = {"model": {}}
        api_key = "abcdef12345"
        kwargs = {"temperature": 0.7, "api_key": api_key}
        updated_kwargs = _update_model_kwargs(mock_provider_cls, "test-model", kwargs)
        assert updated_kwargs == {
            "model": "test-model",
            "temperature": 0.7,
            "api_key": api_key,
        }
