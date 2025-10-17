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
Tests for the special case handlers in the LangChain model initializer.

This module contains tests for the special case handlers that are used to initialize
specific models or providers that require custom logic.
"""

import os
from unittest.mock import patch

import pytest
from langchain.chat_models.base import BaseChatModel
from langchain_core.language_models import BaseLLM

from nemoguardrails.llm.models.langchain_initializer import (
    _PROVIDER_INITIALIZERS,
    _SPECIAL_MODEL_INITIALIZERS,
    ModelInitializationError,
    _handle_model_special_cases,
    _init_gpt35_turbo_instruct,
    _init_nvidia_model,
)


def has_openai():
    """Check if OpenAI package is installed."""
    try:
        import langchain_openai

        return True
    except ImportError:
        return False


def has_nvidia_ai_endpoints():
    """Check if NVIDIA AI Endpoints package is installed."""
    try:
        import langchain_nvidia_ai_endpoints

        return True
    except ImportError:
        return False


class TestSpecialCaseHandlers:
    """Tests for the special case handlers."""

    def test_handle_model_special_cases_no_match(self):
        """Test that _handle_model_special_cases returns None when no special case matche."""

        result = _handle_model_special_cases("unknown-model", "unknown-provider", {})
        assert result is None

    @pytest.mark.skipif(
        not has_openai(), reason="langchain-openai package not installed"
    )
    def test_handle_model_special_cases_model_match(self):
        """Test that model-specific initializers are called correctly."""

        # skip if OpenAI API key is not set
        if not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("OpenAI API key not set")
        model_name = "gpt-3.5-turbo-instruct"
        provider_name = "openai"
        kwargs = {"temperature": 0.7}

        result = _handle_model_special_cases(model_name, provider_name, kwargs)

        # ensure the result is a text completion model
        assert result is not None
        assert hasattr(result, "invoke")
        assert hasattr(result, "generate")
        assert isinstance(result, BaseLLM)

    @pytest.mark.skipif(
        not has_nvidia_ai_endpoints(),
        reason="nvidia-ai-endpoints package not installed",
    )
    def test_handle_model_special_cases_provider_match(self):
        """Test that provider-specific initializers are called correctly."""

        model_name = "meta/llama-3.3-70b-instruct"
        provider_name = "nim"
        kwargs = {"temperature": 0.7}

        result = _handle_model_special_cases(model_name, provider_name, kwargs)

        # enure the result is a chat model
        assert result is not None
        assert isinstance(result, BaseChatModel)
        assert hasattr(result, "invoke")
        assert hasattr(result, "generate")

    def test_special_model_initializers_registry(self):
        """Test that the _SPECIAL_MODEL_INITIALIZERS registry contains the expected entries."""

        assert "gpt-3.5-turbo-instruct" in _SPECIAL_MODEL_INITIALIZERS
        assert (
            _SPECIAL_MODEL_INITIALIZERS["gpt-3.5-turbo-instruct"]
            == _init_gpt35_turbo_instruct
        )

    def test_provider_initializers_registry(self):
        """Test that the _PROVIDER_INITIALIZERS registry contains the expected entries."""
        assert "nvidia_ai_endpoints" in _PROVIDER_INITIALIZERS
        assert "nim" in _PROVIDER_INITIALIZERS
        assert _PROVIDER_INITIALIZERS["nvidia_ai_endpoints"] == _init_nvidia_model
        assert _PROVIDER_INITIALIZERS["nim"] == _init_nvidia_model


class TestGPT35TurboInstructInitializer:
    """Tests for the GPT-3.5 Turbo Instruct initializer."""

    def test_init_gpt35_turbo_instruct(self):
        """Test that _init_gpt35_turbo_instruct calls _init_text_completion_model."""

        with patch(
            "nemoguardrails.llm.models.langchain_initializer._init_text_completion_model"
        ) as mock_init:
            mock_init.return_value = "text_model"
            result = _init_gpt35_turbo_instruct("gpt-3.5-turbo-instruct", "openai", {})
            assert result == "text_model"
            mock_init.assert_called_once_with(
                model_name="gpt-3.5-turbo-instruct", provider_name="openai", kwargs={}
            )

    def test_init_gpt35_turbo_instruct_error(self):
        """Test that _init_gpt35_turbo_instruct raises ModelInitializationError on failure."""
        with patch(
            "nemoguardrails.llm.models.langchain_initializer._init_text_completion_model"
        ) as mock_init:
            mock_init.side_effect = ValueError("Text model failed")
            with pytest.raises(
                ModelInitializationError,
                match="Failed to initialize text completion model gpt-3.5-turbo-instruct",
            ):
                _init_gpt35_turbo_instruct("gpt-3.5-turbo-instruct", "openai", {})


class TestNVIDIAModelInitializer:
    """Tests for the NVIDIA model initializer."""

    @pytest.mark.skipif(
        not has_nvidia_ai_endpoints(),
        reason="Requires  NVIDIA AI Endpoints package",
    )
    def test_init_nvidia_model_success(self):
        """Test that _init_nvidia_model initializes a ChatNVIDIA model."""
        result = _init_nvidia_model(
            "meta/llama-3.3-70b-instruct",
            "nim",
            {
                "api_key": "asdf"
            },  # Note in future version of nvaie this might raise an error
        )
        assert result is not None
        assert hasattr(result, "invoke")
        assert hasattr(result, "generate")
        assert hasattr(result, "agenerate")
        assert isinstance(result, BaseChatModel)

    @pytest.mark.skipif(
        not has_nvidia_ai_endpoints(), reason="Requires NVIDIA AI Endpoints package"
    )
    def test_init_nvidia_model_old_version(self):
        """Test that _init_nvidia_model raises ValueError for old versions."""

        from importlib.metadata import version

        from packaging.version import parse

        current_version = version("langchain_nvidia_ai_endpoints")
        if parse(current_version) < parse("0.2.0"):
            with pytest.raises(
                ValueError,
                match="langchain_nvidia_ai_endpoints version must be 0.2.0 or above",
            ):
                _init_nvidia_model("some-model", "nvidia_ai_endpoints", {})
        else:
            pytest.skip("NVIDIA AI Endpoints version is >= 0.2.0")

    def test_init_nvidia_model_import_error(self):
        """Test that _init_nvidia_model raises ImportError when langchain_nvidia_ai_endpoints is not installed."""

        if has_nvidia_ai_endpoints():
            pytest.skip("NVIDIA AI Endpoints package is installed")
        with pytest.raises(ImportError):
            _init_nvidia_model("nvidia-model", "nvidia_ai_endpoints", {})
