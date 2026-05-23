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

"""
Tests for the special case handlers in the LangChain model initializer.

This module contains tests for the special case handlers that are used to initialize
provider-specific chat models that require custom logic.
"""

import pytest
from langchain_core.language_models import BaseChatModel

from nemoguardrails.integrations.langchain.langchain_initializer import (
    _PROVIDER_INITIALIZERS,
    _handle_model_special_cases,
    _init_nvidia_model,
)


def has_openai():
    """Check if OpenAI package is installed."""
    from nemoguardrails.imports import check_optional_dependency

    return check_optional_dependency("langchain_openai")


def has_nvidia_ai_endpoints():
    """Check if NVIDIA AI Endpoints package is installed."""
    from nemoguardrails.imports import check_optional_dependency

    return check_optional_dependency("langchain_nvidia_ai_endpoints")


class TestSpecialCaseHandlers:
    """Tests for the special case handlers."""

    def test_handle_model_special_cases_no_match(self):
        """Test that _handle_model_special_cases returns None when no special case matches."""

        result = _handle_model_special_cases("unknown-model", "unknown-provider", {})
        assert result is None

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

        # ensure the result is a chat model
        assert result is not None
        assert isinstance(result, BaseChatModel)
        assert hasattr(result, "invoke")
        assert hasattr(result, "generate")

    def test_provider_initializers_registry(self):
        """Test that the _PROVIDER_INITIALIZERS registry contains the expected entries."""
        assert "nvidia_ai_endpoints" in _PROVIDER_INITIALIZERS
        assert "nim" in _PROVIDER_INITIALIZERS
        assert _PROVIDER_INITIALIZERS["nvidia_ai_endpoints"] == _init_nvidia_model
        assert _PROVIDER_INITIALIZERS["nim"] == _init_nvidia_model


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
            {"api_key": "asdf"},  # Note in future version of nvaie this might raise an error
        )
        assert result is not None
        assert hasattr(result, "invoke")
        assert hasattr(result, "generate")
        assert hasattr(result, "agenerate")
        assert isinstance(result, BaseChatModel)

    @pytest.mark.skipif(not has_nvidia_ai_endpoints(), reason="Requires NVIDIA AI Endpoints package")
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
