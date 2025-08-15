# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Tests for LLM isolation with models that don't have model_kwargs field."""

from typing import Any, Dict, List, Optional
from unittest.mock import Mock

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import BaseModel, Field

from nemoguardrails.rails.llm.config import RailsConfig
from nemoguardrails.rails.llm.llmrails import LLMRails


class StrictPydanticLLM(BaseModel):
    """Mock Pydantic LLM that doesn't allow arbitrary attributes (like ChatNVIDIA)."""

    class Config:
        extra = "forbid"

    temperature: float = Field(default=0.7)
    max_tokens: Optional[int] = Field(default=None)


class MockChatNVIDIA(BaseChatModel):
    """Mock ChatNVIDIA-like model that doesn't have model_kwargs."""

    model: str = "nvidia-model"
    temperature: float = 0.7

    class Config:
        extra = "forbid"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Mock generation method."""
        return ChatResult(generations=[ChatGeneration(message=Mock())])

    @property
    def _llm_type(self) -> str:
        """Return the type of language model."""
        return "nvidia"


class FlexibleLLMWithModelKwargs(BaseModel):
    """Mock LLM that has model_kwargs and allows modifications."""

    class Config:
        extra = "allow"

    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    temperature: float = 0.7


class FlexibleLLMWithoutModelKwargs(BaseModel):
    """Mock LLM that doesn't have model_kwargs but allows adding attributes."""

    class Config:
        extra = "allow"

    temperature: float = 0.7
    # no model_kwargs field


@pytest.fixture
def test_config():
    """Create test configuration."""
    return RailsConfig.from_content(
        """
        models:
          - type: main
            engine: openai
            model: gpt-3.5-turbo
        """
    )


class TestLLMIsolationModelKwargsFix:
    """Test LLM isolation with different model types."""

    def test_strict_pydantic_model_without_model_kwargs(self, test_config):
        """Test isolation with strict Pydantic model that doesn't have model_kwargs."""
        rails = LLMRails(config=test_config, verbose=False)

        strict_llm = StrictPydanticLLM(temperature=0.5)

        isolated_llm = rails._create_action_llm_copy(strict_llm, "test_action")

        assert isolated_llm is not None
        assert isolated_llm is not strict_llm
        assert isolated_llm.temperature == 0.5
        assert not hasattr(isolated_llm, "model_kwargs")

    def test_mock_chat_nvidia_without_model_kwargs(self, test_config):
        """Test with a ChatNVIDIA-like model that doesn't allow arbitrary attributes."""
        rails = LLMRails(config=test_config, verbose=False)

        nvidia_llm = MockChatNVIDIA()

        isolated_llm = rails._create_action_llm_copy(nvidia_llm, "self_check_output")

        assert isolated_llm is not None
        assert isolated_llm is not nvidia_llm
        assert isolated_llm.model == "nvidia-model"
        assert isolated_llm.temperature == 0.7
        assert not hasattr(isolated_llm, "model_kwargs")

    def test_flexible_llm_with_model_kwargs(self, test_config):
        """Test with LLM that has model_kwargs field."""
        rails = LLMRails(config=test_config, verbose=False)

        llm_with_kwargs = FlexibleLLMWithModelKwargs(
            model_kwargs={"custom_param": "value"}, temperature=0.3
        )

        isolated_llm = rails._create_action_llm_copy(llm_with_kwargs, "test_action")

        assert isolated_llm is not None
        assert isolated_llm is not llm_with_kwargs
        assert hasattr(isolated_llm, "model_kwargs")
        assert isolated_llm.model_kwargs == {"custom_param": "value"}
        assert isolated_llm.model_kwargs is not llm_with_kwargs.model_kwargs

        isolated_llm.model_kwargs["new_param"] = "new_value"
        assert "new_param" not in llm_with_kwargs.model_kwargs

    def test_flexible_llm_without_model_kwargs_but_allows_adding(self, test_config):
        """Test with LLM that doesn't have model_kwargs but allows adding attributes."""
        rails = LLMRails(config=test_config, verbose=False)

        flexible_llm = FlexibleLLMWithoutModelKwargs(temperature=0.8)

        isolated_llm = rails._create_action_llm_copy(flexible_llm, "test_action")

        assert isolated_llm is not None
        assert isolated_llm is not flexible_llm
        assert isolated_llm.temperature == 0.8
        # since it allows extra attributes, model_kwargs might have been added
        # but it shouldn't cause an error either way

    def test_llm_with_none_model_kwargs(self, test_config):
        """Test with LLM that has model_kwargs set to None."""
        rails = LLMRails(config=test_config, verbose=False)

        llm_with_none = FlexibleLLMWithModelKwargs(temperature=0.6)
        llm_with_none.model_kwargs = None

        isolated_llm = rails._create_action_llm_copy(llm_with_none, "test_action")

        assert isolated_llm is not None
        assert isolated_llm is not llm_with_none
        if hasattr(isolated_llm, "model_kwargs"):
            assert isolated_llm.model_kwargs in (None, {})

    def test_copy_preserves_other_attributes(self, test_config):
        """Test that copy preserves other attributes correctly."""
        rails = LLMRails(config=test_config, verbose=False)

        strict_llm = StrictPydanticLLM(temperature=0.2, max_tokens=100)
        isolated_strict = rails._create_action_llm_copy(strict_llm, "action1")

        assert isolated_strict.temperature == 0.2
        assert isolated_strict.max_tokens == 100

        flexible_llm = FlexibleLLMWithModelKwargs(
            model_kwargs={"key": "value"}, temperature=0.9
        )
        isolated_flexible = rails._create_action_llm_copy(flexible_llm, "action2")

        assert isolated_flexible.temperature == 0.9
        assert isolated_flexible.model_kwargs == {"key": "value"}
