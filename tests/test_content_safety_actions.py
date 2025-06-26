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

from unittest.mock import AsyncMock, MagicMock

import pytest

from nemoguardrails.library.content_safety.actions import (
    content_safety_check_input,
    content_safety_check_output,
    content_safety_check_output_mapping,
)
from tests.utils import FakeLLM


@pytest.mark.asyncio
async def test_content_safety_check_input_with_tuple_result():
    """Test content_safety_check_input when result is a tuple with both is_safe and violated_policies."""
    mock_llm = FakeLLM(responses=["safe"])
    llms = {"test_model": mock_llm}

    mock_task_manager = MagicMock()
    mock_parsed_result = MagicMock()
    mock_parsed_result.text = [
        True,
        "policy1",
        "policy2",
    ]  # is_safe=True, violated_policies=["policy1", "policy2"]
    mock_task_manager.render_task_prompt.return_value = "test prompt"
    mock_task_manager.get_stop_tokens.return_value = []
    mock_task_manager.get_max_tokens.return_value = 3
    mock_task_manager.parse_task_output.return_value = mock_parsed_result

    context = {"user_message": "test input"}

    result = await content_safety_check_input(
        llms=llms,
        llm_task_manager=mock_task_manager,
        model_name="test_model",
        context=context,
    )

    assert result["allowed"] is True
    assert result["policy_violations"] == ["policy1", "policy2"]


@pytest.mark.asyncio
async def test_content_safety_check_input_with_single_result():
    """Test content_safety_check_input when result is a single value (just is_safe)."""
    mock_llm = FakeLLM(responses=["unsafe"])
    llms = {"test_model": mock_llm}

    mock_task_manager = MagicMock()
    mock_parsed_result = MagicMock()
    mock_parsed_result.text = [False]  # only is_safe=False, no violated_policies
    mock_task_manager.render_task_prompt.return_value = "test prompt"
    mock_task_manager.get_stop_tokens.return_value = []
    mock_task_manager.get_max_tokens.return_value = 3
    mock_task_manager.parse_task_output.return_value = mock_parsed_result

    context = {"user_message": "test input"}

    result = await content_safety_check_input(
        llms=llms,
        llm_task_manager=mock_task_manager,
        model_name="test_model",
        context=context,
    )

    assert result["allowed"] is False
    assert result["policy_violations"] == []


@pytest.mark.asyncio
async def test_content_safety_check_output_with_tuple_result():
    """Test content_safety_check_output when result is a tuple with both is_safe and violated_policies."""
    mock_llm = FakeLLM(responses=["unsafe"])
    llms = {"test_model": mock_llm}

    mock_task_manager = MagicMock()
    mock_parsed_result = MagicMock()
    mock_parsed_result.text = [
        False,
        "violence",
        "hate",
    ]  # is_safe=False, violated_policies=["violence", "hate"]
    mock_task_manager.render_task_prompt.return_value = "test prompt"
    mock_task_manager.get_stop_tokens.return_value = []
    mock_task_manager.get_max_tokens.return_value = 3
    mock_task_manager.parse_task_output.return_value = mock_parsed_result

    context = {"user_message": "test input", "bot_message": "test response"}

    result = await content_safety_check_output(
        llms=llms,
        llm_task_manager=mock_task_manager,
        model_name="test_model",
        context=context,
    )

    assert result["allowed"] is False
    assert result["policy_violations"] == ["violence", "hate"]


@pytest.mark.asyncio
async def test_content_safety_check_output_with_single_result():
    """Test content_safety_check_output when result is a single value (just is_safe)."""
    mock_llm = FakeLLM(responses=["safe"])
    llms = {"test_model": mock_llm}

    mock_task_manager = MagicMock()
    mock_parsed_result = MagicMock()
    mock_parsed_result.text = [True]  # Only is_safe=True, no violated_policies
    mock_task_manager.render_task_prompt.return_value = "test prompt"
    mock_task_manager.get_stop_tokens.return_value = []
    mock_task_manager.get_max_tokens.return_value = 3
    mock_task_manager.parse_task_output.return_value = mock_parsed_result

    context = {"user_message": "test input", "bot_message": "test response"}

    result = await content_safety_check_output(
        llms=llms,
        llm_task_manager=mock_task_manager,
        model_name="test_model",
        context=context,
    )

    assert result["allowed"] is True
    assert result["policy_violations"] == []


@pytest.mark.asyncio
async def test_content_safety_check_input_missing_model_name():
    """Test content_safety_check_input raises ValueError when model_name is missing."""
    llms = {}
    mock_task_manager = MagicMock()

    with pytest.raises(ValueError, match="Model name is required"):
        await content_safety_check_input(
            llms=llms, llm_task_manager=mock_task_manager, model_name=None, context={}
        )


@pytest.mark.asyncio
async def test_content_safety_check_input_model_not_found():
    """Test content_safety_check_input raises ValueError when model is not found."""
    llms = {}
    mock_task_manager = MagicMock()

    with pytest.raises(ValueError, match="Model test_model not found"):
        await content_safety_check_input(
            llms=llms,
            llm_task_manager=mock_task_manager,
            model_name="test_model",
            context={},
        )


def test_content_safety_check_output_mapping_allowed():
    """Test content_safety_check_output_mapping returns False when content is allowed."""
    result = {"allowed": True, "policy_violations": []}
    assert content_safety_check_output_mapping(result) is False


def test_content_safety_check_output_mapping_blocked():
    """Test content_safety_check_output_mapping returns True when content should be blocked."""

    result = {"allowed": False, "policy_violations": ["violence"]}
    assert content_safety_check_output_mapping(result) is True


def test_content_safety_check_output_mapping_default():
    """Test content_safety_check_output_mapping defaults to allowed=True when key is missing."""
    result = {"policy_violations": []}
    assert content_safety_check_output_mapping(result) is False
