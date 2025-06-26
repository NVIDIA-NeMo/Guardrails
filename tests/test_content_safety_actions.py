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


# NEW: Tests specifically for the output parsing logic with starred unpacking
@pytest.mark.asyncio
async def test_content_safety_input_parsing_empty_violations():
    """Test content_safety_check_input parsing logic with empty violations list."""
    mock_llm = FakeLLM(responses=["result"])
    llms = {"test_model": mock_llm}

    mock_task_manager = MagicMock()
    mock_parsed_result = MagicMock()
    # Simulate parsing result with only is_safe, no violation policies
    mock_parsed_result.text = [True]
    mock_task_manager.render_task_prompt.return_value = "test prompt"
    mock_task_manager.get_stop_tokens.return_value = []
    mock_task_manager.get_max_tokens.return_value = 3
    mock_task_manager.parse_task_output.return_value = mock_parsed_result

    context = {"user_message": "safe content"}

    result = await content_safety_check_input(
        llms=llms,
        llm_task_manager=mock_task_manager,
        model_name="test_model",
        context=context,
    )

    assert result["allowed"] is True
    assert result["policy_violations"] == []


@pytest.mark.asyncio
async def test_content_safety_input_parsing_single_violation():
    """Test content_safety_check_input parsing logic with single violation policy."""
    mock_llm = FakeLLM(responses=["result"])
    llms = {"test_model": mock_llm}

    mock_task_manager = MagicMock()
    mock_parsed_result = MagicMock()
    mock_parsed_result.text = [False, "spam"]
    mock_task_manager.render_task_prompt.return_value = "test prompt"
    mock_task_manager.get_stop_tokens.return_value = []
    mock_task_manager.get_max_tokens.return_value = 3
    mock_task_manager.parse_task_output.return_value = mock_parsed_result

    context = {"user_message": "spam content"}

    result = await content_safety_check_input(
        llms=llms,
        llm_task_manager=mock_task_manager,
        model_name="test_model",
        context=context,
    )

    assert result["allowed"] is False
    assert result["policy_violations"] == ["spam"]


@pytest.mark.asyncio
async def test_content_safety_input_parsing_multiple_violations():
    """Test content_safety_check_input parsing logic with multiple violation policies."""
    mock_llm = FakeLLM(responses=["result"])
    llms = {"test_model": mock_llm}

    mock_task_manager = MagicMock()
    mock_parsed_result = MagicMock()
    mock_parsed_result.text = [False, "violence", "hate_speech", "harassment"]
    mock_task_manager.render_task_prompt.return_value = "test prompt"
    mock_task_manager.get_stop_tokens.return_value = []
    mock_task_manager.get_max_tokens.return_value = 3
    mock_task_manager.parse_task_output.return_value = mock_parsed_result

    context = {"user_message": "unsafe content"}

    result = await content_safety_check_input(
        llms=llms,
        llm_task_manager=mock_task_manager,
        model_name="test_model",
        context=context,
    )

    assert result["allowed"] is False
    assert result["policy_violations"] == ["violence", "hate_speech", "harassment"]


@pytest.mark.asyncio
async def test_content_safety_output_parsing_empty_violations():
    """Test content_safety_check_output parsing logic with empty violations list."""
    mock_llm = FakeLLM(responses=["result"])
    llms = {"test_model": mock_llm}

    mock_task_manager = MagicMock()
    mock_parsed_result = MagicMock()
    mock_parsed_result.text = [True]
    mock_task_manager.render_task_prompt.return_value = "test prompt"
    mock_task_manager.get_stop_tokens.return_value = []
    mock_task_manager.get_max_tokens.return_value = 3
    mock_task_manager.parse_task_output.return_value = mock_parsed_result

    context = {"user_message": "input", "bot_message": "safe response"}

    result = await content_safety_check_output(
        llms=llms,
        llm_task_manager=mock_task_manager,
        model_name="test_model",
        context=context,
    )

    assert result["allowed"] is True
    assert result["policy_violations"] == []


@pytest.mark.asyncio
async def test_content_safety_output_parsing_single_violation():
    """Test content_safety_check_output parsing logic with single violation policy."""
    mock_llm = FakeLLM(responses=["result"])
    llms = {"test_model": mock_llm}

    mock_task_manager = MagicMock()
    mock_parsed_result = MagicMock()
    mock_parsed_result.text = [False, "inappropriate"]
    mock_task_manager.render_task_prompt.return_value = "test prompt"
    mock_task_manager.get_stop_tokens.return_value = []
    mock_task_manager.get_max_tokens.return_value = 3
    mock_task_manager.parse_task_output.return_value = mock_parsed_result

    context = {"user_message": "input", "bot_message": "inappropriate response"}

    result = await content_safety_check_output(
        llms=llms,
        llm_task_manager=mock_task_manager,
        model_name="test_model",
        context=context,
    )

    assert result["allowed"] is False
    assert result["policy_violations"] == ["inappropriate"]


@pytest.mark.asyncio
async def test_content_safety_output_parsing_multiple_violations():
    """Test content_safety_check_output parsing logic with multiple violation policies."""
    mock_llm = FakeLLM(responses=["result"])
    llms = {"test_model": mock_llm}

    mock_task_manager = MagicMock()
    mock_parsed_result = MagicMock()
    mock_parsed_result.text = [False, "toxic", "offensive", "harmful"]
    mock_task_manager.render_task_prompt.return_value = "test prompt"
    mock_task_manager.get_stop_tokens.return_value = []
    mock_task_manager.get_max_tokens.return_value = 3
    mock_task_manager.parse_task_output.return_value = mock_parsed_result

    context = {"user_message": "input", "bot_message": "unsafe response"}

    result = await content_safety_check_output(
        llms=llms,
        llm_task_manager=mock_task_manager,
        model_name="test_model",
        context=context,
    )

    assert result["allowed"] is False
    assert result["policy_violations"] == ["toxic", "offensive", "harmful"]


@pytest.mark.asyncio
async def test_content_safety_input_parsing_edge_case_safe_with_violations():
    """Test content_safety_check_input parsing logic when marked safe but has violation policies listed."""
    mock_llm = FakeLLM(responses=["result"])
    llms = {"test_model": mock_llm}

    mock_task_manager = MagicMock()
    mock_parsed_result = MagicMock()
    # edge case: is_safe=True but still has violation policies listed
    mock_parsed_result.text = [True, "minor_concern", "flagged"]
    mock_task_manager.render_task_prompt.return_value = "test prompt"
    mock_task_manager.get_stop_tokens.return_value = []
    mock_task_manager.get_max_tokens.return_value = 3
    mock_task_manager.parse_task_output.return_value = mock_parsed_result

    context = {"user_message": "edge case content"}

    result = await content_safety_check_input(
        llms=llms,
        llm_task_manager=mock_task_manager,
        model_name="test_model",
        context=context,
    )

    assert result["allowed"] is True
    assert result["policy_violations"] == ["minor_concern", "flagged"]


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


def test_content_safety_check_output_mapping_blocked_policy_violations_only():
    """Test content_safety_check_output_mapping returns True when content should be blocked."""

    # TODO:@trebedea is this the expected behavior?
    result = {"allowed": True, "policy_violations": ["violence"]}
    assert content_safety_check_output_mapping(result) is False


def test_content_safety_check_output_mapping_default():
    """Test content_safety_check_output_mapping defaults to allowed=True when key is missing."""
    result = {"policy_violations": []}
    assert content_safety_check_output_mapping(result) is False
