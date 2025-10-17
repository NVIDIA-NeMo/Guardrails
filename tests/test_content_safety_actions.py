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

# conftest.py
import pytest

from nemoguardrails.library.content_safety.actions import (
    content_safety_check_input,
    content_safety_check_output,
    content_safety_check_output_mapping,
)
from tests.utils import FakeLLM


@pytest.fixture
def fake_llm():
    def _factory(response):
        llm = FakeLLM(responses=[response])
        return {"test_model": llm}

    return _factory


@pytest.fixture
def mock_task_manager():
    tm = MagicMock()
    tm.render_task_prompt.return_value = "test prompt"
    tm.get_stop_tokens.return_value = []
    tm.get_max_tokens.return_value = 3
    return tm


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "check_fn, context, parsed_text, expected_allowed, expected_violations",
    [
        (
            content_safety_check_input,
            {"user_message": "foo"},
            [True, "policy1", "policy2"],
            True,
            ["policy1", "policy2"],
        ),
        (
            content_safety_check_input,
            {"user_message": "foo"},
            [False],
            False,
            [],
        ),
        (
            content_safety_check_output,
            {"user_message": "foo", "bot_message": "bar"},
            [False, "hate", "violence"],
            False,
            ["hate", "violence"],
        ),
        (
            content_safety_check_output,
            {"user_message": "foo", "bot_message": "bar"},
            [True],
            True,
            [],
        ),
    ],
)
async def test_content_safety_parsing(
    fake_llm,
    mock_task_manager,
    check_fn,
    context,
    parsed_text,
    expected_allowed,
    expected_violations,
):
    llms = fake_llm("irrelevant")
    mock_task_manager.parse_task_output.return_value = parsed_text

    result = await check_fn(
        llms=llms,
        llm_task_manager=mock_task_manager,
        model_name="test_model",
        context=context,
    )
    assert result["allowed"] is expected_allowed
    assert result["policy_violations"] == expected_violations


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
    """Test content_safety_check_output_mapping defaults to allowed=False when key is missing."""
    result = {"policy_violations": []}
    assert content_safety_check_output_mapping(result) is False
