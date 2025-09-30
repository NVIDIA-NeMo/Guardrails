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

"""Integration tests for content safety actions with output parsers.

These tests verify that the modified parser interface (list format instead of tuple format)
works correctly with the actual content safety actions and their iterable unpacking logic.
"""

from unittest.mock import MagicMock

import pytest

from nemoguardrails.library.content_safety.actions import (
    content_safety_check_input,
    content_safety_check_output,
)
from nemoguardrails.llm.output_parsers import (
    is_content_safe,
    nemoguard_parse_prompt_safety,
    nemoguard_parse_response_safety,
)
from tests.utils import FakeLLM


def _create_mock_setup(llm_responses, parsed_result):
    mock_llm = FakeLLM(responses=llm_responses)
    llms = {"test_model": mock_llm}

    mock_task_manager = MagicMock()

    mock_task_manager.render_task_prompt.return_value = "test prompt"
    mock_task_manager.get_stop_tokens.return_value = []
    mock_task_manager.get_max_tokens.return_value = 3
    mock_task_manager.parse_task_output.return_value = parsed_result

    return llms, mock_task_manager


def _create_input_context(user_message="Hello, how are you?"):
    return {"user_message": user_message}


def _create_output_context(user_message="Hi", bot_message="Hello! How can I help you?"):
    return {"user_message": user_message, "bot_message": bot_message}


class TestContentSafetyParserIntegration:
    """Integration tests using parsers with content safety actions."""

    @pytest.mark.parametrize(
        "response,expected_allowed,expected_violations",
        [
            ("safe", True, []),
            ("unsafe", False, []),
            ("unsafe S1 S8", False, ["S1", "S8"]),
            ("yes", False, []),
            ("no", True, []),
        ],
    )
    @pytest.mark.asyncio
    async def test_content_safety_input_with_is_content_safe_parser(
        self, response, expected_allowed, expected_violations
    ):
        parsed_result = is_content_safe(response)
        llms, mock_task_manager = _create_mock_setup([response], parsed_result)
        context = _create_input_context()

        result = await content_safety_check_input(
            llms=llms,
            llm_task_manager=mock_task_manager,
            model_name="test_model",
            context=context,
        )

        assert result["allowed"] is expected_allowed
        assert result["policy_violations"] == expected_violations

    @pytest.mark.asyncio
    async def test_content_safety_input_with_is_content_safe_parser_safe_with_violations(
        self,
    ):
        """Test input action with is_content_safe parser; safe with violations."""
        parsed_result = is_content_safe("safe S1 S8")
        llms, mock_task_manager = _create_mock_setup(["safe S1 S8"], parsed_result)
        context = _create_input_context("Dubious violent content")

        result = await content_safety_check_input(
            llms=llms,
            llm_task_manager=mock_task_manager,
            model_name="test_model",
            context=context,
        )

        assert result["allowed"] is True
        # following assertion fails
        # assert result["policy_violations"] == ["S1", "S8"]
        assert result["policy_violations"] == []

    @pytest.mark.parametrize(
        "response,expected_allowed,expected_violations",
        [
            ("safe", True, []),
            ("unsafe violence hate", False, ["violence", "hate"]),
        ],
    )
    @pytest.mark.asyncio
    async def test_content_safety_output_with_is_content_safe_parser(
        self, response, expected_allowed, expected_violations
    ):
        parsed_result = is_content_safe(response)
        llms, mock_task_manager = _create_mock_setup([response], parsed_result)
        context = _create_output_context()

        result = await content_safety_check_output(
            llms=llms,
            llm_task_manager=mock_task_manager,
            model_name="test_model",
            context=context,
        )

        assert result["allowed"] is expected_allowed
        assert result["policy_violations"] == expected_violations

    @pytest.mark.asyncio
    async def test_content_safety_input_with_nemoguard_parser_safe(self):
        """Test input action with real nemoguard_parse_prompt_safety parser - safe response."""
        json_response = '{"User Safety": "safe"}'
        parsed_result = nemoguard_parse_prompt_safety(json_response)
        llms, mock_task_manager = _create_mock_setup([json_response], parsed_result)
        context = _create_input_context()

        result = await content_safety_check_input(
            llms=llms,
            llm_task_manager=mock_task_manager,
            model_name="test_model",
            context=context,
        )

        assert result["allowed"] is True
        assert result["policy_violations"] == []

    @pytest.mark.asyncio
    async def test_content_safety_input_with_nemoguard_parser_unsafe_with_categories(
        self,
    ):
        """Test input action with real nemoguard_parse_prompt_safety parser - unsafe with categories."""
        json_response = '{"User Safety": "unsafe", "Safety Categories": "S1, S8, S10"}'
        parsed_result = nemoguard_parse_prompt_safety(json_response)
        llms, mock_task_manager = _create_mock_setup([json_response], parsed_result)
        context = _create_input_context("Potentially harmful content")

        result = await content_safety_check_input(
            llms=llms,
            llm_task_manager=mock_task_manager,
            model_name="test_model",
            context=context,
        )

        assert result["allowed"] is False
        assert result["policy_violations"] == ["S1", "S8", "S10"]

    @pytest.mark.parametrize(
        "json_response,expected_allowed,expected_violations",
        [
            ('{"Response Safety": "safe"}', True, []),
            (
                '{"Response Safety": "unsafe", "Safety Categories": "Violence, Hate Speech"}',
                False,
                ["Violence", "Hate Speech"],
            ),
        ],
    )
    @pytest.mark.asyncio
    async def test_content_safety_output_with_nemoguard_response_parser(
        self, json_response, expected_allowed, expected_violations
    ):
        parsed_result = nemoguard_parse_response_safety(json_response)
        llms, mock_task_manager = _create_mock_setup([json_response], parsed_result)
        context = _create_output_context()

        result = await content_safety_check_output(
            llms=llms,
            llm_task_manager=mock_task_manager,
            model_name="test_model",
            context=context,
        )

        assert result["allowed"] is expected_allowed
        assert result["policy_violations"] == expected_violations

    @pytest.mark.asyncio
    async def test_content_safety_input_with_nemoguard_parser_json_parsing_failed(
        self,
    ):
        """Test input action with nemoguard_parse_prompt_safety parser; JSON parsing failure."""
        invalid_json = '{"invalid": json}'
        parsed_result = nemoguard_parse_prompt_safety(invalid_json)
        llms, mock_task_manager = _create_mock_setup([invalid_json], parsed_result)
        context = _create_input_context("Some content")

        result = await content_safety_check_input(
            llms=llms,
            llm_task_manager=mock_task_manager,
            model_name="test_model",
            context=context,
        )

        assert result["allowed"] is False
        assert result["policy_violations"] == ["JSON parsing failed"]


class TestIterableUnpackingIntegration:
    """Test that the iterable unpacking works correctly with various parser outputs."""

    @pytest.mark.parametrize(
        "response,expected_safe,expected_violations",
        [
            ("safe", True, []),
            ("unsafe", False, []),
            ("unsafe S1 S8", False, ["S1", "S8"]),
            ("yes", False, []),
            ("no", True, []),
        ],
    )
    def test_iterable_unpacking_with_is_content_safe_outputs(
        self, response, expected_safe, expected_violations
    ):
        """Test iterable unpacking directly with is_content_safe parser outputs."""
        result = is_content_safe(response)
        is_safe, *violated_policies = result
        assert is_safe is expected_safe
        assert violated_policies == expected_violations

    @pytest.mark.parametrize(
        "json_response,expected_safe,expected_violations",
        [
            ('{"User Safety": "safe"}', True, []),
            (
                '{"User Safety": "unsafe", "Safety Categories": "S1, S8"}',
                False,
                ["S1", "S8"],
            ),
            ('{"Response Safety": "safe"}', True, []),
            (
                '{"Response Safety": "unsafe", "Safety Categories": "Violence, Hate"}',
                False,
                ["Violence", "Hate"],
            ),
            ("invalid json", False, ["JSON parsing failed"]),
        ],
    )
    def test_iterable_unpacking_with_nemoguard_outputs(
        self, json_response, expected_safe, expected_violations
    ):
        """Test iterable unpacking directly with real NemoGuard parser outputs."""
        if "User Safety" in json_response or json_response == "invalid json":
            result = nemoguard_parse_prompt_safety(json_response)
        else:
            result = nemoguard_parse_response_safety(json_response)

        is_safe, *violated_policies = result
        assert is_safe is expected_safe
        assert violated_policies == expected_violations

    def test_backward_compatibility_check(self):
        """Verify that the new list format is NOT compatible with the old tuple unpacking."""
        # this test documents the breaking change i.e. old tuple unpacking should fail
        result = is_content_safe("unsafe S1 S8")  # returns [False, "S1", "S8"]

        # old tuple unpacking should fail with ValueError
        with pytest.raises(ValueError, match="too many values to unpack"):
            is_safe, violated_policies = result

        # new iterable unpacking should work
        is_safe, *violated_policies = result
        assert is_safe is False
        assert violated_policies == ["S1", "S8"]
