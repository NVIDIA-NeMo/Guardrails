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

"""Tests for LLM isolation functionality in LLMRails."""

import inspect
from typing import Optional
from unittest.mock import Mock, patch

import pytest
from pydantic import BaseModel

from nemoguardrails.rails.llm.config import RailsConfig
from nemoguardrails.rails.llm.llmrails import LLMRails


class MockLLM(BaseModel):
    """Mock LLM for testing purposes."""

    model_config = {"extra": "allow"}

    model_kwargs: dict = {}
    temperature: float = 0.7
    max_tokens: Optional[int] = None


class MockActionDispatcher:
    """Mock action dispatcher for testing."""

    def __init__(self):
        self.registered_actions = {
            "action_with_llm": self._mock_action_with_llm,
            "action_without_llm": self._mock_action_without_llm,
            "generate_user_intent": self._mock_generate_user_intent,
            "self_check_output": self._mock_self_check_output,
        }

    def _mock_action_with_llm(self, llm, context: dict):
        """Mock action that requires LLM."""
        pass

    def _mock_action_without_llm(self, context: dict, config):
        """Mock action that doesn't require LLM."""
        pass

    def _mock_generate_user_intent(self, llm: Optional[MockLLM], events: list):
        """Mock generation action with LLM."""
        pass

    def _mock_self_check_output(self, llm, max_tokens: int):
        """Mock self-check action with LLM."""
        pass


@pytest.fixture
def mock_config():
    """Create mock configuration for testing."""
    return RailsConfig.from_content(
        """
    models:
      - type: main
        engine: openai
        model: gpt-4
    """
    )


@pytest.fixture
def mock_llm():
    """Create mock LLM for testing."""
    return MockLLM(model_kwargs={"temperature": 0.7}, temperature=0.7, max_tokens=100)


@pytest.fixture
def rails_with_mock_llm(mock_config, mock_llm):
    """Create LLMRails instance with mocked initialization."""
    with patch("nemoguardrails.rails.llm.llmrails.LLMRails._init_llms"):
        rails = LLMRails(config=mock_config)
        rails.llm = mock_llm
        return rails


class TestLLMIsolation:
    """Test LLM isolation functionality."""

    def test_detect_llm_requiring_actions(self, rails_with_mock_llm):
        """Test detection of actions that require LLM."""

        rails = rails_with_mock_llm

        rails.runtime = Mock()
        rails.runtime.action_dispatcher = MockActionDispatcher()
        rails.runtime.registered_action_params = {}

        actions_needing_llms = rails._detect_llm_requiring_actions()

        expected_actions = {
            "action_with_llm",
            "generate_user_intent",
            "self_check_output",
        }
        assert actions_needing_llms == expected_actions

        assert "action_without_llm" not in actions_needing_llms

    def test_get_action_function_plain_function(self, rails_with_mock_llm):
        """Test extraction of plain function."""
        rails = rails_with_mock_llm

        def plain_function():
            pass

        assert rails._get_action_function(plain_function) == plain_function

    def test_get_action_function_decorated_function(self, rails_with_mock_llm):
        """Test extraction of @action decorated function."""
        rails = rails_with_mock_llm
        from nemoguardrails.actions import action

        @action()
        def decorated_function():
            pass

        assert rails._get_action_function(decorated_function) == decorated_function

    def test_get_action_function_callable_class(self, rails_with_mock_llm):
        """Test extraction of callable class instance."""
        rails = rails_with_mock_llm

        class ActionClass:
            def __call__(self):
                pass

        instance = ActionClass()
        assert rails._get_action_function(instance) == instance

    def test_get_action_function_non_callable(self, rails_with_mock_llm):
        """Test extraction returns None for non-callable objects."""
        rails = rails_with_mock_llm

        assert rails._get_action_function("not_callable") is None
        assert rails._get_action_function(None) is None

    def test_get_action_function_with_real_action_dispatcher(self, rails_with_mock_llm):
        """Test extraction with actual ActionDispatcher registered actions."""
        from nemoguardrails.actions import action
        from nemoguardrails.actions.action_dispatcher import ActionDispatcher

        rails = rails_with_mock_llm

        # create a ActionDispatcher
        action_dispatcher = ActionDispatcher(load_all_actions=False)

        # register some test actions directly
        def plain_action():
            return "plain"

        @action()
        def decorated_action():
            return "decorated"

        class ActionClass:
            def __call__(self):
                return "class"

        action_dispatcher.register_action(plain_action, "plain_action")
        action_dispatcher.register_action(decorated_action, "decorated_action")
        action_dispatcher.register_action(ActionClass(), "class_action")

        for action_name, action_info in action_dispatcher.registered_actions.items():
            result = rails._get_action_function(action_info)
            assert callable(result), f"Action {action_name} should return callable"
            assert (
                result is action_info
            ), f"Should return the action_info directly for {action_name}"

    def test_create_action_llm_copy(self, rails_with_mock_llm):
        """Test creation of isolated LLM copies."""
        rails = rails_with_mock_llm

        original_llm = MockLLM(
            model_kwargs={"temperature": 0.5, "max_tokens": 200},
            temperature=0.5,
            max_tokens=200,
        )

        isolated_llm = rails._create_action_llm_copy(original_llm, "test_action")

        # verify it's a different instance
        assert isolated_llm is not original_llm

        # verify model_kwargs are isolated (different dict instances)
        assert isolated_llm.model_kwargs is not original_llm.model_kwargs

        # verify initial values are copied
        assert isolated_llm.model_kwargs == original_llm.model_kwargs
        assert isolated_llm.temperature == original_llm.temperature
        assert isolated_llm.max_tokens == original_llm.max_tokens

        # verify modifications to isolated LLM don't affect original one
        isolated_llm.model_kwargs["new_param"] = "test_value"
        isolated_llm.temperature = 0.1

        assert "new_param" not in original_llm.model_kwargs
        assert original_llm.temperature == 0.5

    def test_create_action_llm_copy_with_none_model_kwargs(self, rails_with_mock_llm):
        """Test LLM copy creation when model_kwargs is None."""
        rails = rails_with_mock_llm

        original_llm = MockLLM()
        original_llm.model_kwargs = None

        isolated_llm = rails._create_action_llm_copy(original_llm, "test_action")

        assert isolated_llm.model_kwargs == {}
        assert isinstance(isolated_llm.model_kwargs, dict)

    def test_create_action_llm_copy_handles_copy_failure(self, rails_with_mock_llm):
        """Test that copy failures raise detailed error message."""
        rails = rails_with_mock_llm

        # create a mock LLM that fails to copy
        original_llm = Mock()

        with patch("copy.copy", side_effect=Exception("Copy failed")):
            with pytest.raises(RuntimeError) as exc_info:
                rails._create_action_llm_copy(original_llm, "test_action")

            error_msg = str(exc_info.value)
            # verify error message contains key information
            assert (
                "Failed to create isolated LLM instance for action 'test_action'"
                in error_msg
            )
            assert "parameter contamination" in error_msg
            assert "Possible solutions:" in error_msg
            assert "custom LLM class" in error_msg
            assert "dedicated LLM configuration" in error_msg
            assert "Copy failed" in error_msg  # original error
            assert "models:" in error_msg  # config example

    def test_create_isolated_llms_for_actions_integration(self, rails_with_mock_llm):
        """Test the full isolated LLM creation process."""
        rails = rails_with_mock_llm

        rails.runtime = Mock()
        rails.runtime.action_dispatcher = MockActionDispatcher()
        rails.runtime.registered_action_params = {}
        rails.runtime.register_action_param = Mock()

        rails._create_isolated_llms_for_actions()

        expected_calls = [
            "action_with_llm_llm",
            "generate_user_intent_llm",
            "self_check_output_llm",
        ]

        actual_calls = [
            call[0][0] for call in rails.runtime.register_action_param.call_args_list
        ]

        for expected_call in expected_calls:
            assert expected_call in actual_calls

    def test_create_isolated_llms_skips_existing_specialized_llms(
        self, rails_with_mock_llm
    ):
        """Test that existing specialized LLMs are not overridden."""
        rails = rails_with_mock_llm

        rails.runtime = Mock()
        rails.runtime.action_dispatcher = MockActionDispatcher()
        rails.runtime.registered_action_params = {"self_check_output_llm": Mock()}
        rails.runtime.register_action_param = Mock()

        rails._create_isolated_llms_for_actions()

        # verify self_check_output_llm was NOT re-registered
        actual_calls = [
            call[0][0] for call in rails.runtime.register_action_param.call_args_list
        ]
        assert "self_check_output_llm" not in actual_calls

        # but other actions should still get isolated LLMs
        assert "action_with_llm_llm" in actual_calls
        assert "generate_user_intent_llm" in actual_calls

    def test_create_isolated_llms_handles_no_main_llm(self, mock_config):
        """Test graceful handling when no main LLM is available."""
        with patch("nemoguardrails.rails.llm.llmrails.LLMRails._init_llms"):
            rails = LLMRails(config=mock_config)
            rails.llm = None  # no main LLM

        rails.runtime = Mock()
        rails.runtime.action_dispatcher = MockActionDispatcher()
        rails.runtime.registered_action_params = {}
        rails.runtime.register_action_param = Mock()

        rails._create_isolated_llms_for_actions()

        # verify no llms were registered
        rails.runtime.register_action_param.assert_not_called()

    def test_create_isolated_llms_handles_missing_action_dispatcher(
        self, rails_with_mock_llm
    ):
        """Test graceful handling when action dispatcher is not available."""
        rails = rails_with_mock_llm

        # set up runtime without action dispatcher
        rails.runtime = Mock()
        rails.runtime.action_dispatcher = None

        # should not crash
        rails._create_isolated_llms_for_actions()


class TestLLMIsolationEdgeCases:
    """Test edge cases and error scenarios."""

    def test_isolated_llm_preserves_shallow_copy_behavior(self, rails_with_mock_llm):
        """Test that isolated LLMs preserve shared resources via shallow copy."""
        rails = rails_with_mock_llm

        # create LLM with mock HTTP client
        original_llm = MockLLM(model_kwargs={"param": "value"})

        # use setattr to add dynamic attributes (bypassing Pydantic validation)
        setattr(original_llm, "http_client", Mock())  # Simulate HTTP client
        setattr(original_llm, "credentials", {"api_key": "secret"})

        isolated_llm = rails._create_action_llm_copy(original_llm, "test_action")

        # verify shared resources are preserved (shallow copy)
        assert hasattr(isolated_llm, "http_client")
        assert isolated_llm.http_client is original_llm.http_client
        assert isolated_llm.credentials is original_llm.credentials

        # but model_kwargs should be isolated
        assert isolated_llm.model_kwargs is not original_llm.model_kwargs
        assert isolated_llm.model_kwargs == original_llm.model_kwargs

    def test_multiple_isolated_llms_are_independent(self, rails_with_mock_llm):
        """Test that multiple isolated LLMs don't interfere with each other."""
        rails = rails_with_mock_llm

        original_llm = MockLLM(model_kwargs={"shared_param": "original"})

        # create multiple isolated copies
        isolated_llm_1 = rails._create_action_llm_copy(original_llm, "action_1")
        isolated_llm_2 = rails._create_action_llm_copy(original_llm, "action_2")

        # ensure they are different instances
        assert isolated_llm_1 is not isolated_llm_2
        assert isolated_llm_1.model_kwargs is not isolated_llm_2.model_kwargs

        # modify one isolated LLM
        isolated_llm_1.model_kwargs["action_1_param"] = "value_1"
        isolated_llm_1.temperature = 0.1

        # modify another isolated LLM
        isolated_llm_2.model_kwargs["action_2_param"] = "value_2"
        isolated_llm_2.temperature = 0.9

        # verify changes don't affect each other
        assert "action_1_param" not in isolated_llm_2.model_kwargs
        assert "action_2_param" not in isolated_llm_1.model_kwargs
        assert isolated_llm_1.temperature != isolated_llm_2.temperature

        # verify original is unchanged
        assert "action_1_param" not in original_llm.model_kwargs
        assert "action_2_param" not in original_llm.model_kwargs
        assert original_llm.temperature != 0.1 and original_llm.temperature != 0.9

    @pytest.mark.parametrize(
        "action_name,expected_isolated",
        [
            ("action_with_llm", True),
            ("action_without_llm", False),
            ("generate_user_intent", True),
            ("self_check_output", True),
            ("non_existent_action", False),
        ],
    )
    def test_action_detection_parametrized(
        self, rails_with_mock_llm, action_name, expected_isolated
    ):
        """Test action detection with various action names."""
        rails = rails_with_mock_llm

        rails.runtime = Mock()
        rails.runtime.action_dispatcher = MockActionDispatcher()
        rails.runtime.registered_action_params = {}

        actions_needing_llms = rails._detect_llm_requiring_actions()

        if expected_isolated:
            assert action_name in actions_needing_llms
        else:
            assert action_name not in actions_needing_llms
