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

"""Tests for Guardrails AI integration - updated to match current implementation."""

import inspect
from unittest.mock import Mock, patch

import pytest


class TestGuardrailsAIIntegration:
    """Test suite for Guardrails AI integration with current implementation."""

    def test_module_imports_without_guardrails(self):
        """Test that modules can be imported even without guardrails package."""
        from nemoguardrails.library.guardrails_ai.actions import (
            guardrails_ai_validation_mapping,
            validate_guardrails_ai,
        )
        from nemoguardrails.library.guardrails_ai.registry import VALIDATOR_REGISTRY

        assert callable(validate_guardrails_ai)
        assert callable(guardrails_ai_validation_mapping)
        assert isinstance(VALIDATOR_REGISTRY, dict)

    def test_validator_registry_structure(self):
        """Test that the validator registry has the expected structure."""
        from nemoguardrails.library.guardrails_ai.registry import VALIDATOR_REGISTRY

        assert isinstance(VALIDATOR_REGISTRY, dict)
        assert len(VALIDATOR_REGISTRY) >= 6

        expected_validators = [
            "toxic_language",
            "detect_jailbreak",
            "guardrails_pii",
            "competitor_check",
            "restricttotopic",
            "provenance_llm",
        ]

        for validator in expected_validators:
            assert validator in VALIDATOR_REGISTRY
            validator_info = VALIDATOR_REGISTRY[validator]
            assert "module" in validator_info
            assert "class" in validator_info
            assert "hub_path" in validator_info
            assert "default_params" in validator_info
            assert isinstance(validator_info["default_params"], dict)

    def test_validation_mapping_function(self):
        """Test the validation mapping function with current interface."""
        from nemoguardrails.library.guardrails_ai.actions import (
            guardrails_ai_validation_mapping,
        )

        mock_result = Mock()
        mock_result.validation_passed = True
        result1 = {"validation_result": mock_result}
        mapped1 = guardrails_ai_validation_mapping(result1)
        assert mapped1 is True

        mock_result2 = Mock()
        mock_result2.validation_passed = False
        result2 = {"validation_result": mock_result2}
        mapped2 = guardrails_ai_validation_mapping(result2)
        assert mapped2 is False

        result3 = {"validation_result": {"validation_passed": True}}
        mapped3 = guardrails_ai_validation_mapping(result3)
        assert mapped3 is True

    @patch("nemoguardrails.library.guardrails_ai.actions._get_guard")
    def test_validate_guardrails_ai_success(self, mock_get_guard):
        """Test successful validation with current interface."""
        from nemoguardrails.library.guardrails_ai.actions import validate_guardrails_ai

        mock_guard = Mock()
        mock_validation_result = Mock()
        mock_validation_result.validation_passed = True
        mock_guard.validate.return_value = mock_validation_result
        mock_get_guard.return_value = mock_guard

        result = validate_guardrails_ai(
            validator_name="toxic_language",
            text="Hello, this is a safe message",
            threshold=0.5,
        )

        assert "validation_result" in result
        assert result["validation_result"] == mock_validation_result
        mock_guard.validate.assert_called_once_with("Hello, this is a safe message", metadata={})
        mock_get_guard.assert_called_once_with("toxic_language", threshold=0.5)

    @patch("nemoguardrails.library.guardrails_ai.actions._get_guard")
    def test_validate_guardrails_ai_with_metadata(self, mock_get_guard):
        """Test validation with metadata parameter."""
        from nemoguardrails.library.guardrails_ai.actions import validate_guardrails_ai

        mock_guard = Mock()
        mock_validation_result = Mock()
        mock_validation_result.validation_passed = False
        mock_guard.validate.return_value = mock_validation_result
        mock_get_guard.return_value = mock_guard

        metadata = {"source": "user_input"}
        result = validate_guardrails_ai(
            validator_name="detect_jailbreak",
            text="Some text",
            metadata=metadata,
            threshold=0.8,
        )

        assert "validation_result" in result
        assert result["validation_result"] == mock_validation_result
        mock_guard.validate.assert_called_once_with("Some text", metadata=metadata)
        mock_get_guard.assert_called_once_with("detect_jailbreak", threshold=0.8)

    @patch("nemoguardrails.library.guardrails_ai.actions._get_guard")
    def test_validate_guardrails_ai_error_handling(self, mock_get_guard):
        """Test error handling in validation."""
        from nemoguardrails.library.guardrails_ai.actions import validate_guardrails_ai
        from nemoguardrails.library.guardrails_ai.errors import (
            GuardrailsAIValidationError,
        )

        mock_guard = Mock()
        mock_guard.validate.side_effect = Exception("Validation service error")
        mock_get_guard.return_value = mock_guard

        with pytest.raises(GuardrailsAIValidationError) as exc_info:
            validate_guardrails_ai(validator_name="toxic_language", text="Any text")

        assert "Validation failed" in str(exc_info.value)
        assert "Validation service error" in str(exc_info.value)

    @patch("nemoguardrails.library.guardrails_ai.actions._load_validator_class")
    @patch("nemoguardrails.library.guardrails_ai.actions.Guard")
    def test_get_guard_creates_and_caches(self, mock_guard_class, mock_load_validator):
        """Test that _get_guard creates and caches guards properly."""
        from nemoguardrails.library.guardrails_ai.actions import _get_guard

        mock_validator_class = Mock()
        mock_validator_instance = Mock()
        mock_guard_instance = Mock()
        mock_guard = Mock()

        mock_load_validator.return_value = mock_validator_class
        mock_validator_class.return_value = mock_validator_instance
        mock_guard_class.return_value = mock_guard
        mock_guard.use.return_value = mock_guard_instance

        # clear cache
        import nemoguardrails.library.guardrails_ai.actions as actions

        actions._guard_cache.clear()

        # first call should create new guard
        result1 = _get_guard("toxic_language", threshold=0.5)

        assert result1 == mock_guard_instance
        mock_validator_class.assert_called_once_with(threshold=0.5, on_fail="noop")
        mock_guard.use.assert_called_once_with(mock_validator_instance)

        # reset mocks for second call
        mock_load_validator.reset_mock()
        mock_validator_class.reset_mock()
        mock_guard_class.reset_mock()

        # second call with same params should use cache
        result2 = _get_guard("toxic_language", threshold=0.5)

        assert result2 == mock_guard_instance
        # should not create new validator or guard
        mock_load_validator.assert_not_called()
        mock_validator_class.assert_not_called()
        mock_guard_class.assert_not_called()

    @patch("nemoguardrails.library.guardrails_ai.registry.get_validator_info")
    def test_load_validator_class_unknown_validator(self, mock_get_info):
        """Test error handling for unknown validators."""
        from nemoguardrails.library.guardrails_ai.actions import _load_validator_class
        from nemoguardrails.library.guardrails_ai.errors import GuardrailsAIConfigError

        mock_get_info.side_effect = GuardrailsAIConfigError("Unknown validator: unknown_validator")

        with pytest.raises(ImportError) as exc_info:
            _load_validator_class("unknown_validator")

        assert "Failed to load validator unknown_validator" in str(exc_info.value)

    def test_validate_guardrails_ai_signature(self):
        """Test that validate_guardrails_ai has the expected signature."""
        from nemoguardrails.library.guardrails_ai.actions import validate_guardrails_ai

        sig = inspect.signature(validate_guardrails_ai)
        params = list(sig.parameters.keys())

        assert "validator_name" in params
        assert "text" in params
        assert any(param.kind == param.VAR_KEYWORD for param in sig.parameters.values())

    @patch("nemoguardrails.library.guardrails_ai.actions._load_validator_class")
    @patch("nemoguardrails.library.guardrails_ai.actions.Guard")
    def test_guard_cache_key_generation(self, mock_guard_class, mock_load):
        """Test that guard cache keys are generated correctly for different parameter combinations."""
        from nemoguardrails.library.guardrails_ai.actions import _get_guard

        mock_validator_class = Mock()
        mock_guard_instance = Mock()
        mock_guard = Mock()

        mock_load.return_value = mock_validator_class
        mock_guard_class.return_value = mock_guard
        mock_guard.use.return_value = mock_guard_instance

        import nemoguardrails.library.guardrails_ai.actions as actions

        actions._guard_cache.clear()

        # create guards with different parameters
        _get_guard("toxic_language", threshold=0.5)
        _get_guard("toxic_language", threshold=0.8)
        _get_guard("detect_jailbreak", threshold=0.5)

        assert len(actions._guard_cache) == 3
