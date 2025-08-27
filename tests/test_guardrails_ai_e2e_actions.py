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

"""End-to-End tests for Guardrails AI integration with real validators.

These tests run against actual Guardrails validators when installed.
They can be skipped in CI/environments where validators aren't available.
"""

import pytest

GUARDRAILS_AVAILABLE = False
VALIDATORS_AVAILABLE = {}

try:
    from guardrails import Guard

    GUARDRAILS_AVAILABLE = True

    try:
        from guardrails.hub import ToxicLanguage

        VALIDATORS_AVAILABLE["toxic_language"] = True
    except ImportError:
        VALIDATORS_AVAILABLE["toxic_language"] = False

    try:
        from guardrails.hub import RegexMatch

        VALIDATORS_AVAILABLE["regex_match"] = True
    except ImportError:
        VALIDATORS_AVAILABLE["regex_match"] = False

    try:
        from guardrails.hub import ValidLength

        VALIDATORS_AVAILABLE["valid_length"] = True
    except ImportError:
        VALIDATORS_AVAILABLE["valid_length"] = False

    try:
        from guardrails.hub import CompetitorCheck

        VALIDATORS_AVAILABLE["competitor_check"] = True
    except ImportError:
        VALIDATORS_AVAILABLE["competitor_check"] = False

except ImportError:
    GUARDRAILS_AVAILABLE = False


class TestGuardrailsAIE2EIntegration:
    """End-to-End tests using real Guardrails validators when available."""

    @pytest.mark.skipif(
        not GUARDRAILS_AVAILABLE or not VALIDATORS_AVAILABLE.get("regex_match", False),
        reason="Guardrails or RegexMatch validator not installed. Install with: guardrails hub install hub://guardrails/regex_match",
    )
    def test_regex_match_e2e_success(self):
        """E2E test: RegexMatch validator with text that should pass."""
        from nemoguardrails.library.guardrails_ai.actions import validate_guardrails_ai

        result = validate_guardrails_ai(
            validator_name="regex_match",
            text="Hello world",
            regex="^[A-Z].*",
            on_fail="noop",
        )

        assert "validation_result" in result
        assert hasattr(result["validation_result"], "validation_passed")
        assert result["validation_result"].validation_passed is True

    @pytest.mark.skipif(
        not GUARDRAILS_AVAILABLE or not VALIDATORS_AVAILABLE.get("regex_match", False),
        reason="Guardrails or RegexMatch validator not installed",
    )
    def test_regex_match_e2e_failure(self):
        """E2E test: RegexMatch validator with text that should fail."""
        from nemoguardrails.library.guardrails_ai.actions import validate_guardrails_ai

        result = validate_guardrails_ai(
            validator_name="regex_match",
            text="hello world",
            regex="^[A-Z].*",
            on_fail="noop",
        )

        assert "validation_result" in result
        assert hasattr(result["validation_result"], "validation_passed")
        assert result["validation_result"].validation_passed is False

    @pytest.mark.skipif(
        not GUARDRAILS_AVAILABLE or not VALIDATORS_AVAILABLE.get("valid_length", False),
        reason="Guardrails or ValidLength validator not installed",
    )
    def test_valid_length_e2e(self):
        """E2E test: ValidLength validator."""
        from nemoguardrails.library.guardrails_ai.actions import validate_guardrails_ai

        result_pass = validate_guardrails_ai(
            validator_name="valid_length", text="Hello", min=1, max=10, on_fail="noop"
        )

        assert result_pass["validation_result"].validation_passed is True

        result_fail = validate_guardrails_ai(
            validator_name="valid_length",
            text="This is a very long text that exceeds the maximum length",
            min=1,
            max=10,
            on_fail="noop",
        )

        assert result_fail["validation_result"].validation_passed is False

    @pytest.mark.skipif(
        not GUARDRAILS_AVAILABLE
        or not VALIDATORS_AVAILABLE.get("toxic_language", False),
        reason="Guardrails or ToxicLanguage validator not installed. Install with: guardrails hub install hub://guardrails/toxic_language",
    )
    def test_toxic_language_e2e(self):
        """E2E test: ToxicLanguage validator with real content."""
        from nemoguardrails.library.guardrails_ai.actions import validate_guardrails_ai

        result_safe = validate_guardrails_ai(
            validator_name="toxic_language",
            text="Have a wonderful day! Thank you for your help.",
            threshold=0.5,
            on_fail="noop",
        )

        assert "validation_result" in result_safe
        assert hasattr(result_safe["validation_result"], "validation_passed")
        assert result_safe["validation_result"].validation_passed is True

    @pytest.mark.skipif(
        not GUARDRAILS_AVAILABLE
        or not VALIDATORS_AVAILABLE.get("competitor_check", False),
        reason="Guardrails or CompetitorCheck validator not installed",
    )
    def test_competitor_check_e2e(self):
        """E2E test: CompetitorCheck validator."""
        from nemoguardrails.library.guardrails_ai.actions import validate_guardrails_ai

        competitors = ["Apple", "Google", "Microsoft"]

        result_safe = validate_guardrails_ai(
            validator_name="competitor_check",
            text="Our company provides excellent services.",
            competitors=competitors,
            on_fail="noop",
        )

        assert result_safe["validation_result"].validation_passed is True

        result_competitor = validate_guardrails_ai(
            validator_name="competitor_check",
            text="Apple makes great products.",
            competitors=competitors,
            on_fail="noop",
        )

        assert result_competitor["validation_result"].validation_passed is False

    @pytest.mark.skipif(not GUARDRAILS_AVAILABLE, reason="Guardrails not installed")
    def test_validation_mapping_e2e(self):
        """E2E test: Validation mapping with real validation results."""
        from nemoguardrails.library.guardrails_ai.actions import (
            guardrails_ai_validation_mapping,
            validate_guardrails_ai,
        )

        if VALIDATORS_AVAILABLE.get("regex_match", False):
            result = validate_guardrails_ai(
                validator_name="regex_match",
                text="Hello world",
                regex="^[A-Z].*",
                on_fail="noop",
            )

            mapped = guardrails_ai_validation_mapping(result)
            assert mapped["valid"] is True
            assert "validation_result" in mapped

            result_fail = validate_guardrails_ai(
                validator_name="regex_match",
                text="hello world",
                regex="^[A-Z].*",
                on_fail="noop",
            )

            mapped_fail = guardrails_ai_validation_mapping(result_fail)
            assert mapped_fail["valid"] is False

    @pytest.mark.skipif(not GUARDRAILS_AVAILABLE, reason="Guardrails not installed")
    def test_metadata_parameter_e2e(self):
        """E2E test: Metadata parameter handling with real validators."""
        from nemoguardrails.library.guardrails_ai.actions import validate_guardrails_ai

        if VALIDATORS_AVAILABLE.get("regex_match", False):
            metadata = {"source": "user_input", "context": "test"}
            result = validate_guardrails_ai(
                validator_name="regex_match",
                text="Hello world",
                regex="^[A-Z].*",
                metadata=metadata,
                on_fail="noop",
            )

            assert "validation_result" in result
            assert result["validation_result"].validation_passed is True

    @pytest.mark.skipif(not GUARDRAILS_AVAILABLE, reason="Guardrails not installed")
    def test_guard_caching_e2e(self):
        """E2E test: Verify guard caching works with real validators."""
        from nemoguardrails.library.guardrails_ai.actions import _get_guard

        if VALIDATORS_AVAILABLE.get("regex_match", False):
            import nemoguardrails.library.guardrails_ai.actions as actions

            actions._guard_cache.clear()

            guard1 = _get_guard("regex_match", regex="^[A-Z].*", on_fail="noop")
            guard2 = _get_guard("regex_match", regex="^[A-Z].*", on_fail="noop")

            # should be the same instance (cached)
            assert guard1 is guard2

            # different parameters should create different guard
            guard3 = _get_guard("regex_match", regex="^[a-z].*", on_fail="noop")
            assert guard3 is not guard1

    def test_error_handling_unknown_validator_e2e(self):
        """E2E test: Error handling for unknown validators."""
        from nemoguardrails.library.guardrails_ai.actions import validate_guardrails_ai
        from nemoguardrails.library.guardrails_ai.errors import (
            GuardrailsAIValidationError,
        )

        # Test with completely unknown validator
        with pytest.raises(GuardrailsAIValidationError) as exc_info:
            validate_guardrails_ai(
                validator_name="completely_unknown_validator", text="Test text"
            )

        assert "Validation failed" in str(exc_info.value)

    @pytest.mark.skipif(not GUARDRAILS_AVAILABLE, reason="Guardrails not installed")
    def test_multiple_validators_sequence_e2e(self):
        """E2E test: Using multiple validators in sequence."""
        from nemoguardrails.library.guardrails_ai.actions import validate_guardrails_ai

        test_text = "Hello World Test"

        available_validators = []
        if VALIDATORS_AVAILABLE.get("regex_match", False):
            available_validators.append(("regex_match", {"regex": "^[A-Z].*"}))
        if VALIDATORS_AVAILABLE.get("valid_length", False):
            available_validators.append(("valid_length", {"min": 1, "max": 50}))

        # run each available validator
        for validator_name, params in available_validators:
            result = validate_guardrails_ai(
                validator_name=validator_name, text=test_text, on_fail="noop", **params
            )

            assert "validation_result" in result
            assert hasattr(result["validation_result"], "validation_passed")
            # all should pass with the test text
            assert result["validation_result"].validation_passed is True


def print_validator_availability():
    """Helper function to print which validators are available for testing."""
    print(f"Guardrails available: {GUARDRAILS_AVAILABLE}")
    if GUARDRAILS_AVAILABLE:
        for validator, available in VALIDATORS_AVAILABLE.items():
            print(f"  {validator}: {available}")


if __name__ == "__main__":
    print_validator_availability()
    pytest.main([__file__, "-v", "-s"])
