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


from nemoguardrails.llm.output_parsers import (
    is_content_safe,
    nemoguard_parse_prompt_safety,
    nemoguard_parse_response_safety,
)


class TestIsContentSafe:
    """Test the is_content_safe output parser used in content safety actions."""

    def test_safe_content(self):
        """Test parsing safe content."""
        response = "safe"
        is_safe, *violated_policies = is_content_safe(response)
        assert is_safe is True
        assert violated_policies == []

    def test_unsafe_content(self):
        """Test parsing unsafe content."""
        response = "unsafe"
        is_safe, *violated_policies = is_content_safe(response)
        assert is_safe is False
        assert violated_policies == []

    def test_unsafe_content_with_violations(self):
        """Test parsing unsafe content with violation categories."""
        response = "unsafe violence hate speech"
        is_safe, *violated_policies = is_content_safe(response)
        assert is_safe is False
        assert "violence" in violated_policies
        assert "hate" in violated_policies
        assert "speech" in violated_policies

    def test_yes_response_blocks_content(self):
        """Test parsing 'yes' response (content should be blocked)."""
        response = "yes"
        is_safe, *violated_policies = is_content_safe(response)
        assert is_safe is False
        assert violated_policies == []

    def test_no_response_allows_content(self):
        """Test parsing 'no' response (content is safe)."""
        response = "no"
        is_safe, *violated_policies = is_content_safe(response)
        assert is_safe is True
        assert violated_policies == []

    def test_case_insensitive_parsing(self):
        """Test parsing is case insensitive."""
        response = "UNSAFE"
        is_safe, *violated_policies = is_content_safe(response)
        assert is_safe is False
        assert violated_policies == []

    def test_response_with_extra_whitespace(self):
        """Test parsing response with extra whitespace."""
        response = "  safe  "
        is_safe, *violated_policies = is_content_safe(response)
        assert is_safe is True
        assert violated_policies == []

    def test_response_with_punctuation(self):
        """Test parsing response with punctuation gets normalized."""
        response = "safe!!!"
        is_safe, *violated_policies = is_content_safe(response)
        assert is_safe is True
        assert violated_policies == []

    def test_unknown_response_defaults_to_unsafe(self):
        """Test that unknown responses default to unsafe."""
        response = "unknown_response"
        is_safe, *violated_policies = is_content_safe(response)
        assert is_safe is False
        assert violated_policies == []

    def test_empty_response_defaults_to_unsafe(self):
        """Test that empty responses default to unsafe."""
        response = ""
        is_safe, *violated_policies = is_content_safe(response)
        assert is_safe is False
        assert violated_policies == []

    def test_llama_guard_safe_response(self):
        """Test LlamaGuard style safe response."""
        response = "safe"
        is_safe, *violated_policies = is_content_safe(response)
        assert is_safe is True
        assert violated_policies == []

    def test_llama_guard_unsafe_with_categories(self):
        """Test LlamaGuard style unsafe response with categories."""
        response = "unsafe S1 S8"
        is_safe, *violated_policies = is_content_safe(response)
        assert is_safe is False
        assert "S1" in violated_policies
        assert "S8" in violated_policies


class TestNemoguardParsePromptSafety:
    """Test the nemoguard_parse_prompt_safety JSON output parser."""

    def test_safe_user_content_json(self):
        """Test parsing safe user content from JSON."""
        response = '{"User Safety": "safe"}'
        is_safe, *violated_policies = nemoguard_parse_prompt_safety(response)
        assert is_safe is True
        assert violated_policies == []

    def test_unsafe_user_content_json(self):
        """Test parsing unsafe user content from JSON."""
        response = '{"User Safety": "unsafe"}'
        is_safe, *violated_policies = nemoguard_parse_prompt_safety(response)
        assert is_safe is False
        assert violated_policies == []

    def test_unsafe_with_safety_categories(self):
        """Test parsing unsafe content with safety categories."""
        response = '{"User Safety": "unsafe", "Safety Categories": "S1, S8, S10"}'
        is_safe, *violated_policies = nemoguard_parse_prompt_safety(response)
        assert is_safe is False
        assert "S1" in violated_policies
        assert "S8" in violated_policies
        assert "S10" in violated_policies

    def test_case_insensitive_safety_status(self):
        """Test parsing is case insensitive for safety status."""
        response = '{"User Safety": "SAFE"}'
        is_safe, *violated_policies = nemoguard_parse_prompt_safety(response)
        assert is_safe is True
        assert violated_policies == []

    def test_categories_with_whitespace_trimming(self):
        """Test parsing categories with extra whitespace gets trimmed."""
        response = '{"User Safety": "unsafe", "Safety Categories": " S1 , S8 , S10 "}'
        is_safe, *violated_policies = nemoguard_parse_prompt_safety(response)
        assert is_safe is False
        assert "S1" in violated_policies
        assert "S8" in violated_policies
        assert "S10" in violated_policies

    def test_empty_safety_categories(self):
        """Test parsing with empty safety categories string."""
        response = '{"User Safety": "unsafe", "Safety Categories": ""}'
        is_safe, *violated_policies = nemoguard_parse_prompt_safety(response)
        assert is_safe is False
        assert violated_policies == [""]

    def test_missing_safety_categories_field(self):
        """Test parsing when Safety Categories field is missing."""
        response = '{"User Safety": "unsafe"}'
        is_safe, *violated_policies = nemoguard_parse_prompt_safety(response)
        assert is_safe is False
        assert violated_policies == []

    def test_invalid_json_defaults_to_unsafe(self):
        """Test that invalid JSON defaults to unsafe with error message."""
        response = '{"invalid": json}'
        is_safe, *violated_policies = nemoguard_parse_prompt_safety(response)
        assert is_safe is False
        assert violated_policies == ["JSON parsing failed"]

    def test_missing_user_safety_field(self):
        """Test parsing when User Safety field is missing."""
        response = '{"Response Safety": "safe"}'
        is_safe, *violated_policies = nemoguard_parse_prompt_safety(response)
        assert is_safe is False
        assert violated_policies == ["JSON parsing failed"]

    def test_single_category(self):
        """Test parsing with single safety category."""
        response = '{"User Safety": "unsafe", "Safety Categories": "Violence"}'
        is_safe, *violated_policies = nemoguard_parse_prompt_safety(response)
        assert is_safe is False
        assert violated_policies == ["Violence"]

    def test_complex_category_names(self):
        """Test parsing with descriptive category names."""
        response = '{"User Safety": "unsafe", "Safety Categories": "Violence, Hate Speech, Sexual Content"}'
        is_safe, *violated_policies = nemoguard_parse_prompt_safety(response)
        assert is_safe is False
        assert "Violence" in violated_policies
        assert "Hate Speech" in violated_policies
        assert "Sexual Content" in violated_policies


class TestNemoguardParseResponseSafety:
    """Test the nemoguard_parse_response_safety JSON output parser."""

    def test_safe_response_content_json(self):
        """Test parsing safe response content from JSON."""
        response = '{"Response Safety": "safe"}'
        is_safe, *violated_policies = nemoguard_parse_response_safety(response)
        assert is_safe is True
        assert violated_policies == []

    def test_unsafe_response_content_json(self):
        """Test parsing unsafe response content from JSON."""
        response = '{"Response Safety": "unsafe"}'
        is_safe, *violated_policies = nemoguard_parse_response_safety(response)
        assert is_safe is False
        assert violated_policies == []

    def test_unsafe_with_safety_categories(self):
        """Test parsing unsafe response with safety categories."""
        response = '{"Response Safety": "unsafe", "Safety Categories": "S1, S8, S10"}'
        is_safe, *violated_policies = nemoguard_parse_response_safety(response)
        assert is_safe is False
        assert "S1" in violated_policies
        assert "S8" in violated_policies
        assert "S10" in violated_policies

    def test_case_insensitive_safety_status(self):
        """Test parsing is case insensitive for safety status."""
        response = '{"Response Safety": "SAFE"}'
        is_safe, *violated_policies = nemoguard_parse_response_safety(response)
        assert is_safe is True
        assert violated_policies == []

    def test_categories_with_whitespace_trimming(self):
        """Test parsing categories with extra whitespace gets trimmed."""
        response = (
            '{"Response Safety": "unsafe", "Safety Categories": " S1 , S8 , S10 "}'
        )
        is_safe, *violated_policies = nemoguard_parse_response_safety(response)
        assert is_safe is False
        assert "S1" in violated_policies
        assert "S8" in violated_policies
        assert "S10" in violated_policies

    def test_missing_safety_categories_field(self):
        """Test parsing when Safety Categories field is missing."""
        response = '{"Response Safety": "unsafe"}'
        is_safe, *violated_policies = nemoguard_parse_response_safety(response)
        assert is_safe is False
        assert violated_policies == []

    def test_invalid_json_defaults_to_unsafe(self):
        """Test that invalid JSON defaults to unsafe with error message."""
        response = '{"invalid": json}'
        is_safe, *violated_policies = nemoguard_parse_response_safety(response)
        assert is_safe is False
        assert violated_policies == ["JSON parsing failed"]

    def test_missing_response_safety_field(self):
        """Test parsing when Response Safety field is missing."""
        response = '{"User Safety": "safe"}'
        is_safe, *violated_policies = nemoguard_parse_response_safety(response)
        assert is_safe is False
        assert violated_policies == ["JSON parsing failed"]

    def test_full_nemoguard_response(self):
        """Test parsing a full NemoGuard response with both user and response safety."""
        response = '{"User Safety": "safe", "Response Safety": "unsafe", "Safety Categories": "S1, S8"}'
        is_safe, *violated_policies = nemoguard_parse_response_safety(response)
        assert is_safe is False
        assert "S1" in violated_policies
        assert "S8" in violated_policies

    def test_malformed_json_with_extra_text(self):
        """Test parsing malformed JSON with extra characters."""
        response = '{"Response Safety": "unsafe", "Safety Categories": "S1"} extra text'
        is_safe, *violated_policies = nemoguard_parse_response_safety(response)
        assert is_safe is False
        assert violated_policies == ["JSON parsing failed"]


class TestOutputParsersRealWorldScenarios:
    """Test output parsers with real-world LLM response scenarios."""

    def test_llama_guard_typical_responses(self):
        """Test typical LlamaGuard responses."""
        response = "safe"
        is_safe, *violated_policies = is_content_safe(response)
        assert is_safe is True
        assert violated_policies == []

        response = "unsafe S1 S8"
        is_safe, *violated_policies = is_content_safe(response)
        assert is_safe is False

    def test_nemoguard_content_safety_responses(self):
        """Test typical NemoGuard ContentSafety model responses."""
        response = '{"User Safety": "unsafe", "Safety Categories": "S1: Violence, S8: Hate/Identity Hate"}'
        is_safe, *violated_policies = nemoguard_parse_prompt_safety(response)
        assert is_safe is False
        assert "S1: Violence" in violated_policies
        assert "S8: Hate/Identity Hate" in violated_policies

        response = '{"User Safety": "safe", "Response Safety": "unsafe", "Safety Categories": "S11: Sexual Content"}'
        is_safe, *violated_policies = nemoguard_parse_response_safety(response)
        assert is_safe is False
        assert violated_policies == ["S11: Sexual Content"]

    def test_edge_case_llm_responses(self):
        """Test edge cases in LLM responses."""
        response = "Let me think about this... The content appears to be safe"
        is_safe, *violated_policies = is_content_safe(response)
        assert is_safe is False
        assert violated_policies == []

        response = "**UNSAFE**"
        is_safe, *violated_policies = is_content_safe(response)
        assert is_safe is False
        assert violated_policies == []

    def test_parser_robustness(self):
        """Test parser robustness with various edge cases."""
        invalid_response = "The model refused to answer"

        is_safe, *violated_policies = is_content_safe(invalid_response)
        assert is_safe is False

        is_safe, *violated_policies = nemoguard_parse_prompt_safety(invalid_response)
        assert is_safe is False
        assert violated_policies == ["JSON parsing failed"]

        is_safe, *violated_policies = nemoguard_parse_response_safety(invalid_response)
        assert is_safe is False
        assert violated_policies == ["JSON parsing failed"]

    def test_starred_unpacking_compatibility(self):
        """Test that parser outputs are compatible with starred unpacking logic."""

        response = "safe"
        result = is_content_safe(response)
        is_safe, *violated_policies = result
        assert is_safe is True
        assert violated_policies == []

        response = "unsafe violence hate"
        result = is_content_safe(response)
        is_safe, *violated_policies = result
        assert is_safe is False
        assert len(violated_policies) > 0
        assert "violence" in violated_policies
        assert "hate" in violated_policies

        response = '{"User Safety": "safe"}'
        result = nemoguard_parse_prompt_safety(response)
        is_safe, *violated_policies = result
        assert is_safe is True
        assert violated_policies == []

        response = '{"Response Safety": "unsafe", "Safety Categories": "S1, S8"}'
        result = nemoguard_parse_response_safety(response)
        is_safe, *violated_policies = result
        assert is_safe is False
        assert len(violated_policies) > 0
        assert "S1" in violated_policies
        assert "S8" in violated_policies
