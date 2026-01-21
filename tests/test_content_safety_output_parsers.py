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


from nemoguardrails.llm.output_parsers import (
    _extract_harm_value,
    _strip_think_tags,
    is_content_safe,
    nemoguard_parse_prompt_safety,
    nemoguard_parse_response_safety,
    nemotron_reasoning_parse_prompt_safety,
    nemotron_reasoning_parse_response_safety,
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
        response = '{"Response Safety": "unsafe", "Safety Categories": " S1 , S8 , S10 "}'
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


class TestStripThinkTags:
    """Test the _strip_think_tags helper function."""

    def test_no_think_tags(self):
        """Test input without think tags returns unchanged."""
        response = "Prompt harm: unharmful\nResponse Harm: unharmful"
        result = _strip_think_tags(response)
        assert result == response

    def test_single_line_think_tags(self):
        """Test stripping single-line think tags."""
        response = "<think>some reasoning</think>\nPrompt harm: harmful"
        result = _strip_think_tags(response)
        assert result == "Prompt harm: harmful"

    def test_multiline_think_tags(self):
        """Test stripping multi-line think tags."""
        response = """<think>
The user's request falls under S21 (Illegal Activity).
This is clearly harmful content.
</think>

Prompt harm: harmful
Response Harm: unharmful"""
        result = _strip_think_tags(response)
        assert "<think>" not in result
        assert "</think>" not in result
        assert "Prompt harm: harmful" in result
        assert "Response Harm: unharmful" in result

    def test_empty_think_tags(self):
        """Test stripping empty think tags."""
        response = "<think></think>Prompt harm: unharmful"
        result = _strip_think_tags(response)
        assert result == "Prompt harm: unharmful"

    def test_whitespace_handling(self):
        """Test that result is stripped of leading/trailing whitespace."""
        response = "  <think>reasoning</think>  \n  Prompt harm: unharmful  "
        result = _strip_think_tags(response)
        assert result == "Prompt harm: unharmful"


class TestExtractHarmValue:
    """Test the _extract_harm_value helper function."""

    def test_extract_harmful(self):
        """Test extracting harmful value."""
        response = "Prompt harm: harmful"
        result = _extract_harm_value(response, "Prompt harm")
        assert result == "harmful"

    def test_extract_unharmful(self):
        """Test extracting unharmful value."""
        response = "Prompt harm: unharmful"
        result = _extract_harm_value(response, "Prompt harm")
        assert result == "unharmful"

    def test_case_insensitive_field(self):
        """Test case insensitive field matching."""
        response = "PROMPT HARM: unharmful"
        result = _extract_harm_value(response, "Prompt harm")
        assert result == "unharmful"

    def test_case_insensitive_value(self):
        """Test value is lowercased."""
        response = "Prompt harm: UNHARMFUL"
        result = _extract_harm_value(response, "Prompt harm")
        assert result == "unharmful"

    def test_response_harm_field(self):
        """Test extracting Response Harm field."""
        response = "Response Harm: harmful"
        result = _extract_harm_value(response, "Response Harm")
        assert result == "harmful"

    def test_missing_field_defaults_to_harmful(self):
        """Test missing field defaults to harmful."""
        response = "Some other text without the field"
        result = _extract_harm_value(response, "Prompt harm")
        assert result == "harmful"

    def test_extra_whitespace_around_colon(self):
        """Test handling of extra whitespace around colon."""
        response = "Prompt harm  :   unharmful"
        result = _extract_harm_value(response, "Prompt harm")
        assert result == "unharmful"

    def test_multiline_response(self):
        """Test extracting from multiline response."""
        response = "Prompt harm: harmful\nResponse Harm: unharmful"
        result = _extract_harm_value(response, "Response Harm")
        assert result == "unharmful"


class TestNemotronReasoningParsePromptSafety:
    """Test the nemotron_reasoning_parse_prompt_safety output parser."""

    def test_unharmful_prompt_no_reasoning(self):
        """Test parsing unharmful prompt without reasoning."""
        response = "Prompt harm: unharmful\nResponse Harm: unharmful"
        is_safe, *violated_policies = nemotron_reasoning_parse_prompt_safety(response)
        assert is_safe is True
        assert violated_policies == []

    def test_harmful_prompt_no_reasoning(self):
        """Test parsing harmful prompt without reasoning."""
        response = "Prompt harm: harmful\nResponse Harm: unharmful"
        is_safe, *violated_policies = nemotron_reasoning_parse_prompt_safety(response)
        assert is_safe is False
        assert violated_policies == []

    def test_unharmful_prompt_with_reasoning(self):
        """Test parsing unharmful prompt with reasoning tags."""
        response = """<think>
The user is asking about exercise benefits, which is a safe topic.
</think>

Prompt harm: unharmful
Response Harm: unharmful"""
        is_safe, *violated_policies = nemotron_reasoning_parse_prompt_safety(response)
        assert is_safe is True
        assert violated_policies == []

    def test_harmful_prompt_with_reasoning(self):
        """Test parsing harmful prompt with reasoning tags."""
        response = """<think>
The user's request falls under S21 (Illegal Activity) as it seeks guidance
on shoplifting, which is a criminal act.
</think>

Prompt harm: harmful
Response Harm: unharmful"""
        is_safe, *violated_policies = nemotron_reasoning_parse_prompt_safety(response)
        assert is_safe is False
        assert violated_policies == []

    def test_case_insensitive_parsing(self):
        """Test parsing is case insensitive."""
        response = "PROMPT HARM: UNHARMFUL"
        is_safe, *violated_policies = nemotron_reasoning_parse_prompt_safety(response)
        assert is_safe is True

    def test_missing_field_defaults_to_unsafe(self):
        """Test missing field defaults to unsafe (harmful)."""
        response = "Response Harm: unharmful"
        is_safe, *violated_policies = nemotron_reasoning_parse_prompt_safety(response)
        assert is_safe is False

    def test_empty_response_defaults_to_unsafe(self):
        """Test empty response defaults to unsafe."""
        response = ""
        is_safe, *violated_policies = nemotron_reasoning_parse_prompt_safety(response)
        assert is_safe is False

    def test_malformed_response_defaults_to_unsafe(self):
        """Test malformed response defaults to unsafe."""
        response = "The model refused to answer"
        is_safe, *violated_policies = nemotron_reasoning_parse_prompt_safety(response)
        assert is_safe is False


class TestNemotronReasoningParseResponseSafety:
    """Test the nemotron_reasoning_parse_response_safety output parser."""

    def test_unharmful_response_no_reasoning(self):
        """Test parsing unharmful response without reasoning."""
        response = "Prompt harm: unharmful\nResponse Harm: unharmful"
        is_safe, *violated_policies = nemotron_reasoning_parse_response_safety(response)
        assert is_safe is True
        assert violated_policies == []

    def test_harmful_response_no_reasoning(self):
        """Test parsing harmful response without reasoning."""
        response = "Prompt harm: unharmful\nResponse Harm: harmful"
        is_safe, *violated_policies = nemotron_reasoning_parse_response_safety(response)
        assert is_safe is False
        assert violated_policies == []

    def test_unharmful_response_with_reasoning(self):
        """Test parsing unharmful response with reasoning tags."""
        response = """<think>
The AI assistant's response provides helpful exercise information.
This is safe content.
</think>

Prompt harm: unharmful
Response Harm: unharmful"""
        is_safe, *violated_policies = nemotron_reasoning_parse_response_safety(response)
        assert is_safe is True
        assert violated_policies == []

    def test_harmful_response_with_reasoning(self):
        """Test parsing harmful response with reasoning tags."""
        response = """<think>
The AI assistant's response contains violent content that falls under S1.
</think>

Prompt harm: unharmful
Response Harm: harmful"""
        is_safe, *violated_policies = nemotron_reasoning_parse_response_safety(response)
        assert is_safe is False
        assert violated_policies == []

    def test_case_insensitive_parsing(self):
        """Test parsing is case insensitive."""
        response = "RESPONSE HARM: UNHARMFUL"
        is_safe, *violated_policies = nemotron_reasoning_parse_response_safety(response)
        assert is_safe is True

    def test_missing_field_defaults_to_unsafe(self):
        """Test missing field defaults to unsafe (harmful)."""
        response = "Prompt harm: unharmful"
        is_safe, *violated_policies = nemotron_reasoning_parse_response_safety(response)
        assert is_safe is False

    def test_empty_response_defaults_to_unsafe(self):
        """Test empty response defaults to unsafe."""
        response = ""
        is_safe, *violated_policies = nemotron_reasoning_parse_response_safety(response)
        assert is_safe is False

    def test_starred_unpacking_compatibility(self):
        """Test parser output is compatible with starred unpacking."""
        response = "Prompt harm: unharmful\nResponse Harm: unharmful"
        result = nemotron_reasoning_parse_response_safety(response)
        is_safe, *violated_policies = result
        assert is_safe is True
        assert violated_policies == []

        response = "Response Harm: harmful"
        result = nemotron_reasoning_parse_response_safety(response)
        is_safe, *violated_policies = result
        assert is_safe is False
        assert violated_policies == []


class TestNemotronReasoningRealWorldScenarios:
    """Test nemotron reasoning parsers with real-world scenarios."""

    def test_typical_safe_request_response(self):
        """Test typical safe request with reasoning mode off."""
        response = """Prompt harm: unharmful
Response harm: unharmful"""
        is_safe, *_ = nemotron_reasoning_parse_prompt_safety(response)
        assert is_safe is True

        is_safe, *_ = nemotron_reasoning_parse_response_safety(response)
        assert is_safe is True

    def test_typical_unsafe_request_with_reasoning(self):
        """Test typical unsafe request with full reasoning trace."""
        response = """<think>
The human user's request falls under S21 (Illegal Activity) as it seeks guidance on theft,
which is a criminal act. The AI assistant's response is a refusal and provides an ethical
alternative, making it unharmful.
</think>

Prompt harm: harmful
Response harm: unharmful"""

        is_safe, *_ = nemotron_reasoning_parse_prompt_safety(response)
        assert is_safe is False

        is_safe, *_ = nemotron_reasoning_parse_response_safety(response)
        assert is_safe is True

    def test_response_harm_lowercase(self):
        """Test parsing 'Response harm' (lowercase h) which is used in prompts."""
        response = "Prompt harm: unharmful\nResponse harm: harmful"
        is_safe, *_ = nemotron_reasoning_parse_response_safety(response)
        assert is_safe is False
