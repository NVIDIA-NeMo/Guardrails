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


import pytest

from nemoguardrails.llm.output_parsers import (
    _extract_harm_value,
    _strip_think_tags,
    is_content_safe,
    nemoguard_parse_prompt_safety,
    nemoguard_parse_response_safety,
    nemotron_content_safety_parse_prompt_safety,
    nemotron_content_safety_parse_response_safety,
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

    @pytest.mark.parametrize("response", ["", '{"invalid": json}'])
    def test_invalid_response_raises_parsing_error(self, response):
        with pytest.raises(ValueError, match="Failed to parse content safety model response"):
            nemoguard_parse_prompt_safety(response)

    def test_missing_user_safety_field(self):
        response = '{"Response Safety": "safe"}'
        with pytest.raises(ValueError, match="Failed to parse content safety model response"):
            nemoguard_parse_prompt_safety(response)

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

    @pytest.mark.parametrize("response", ["", '{"invalid": json}'])
    def test_invalid_response_raises_parsing_error(self, response):
        with pytest.raises(ValueError, match="Failed to parse content safety model response"):
            nemoguard_parse_response_safety(response)

    def test_missing_response_safety_field(self):
        response = '{"User Safety": "safe"}'
        with pytest.raises(ValueError, match="Failed to parse content safety model response"):
            nemoguard_parse_response_safety(response)

    def test_full_nemoguard_response(self):
        """Test parsing a full NemoGuard response with both user and response safety."""
        response = '{"User Safety": "safe", "Response Safety": "unsafe", "Safety Categories": "S1, S8"}'
        is_safe, *violated_policies = nemoguard_parse_response_safety(response)
        assert is_safe is False
        assert "S1" in violated_policies
        assert "S8" in violated_policies

    def test_malformed_json_with_extra_text(self):
        response = '{"Response Safety": "unsafe", "Safety Categories": "S1"} extra text'
        with pytest.raises(ValueError, match="Failed to parse content safety model response"):
            nemoguard_parse_response_safety(response)


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
        invalid_response = "The model refused to answer"

        is_safe, *violated_policies = is_content_safe(invalid_response)
        assert is_safe is False

        with pytest.raises(ValueError, match="Failed to parse content safety model response"):
            nemoguard_parse_prompt_safety(invalid_response)

        with pytest.raises(ValueError, match="Failed to parse content safety model response"):
            nemoguard_parse_response_safety(invalid_response)

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


MODEL_SAFE = "User Safety: safe"
MODEL_UNSAFE = "User Safety: unsafe\nSafety Categories: Criminal Planning/Confessions"
MODEL_EXCHANGE_ALL_SAFE = "User Safety: safe\nResponse Safety: safe"
MODEL_EXCHANGE_RESPONSE_UNSAFE = (
    "User Safety: safe\nResponse Safety: unsafe\nSafety Categories: Violence, Criminal Planning/Confessions"
)
MODEL_EXCHANGE_PROMPT_UNSAFE = (
    "User Safety: unsafe\nResponse Safety: safe\nSafety Categories: Criminal Planning/Confessions, Violence"
)
MODEL_EXCHANGE_ALL_UNSAFE = (
    "User Safety: unsafe\nResponse Safety: unsafe\nSafety Categories: Criminal Planning/Confessions, Violence"
)

ARSON_CATEGORIES = ["Criminal Planning/Confessions", "Violence"]

MODEL_DUPLICATE_USER_SAFETY = "User Safety: safe\nUser Safety: unsafe\nResponse Safety: safe"
MODEL_DUPLICATE_RESPONSE_SAFETY = "User Safety: safe\nResponse Safety: safe\nResponse Safety: unsafe"
MODEL_DUPLICATE_AGREEING_VERDICT = "User Safety: safe\nUser Safety: safe\nResponse Safety: safe"
MODEL_LEAKED_TRACE_WITH_REVISED_VERDICT = (
    "We need to decide.\n"
    "Response Safety: safe\n"
    "On reflection the assistant supplied arson instructions.\n"
    "Response Safety: unsafe\n"
    "User Safety: unsafe"
)

# A repeat need not start its own line. Every one of these keeps a well-formed line for the other
# field, so each case is red for the duplicate rather than for a field that is simply missing.
MODEL_SAME_LINE_DUPLICATE_USER_SAFETY = "User Safety: safe; User Safety: unsafe\nResponse Safety: safe"
MODEL_SAME_LINE_DUPLICATE_RESPONSE_SAFETY = "User Safety: safe\nResponse Safety: safe; Response Safety: unsafe"
MODEL_SAME_LINE_DUPLICATE_COMMA_SEPARATED = "User Safety: unsafe, User Safety: safe\nResponse Safety: safe"

# Trailing prose on an otherwise well-formed verdict line is tolerated rather than rejected, so a
# single stray remark cannot take the rail down. Only a repeated field is treated as a malfunction.
MODEL_VERDICT_WITH_TRAILING_PROSE = "User Safety: safe (no assistant response present)\nResponse Safety: safe"

# Shaped after a reasoning_content trace captured from the live NIM, which quotes the verdict lines
# back before restating them as the answer. Stripping has to happen before duplicates are counted.
MODEL_THINK_TRACE_REPEATING_VERDICTS = (
    "<think>\n"
    "We need to output:\n\n"
    "User Safety: unsafe\n"
    "Response Safety: safe\n\n"
    "Thus output exactly:\n\n"
    "User Safety: unsafe\n"
    "Response Safety: safe\n\n"
    "No extra text.\n"
    "</think>\n"
    "User Safety: unsafe\nResponse Safety: safe"
)


class TestNemotronContentSafetyParsePromptSafety:
    """Test the nemotron_content_safety_parse_prompt_safety output parser."""

    @pytest.mark.parametrize(
        "response,expected_safe,expected_categories",
        [
            (MODEL_SAFE, True, []),
            (MODEL_UNSAFE, False, ["Criminal Planning/Confessions"]),
            (MODEL_EXCHANGE_ALL_SAFE, True, []),
            (MODEL_EXCHANGE_RESPONSE_UNSAFE, True, []),
            (MODEL_EXCHANGE_PROMPT_UNSAFE, False, ARSON_CATEGORIES),
            (MODEL_EXCHANGE_ALL_UNSAFE, False, ARSON_CATEGORIES),
        ],
        ids=[
            "input_safe",
            "input_unsafe",
            "exchange_all_safe",
            "exchange_response_unsafe",
            "exchange_prompt_unsafe",
            "exchange_all_unsafe",
        ],
    )
    def test_captured_model_responses(self, response, expected_safe, expected_categories):
        """Test every captured model response yields the user-turn verdict and its categories."""
        is_safe, *violated_policies = nemotron_content_safety_parse_prompt_safety(response)
        assert is_safe is expected_safe
        assert sorted(violated_policies) == sorted(expected_categories)

    def test_unsafe_without_categories_line(self):
        """Test an unsafe verdict under /no_categories parses without raising."""
        is_safe, *violated_policies = nemotron_content_safety_parse_prompt_safety("User Safety: unsafe")
        assert is_safe is False
        assert violated_policies == []

    def test_multiline_think_trace(self):
        """Test a multi-line reasoning trace preceding the verdict is stripped."""
        response = "<think>\nThe request seeks instructions for arson.\nThat is unsafe.\n</think>\n\n" + MODEL_UNSAFE
        is_safe, *violated_policies = nemotron_content_safety_parse_prompt_safety(response)
        assert is_safe is False
        assert violated_policies == ["Criminal Planning/Confessions"]

    def test_case_insensitive_parsing(self):
        """Test parsing is case insensitive."""
        is_safe, *_ = nemotron_content_safety_parse_prompt_safety("USER SAFETY: UNSAFE")
        assert is_safe is False

    @pytest.mark.parametrize(
        "response",
        ["", "<think>truncated reasoning", '{"User Safety": "safe"}', "User Safety: undetermined"],
        ids=["empty", "truncated_reasoning", "json", "unknown_verdict"],
    )
    def test_invalid_response_raises(self, response):
        with pytest.raises(ValueError, match="Failed to parse content safety model response"):
            nemotron_content_safety_parse_prompt_safety(response)


class TestNemotronContentSafetyParseResponseSafety:
    """Test the nemotron_content_safety_parse_response_safety output parser."""

    @pytest.mark.parametrize(
        "response,expected_safe,expected_categories",
        [
            (MODEL_EXCHANGE_ALL_SAFE, True, []),
            (MODEL_EXCHANGE_RESPONSE_UNSAFE, False, ARSON_CATEGORIES),
            (MODEL_EXCHANGE_PROMPT_UNSAFE, True, []),
            (MODEL_EXCHANGE_ALL_UNSAFE, False, ARSON_CATEGORIES),
        ],
        ids=[
            "exchange_all_safe",
            "exchange_response_unsafe",
            "exchange_prompt_unsafe",
            "exchange_all_unsafe",
        ],
    )
    def test_captured_two_turn_responses(self, response, expected_safe, expected_categories):
        """Test every captured two-turn response yields the assistant-turn verdict and its categories."""
        is_safe, *violated_policies = nemotron_content_safety_parse_response_safety(response)
        assert is_safe is expected_safe
        assert sorted(violated_policies) == sorted(expected_categories)

    @pytest.mark.parametrize("response", [MODEL_SAFE, MODEL_UNSAFE], ids=["input_safe", "input_unsafe"])
    def test_missing_response_safety_line_raises(self, response):
        """Test a verdict with no assistant turn raises rather than reusing the user verdict."""
        with pytest.raises(ValueError, match="Failed to parse content safety model response"):
            nemotron_content_safety_parse_response_safety(response)

    def test_unsafe_without_categories_line(self):
        """Test an unsafe verdict under /no_categories parses without raising."""
        response = "User Safety: safe\nResponse Safety: unsafe"
        is_safe, *violated_policies = nemotron_content_safety_parse_response_safety(response)
        assert is_safe is False
        assert violated_policies == []

    def test_empty_response_raises(self):
        """Test an empty response raises rather than silently blocking."""
        with pytest.raises(ValueError, match="Failed to parse content safety model response"):
            nemotron_content_safety_parse_response_safety("")


class TestNemotronContentSafetyDuplicateVerdicts:
    """A verdict field stated more than once means the model contradicted itself and cannot be trusted."""

    @pytest.mark.parametrize(
        "parser",
        [nemotron_content_safety_parse_prompt_safety, nemotron_content_safety_parse_response_safety],
        ids=["prompt", "response"],
    )
    @pytest.mark.parametrize(
        "response",
        [
            MODEL_DUPLICATE_USER_SAFETY,
            MODEL_DUPLICATE_RESPONSE_SAFETY,
            MODEL_DUPLICATE_AGREEING_VERDICT,
            MODEL_LEAKED_TRACE_WITH_REVISED_VERDICT,
            MODEL_SAME_LINE_DUPLICATE_USER_SAFETY,
            MODEL_SAME_LINE_DUPLICATE_RESPONSE_SAFETY,
            MODEL_SAME_LINE_DUPLICATE_COMMA_SEPARATED,
        ],
        ids=[
            "user_safety_twice",
            "response_safety_twice",
            "same_verdict_twice",
            "leaked_trace_revises_verdict",
            "user_safety_twice_on_one_line",
            "response_safety_twice_on_one_line",
            "same_line_comma_separated",
        ],
    )
    def test_duplicate_verdict_field_raises(self, parser, response):
        """Test either field stated twice raises for both parsers, rather than trusting the first match."""
        with pytest.raises(ValueError, match="Failed to parse content safety model response"):
            parser(response)

    @pytest.mark.parametrize(
        "parser,expected_safe",
        [
            (nemotron_content_safety_parse_prompt_safety, False),
            (nemotron_content_safety_parse_response_safety, True),
        ],
        ids=["prompt", "response"],
    )
    def test_verdicts_quoted_inside_a_think_trace_are_not_duplicates(self, parser, expected_safe):
        """Test verdicts repeated inside a stripped reasoning trace do not count toward the duplicate check."""
        is_safe, *_ = parser(MODEL_THINK_TRACE_REPEATING_VERDICTS)
        assert is_safe is expected_safe

    @pytest.mark.parametrize(
        "parser",
        [nemotron_content_safety_parse_prompt_safety, nemotron_content_safety_parse_response_safety],
        ids=["prompt", "response"],
    )
    def test_trailing_prose_on_a_verdict_line_is_not_a_duplicate(self, parser):
        """Test a verdict line carrying trailing prose still parses, since the field is stated only once."""
        is_safe, *_ = parser(MODEL_VERDICT_WITH_TRAILING_PROSE)
        assert is_safe is True
