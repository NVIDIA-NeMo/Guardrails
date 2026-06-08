# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for token counting and context length validation."""

import pytest

from nemoguardrails.llm.token_counter import (
    ContextLengthExceededError,
    TokenCounter,
    validate_context_length,
)


class TestTokenCounter:
    """Test suite for TokenCounter."""

    def test_estimate_tokens_empty_string(self):
        """Empty string should return 0 tokens."""
        assert TokenCounter.estimate_tokens("") == 0

    def test_estimate_tokens_short_text(self):
        """Short text estimation."""
        text = "Hello"
        tokens = TokenCounter.estimate_tokens(text)
        assert tokens >= 1

    def test_estimate_tokens_long_text(self):
        """Long text should estimate reasonable token count."""
        text = "a" * 1000  # 1000 characters
        tokens = TokenCounter.estimate_tokens(text)
        # Roughly 4 chars per token, so ~250 tokens
        assert 200 < tokens < 300

    def test_estimate_tokens_realistic_prompt(self):
        """Realistic prompt should estimate reasonable tokens."""
        prompt = "What is the capital of France? " * 10  # Repeat to get ~320 chars
        tokens = TokenCounter.estimate_tokens(prompt)
        assert tokens > 0

    def test_estimate_message_tokens_empty_list(self):
        """Empty message list should return 0."""
        assert TokenCounter.estimate_message_tokens([]) == 0

    def test_estimate_message_tokens_single_message(self):
        """Single message token count."""
        messages = [{"role": "user", "content": "Hello"}]
        tokens = TokenCounter.estimate_message_tokens(messages)
        assert tokens > 0

    def test_estimate_message_tokens_multiple_messages(self):
        """Multiple messages token count."""
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "2+2 equals 4"},
        ]
        tokens = TokenCounter.estimate_message_tokens(messages)
        # Should account for message overhead + content
        assert tokens > 10

    def test_estimate_message_tokens_includes_overhead(self):
        """Message token count should include structure overhead."""
        messages = [{"role": "user", "content": ""}]
        tokens = TokenCounter.estimate_message_tokens(messages)
        # Even empty message should account for structure
        assert tokens >= 4

    def test_get_model_context_window_known_model(self):
        """Known model should return correct context window."""
        assert TokenCounter.get_model_context_window("gpt-4o") == 128000
        assert TokenCounter.get_model_context_window("claude-3-opus") == 200000

    def test_get_model_context_window_partial_match(self):
        """Partial model name match should work."""
        assert TokenCounter.get_model_context_window("gpt-4") == 8192
        assert TokenCounter.get_model_context_window("claude-3") == 200000

    def test_get_model_context_window_unknown_model(self):
        """Unknown model should return None."""
        assert TokenCounter.get_model_context_window("unknown-model-xyz") is None

    def test_get_model_context_window_none(self):
        """None model should return None."""
        assert TokenCounter.get_model_context_window(None) is None

    def test_validate_context_length_skips_unknown_model(self):
        """Validation is skipped for unrecognised models rather than using a silent fallback."""
        long_prompt = "a" * 50000
        # Should not raise — context window is unknown so validation is skipped
        TokenCounter.validate_context_length(long_prompt, model_name="my-custom-ollama-model")
        TokenCounter.validate_context_length(long_prompt, model_name=None)

    def test_validate_context_length_string_prompt_valid(self):
        """Valid string prompt should not raise."""
        prompt = "What is the capital of France?"
        # Should not raise
        TokenCounter.validate_context_length(prompt, model_name="gpt-4")

    def test_validate_context_length_string_prompt_too_long(self):
        """String prompt exceeding limit should raise."""
        prompt = "a" * 100000  # Very long prompt
        with pytest.raises(ContextLengthExceededError) as exc_info:
            TokenCounter.validate_context_length(prompt, model_name="gpt-3.5-turbo")
        assert exc_info.value.model_name == "gpt-3.5-turbo"

    def test_validate_context_length_message_list_valid(self):
        """Valid message list should not raise."""
        messages = [{"role": "user", "content": "What is the capital of France?"}]
        # Should not raise
        TokenCounter.validate_context_length(messages, model_name="gpt-4")

    def test_validate_context_length_message_list_too_long(self):
        """Message list exceeding limit should raise."""
        messages = [{"role": "user", "content": "a" * 100000}]
        with pytest.raises(ContextLengthExceededError):
            TokenCounter.validate_context_length(messages, model_name="gpt-3.5-turbo")

    def test_validate_context_length_uses_safety_threshold(self):
        """Should use 90% safety threshold."""
        # Create prompt that fits in 90% but exceeds 100%
        # gpt-4 has 8192 token window, so 90% = 7372
        # A prompt with ~8000 chars should exceed threshold
        prompt = "a" * 32000  # ~8000 tokens
        with pytest.raises(ContextLengthExceededError):
            TokenCounter.validate_context_length(prompt, model_name="gpt-4")

    def test_validate_context_length_with_custom_max_tokens(self):
        """Should respect custom max_tokens parameter."""
        prompt = "test" * 100  # ~100 tokens
        # Custom limit of 50 tokens should raise
        with pytest.raises(ContextLengthExceededError):
            TokenCounter.validate_context_length(prompt, max_tokens=50)

    def test_validate_context_length_exception_details(self):
        """Exception should contain useful debugging info."""
        prompt = "a" * 50000
        try:
            TokenCounter.validate_context_length(prompt, model_name="gpt-3.5-turbo")
            assert False, "Should have raised"
        except ContextLengthExceededError as e:
            assert e.prompt_tokens > 0
            assert e.max_tokens == 4096
            assert e.model_name == "gpt-3.5-turbo"
            assert "tokens" in str(e).lower()

    def test_validate_context_length_unknown_type(self):
        """Should handle unknown prompt types gracefully."""
        # Should not raise for unknown types
        TokenCounter.validate_context_length(12345)  # Invalid type
        TokenCounter.validate_context_length(None)  # None
        TokenCounter.validate_context_length({})  # Dict

    def test_convenience_function_validate_context_length(self):
        """Convenience function should work."""
        prompt = "What is the capital of France?"
        # Should not raise
        validate_context_length(prompt, model_name="gpt-4")

    def test_convenience_function_raises(self):
        """Convenience function should raise on too long prompt."""
        prompt = "a" * 100000
        with pytest.raises(ContextLengthExceededError):
            validate_context_length(prompt, model_name="gpt-3.5-turbo")

    def test_message_with_missing_content(self):
        """Messages with missing content should be handled."""
        messages = [
            {"role": "user"},  # Missing content
            {"role": "user", "content": None},  # None content
        ]
        # Should not raise
        tokens = TokenCounter.estimate_message_tokens(messages)
        assert tokens >= 0

    def test_message_with_multimodal_content(self):
        """Messages with multimodal content should be estimated."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}},
                ],
            }
        ]
        tokens = TokenCounter.estimate_message_tokens(messages)
        # Should account for text + image
        assert tokens > 0

    def test_small_prompt_validation_passes(self):
        """Very small prompts should always pass."""
        tiny_prompts = [
            "Hi",
            "2+2",
            "What?",
        ]
        for prompt in tiny_prompts:
            # Should not raise
            validate_context_length(prompt, model_name="gpt-3.5-turbo")

    def test_large_context_model_allows_longer_prompts(self):
        """Large context models should accept longer prompts."""
        prompt = "a" * 50000  # ~12500 tokens
        # Claude has 200k context, should accept this
        validate_context_length(prompt, model_name="claude-3-opus")

        # GPT-3.5 with 4k context should reject it
        with pytest.raises(ContextLengthExceededError):
            validate_context_length(prompt, model_name="gpt-3.5-turbo")

    def test_context_length_error_inheritance(self):
        """ContextLengthExceededError should be ValueError."""
        assert issubclass(ContextLengthExceededError, ValueError)

    def test_estimate_message_tokens_chat_message_dataclass(self):
        """ChatMessage dataclass content must be counted, not just 4-token overhead."""
        from nemoguardrails.types import ChatMessage, Role

        chat_messages = [
            ChatMessage(role=Role.USER, content="What is the capital of France?"),
            ChatMessage(role=Role.ASSISTANT, content="The capital of France is Paris."),
        ]
        dict_messages = [
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "The capital of France is Paris."},
        ]
        chat_tokens = TokenCounter.estimate_message_tokens(chat_messages)
        dict_tokens = TokenCounter.estimate_message_tokens(dict_messages)
        # Dataclass path and dict path should produce identical counts
        assert chat_tokens == dict_tokens
        # Sanity: content tokens must be included, not just 4-token overhead per message
        assert chat_tokens > len(chat_messages) * 4

    def test_validate_context_length_chat_messages_too_long(self):
        """ChatMessage list exceeding context window must raise ContextLengthExceededError."""
        from nemoguardrails.types import ChatMessage, Role

        messages = [ChatMessage(role=Role.USER, content="a" * 100000)]
        with pytest.raises(ContextLengthExceededError):
            TokenCounter.validate_context_length(messages, model_name="gpt-3.5-turbo")

    def test_estimate_message_tokens_skips_unknown_type_items(self):
        """Non-dict, non-dataclass items in message list are skipped via continue (line 132)."""
        messages = [
            "a plain string",
            42,
            {"role": "user", "content": "Hello"},
        ]
        tokens = TokenCounter.estimate_message_tokens(messages)
        # Only the dict message contributes content tokens; the string/int are skipped
        dict_only_tokens = TokenCounter.estimate_message_tokens([{"role": "user", "content": "Hello"}])
        # Overhead: 3 messages * 4 tokens vs 1 message * 4 tokens
        assert tokens == dict_only_tokens + 2 * 4

    def test_get_model_context_window_partial_match_via_loop(self):
        """Partial match returns context window via the loop branch (line 172)."""
        # "gpt-4-custom" is not an exact key but "gpt-4" is a partial match
        window = TokenCounter.get_model_context_window("gpt-4-custom-variant")
        assert window == TokenCounter.MODEL_CONTEXT_WINDOWS["gpt-4"]

        # "claude-3-custom" is not an exact key but "claude-3" is a partial match
        window2 = TokenCounter.get_model_context_window("claude-3-custom-variant")
        assert window2 == TokenCounter.MODEL_CONTEXT_WINDOWS["claude-3"]

    def test_gpt35_turbo_variants_use_16k_window(self):
        """gpt-3.5-turbo-0125/1106/16k resolve to 16384, not the legacy 4096."""
        assert TokenCounter.get_model_context_window("gpt-3.5-turbo-0125") == 16384
        assert TokenCounter.get_model_context_window("gpt-3.5-turbo-1106") == 16384
        assert TokenCounter.get_model_context_window("gpt-3.5-turbo-16k") == 16384
        # Generic key still maps to legacy 4096
        assert TokenCounter.get_model_context_window("gpt-3.5-turbo") == 4096

    def test_gpt4_32k_context_window(self):
        """gpt-4-32k resolves to 32768."""
        assert TokenCounter.get_model_context_window("gpt-4-32k") == 32768


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
