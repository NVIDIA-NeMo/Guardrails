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
        assert TokenCounter.get_model_context_window('gpt-4o') == 128000
        assert TokenCounter.get_model_context_window('claude-3-opus') == 200000

    def test_get_model_context_window_partial_match(self):
        """Partial model name match should work."""
        assert TokenCounter.get_model_context_window('gpt-4') == 8192
        assert TokenCounter.get_model_context_window('claude-3') == 200000

    def test_get_model_context_window_unknown_model(self):
        """Unknown model should return default."""
        default_window = TokenCounter.get_model_context_window('unknown-model-xyz')
        assert default_window == TokenCounter.MODEL_CONTEXT_WINDOWS['default']

    def test_get_model_context_window_none(self):
        """None model should return default."""
        default_window = TokenCounter.get_model_context_window(None)
        assert default_window == TokenCounter.MODEL_CONTEXT_WINDOWS['default']

    def test_validate_context_length_string_prompt_valid(self):
        """Valid string prompt should not raise."""
        prompt = "What is the capital of France?"
        # Should not raise
        TokenCounter.validate_context_length(prompt, model_name='gpt-4')

    def test_validate_context_length_string_prompt_too_long(self):
        """String prompt exceeding limit should raise."""
        prompt = "a" * 100000  # Very long prompt
        with pytest.raises(ContextLengthExceededError) as exc_info:
            TokenCounter.validate_context_length(prompt, model_name='gpt-3.5-turbo')
        assert exc_info.value.model_name == 'gpt-3.5-turbo'

    def test_validate_context_length_message_list_valid(self):
        """Valid message list should not raise."""
        messages = [
            {"role": "user", "content": "What is the capital of France?"}
        ]
        # Should not raise
        TokenCounter.validate_context_length(messages, model_name='gpt-4')

    def test_validate_context_length_message_list_too_long(self):
        """Message list exceeding limit should raise."""
        messages = [
            {"role": "user", "content": "a" * 100000}
        ]
        with pytest.raises(ContextLengthExceededError):
            TokenCounter.validate_context_length(messages, model_name='gpt-3.5-turbo')

    def test_validate_context_length_uses_safety_threshold(self):
        """Should use 90% safety threshold."""
        # Create prompt that fits in 90% but exceeds 100%
        # gpt-4 has 8192 token window, so 90% = 7372
        # A prompt with ~8000 chars should exceed threshold
        prompt = "a" * 32000  # ~8000 tokens
        with pytest.raises(ContextLengthExceededError):
            TokenCounter.validate_context_length(prompt, model_name='gpt-4')

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
            TokenCounter.validate_context_length(prompt, model_name='gpt-3.5-turbo')
            assert False, "Should have raised"
        except ContextLengthExceededError as e:
            assert e.prompt_tokens > 0
            assert e.max_tokens == 4096
            assert e.model_name == 'gpt-3.5-turbo'
            assert 'tokens' in str(e).lower()

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
        validate_context_length(prompt, model_name='gpt-4')

    def test_convenience_function_raises(self):
        """Convenience function should raise on too long prompt."""
        prompt = "a" * 100000
        with pytest.raises(ContextLengthExceededError):
            validate_context_length(prompt, model_name='gpt-3.5-turbo')

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
            validate_context_length(prompt, model_name='gpt-3.5-turbo')

    def test_large_context_model_allows_longer_prompts(self):
        """Large context models should accept longer prompts."""
        prompt = "a" * 50000  # ~12500 tokens
        # Claude has 200k context, should accept this
        validate_context_length(prompt, model_name='claude-3-opus')

        # GPT-3.5 with 4k context should reject it
        with pytest.raises(ContextLengthExceededError):
            validate_context_length(prompt, model_name='gpt-3.5-turbo')

    def test_context_length_error_inheritance(self):
        """ContextLengthExceededError should be ValueError."""
        assert issubclass(ContextLengthExceededError, ValueError)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
