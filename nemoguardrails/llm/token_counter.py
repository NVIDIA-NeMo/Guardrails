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

"""Token counting and context length validation utilities.

Provides methods to estimate token counts for prompts and validate
that prompts don't exceed model context windows.
"""

import logging
from typing import List, Optional, Union

log = logging.getLogger(__name__)


class ContextLengthExceededError(ValueError):
    """Raised when prompt exceeds model context length."""

    def __init__(
        self,
        message: str,
        prompt_tokens: int,
        max_tokens: int,
        model_name: Optional[str] = None,
    ):
        self.prompt_tokens = prompt_tokens
        self.max_tokens = max_tokens
        self.model_name = model_name
        super().__init__(message)


class TokenCounter:
    """Estimates token counts for various model types."""

    # Approximate tokens per character ratios for different model families
    # These are conservative estimates; actual counts depend on tokenizer
    TOKENS_PER_CHAR = {
        "gpt": 0.25,  # OpenAI models: ~4 chars per token
        "claude": 0.27,  # Anthropic: ~3.7 chars per token
        "llama": 0.28,  # Meta: ~3.6 chars per token
        "mistral": 0.28,
        "gemini": 0.26,
        "default": 0.27,
    }

    # Model context window limits (in tokens)
    MODEL_CONTEXT_WINDOWS = {
        # OpenAI
        "gpt-4o": 128000,
        "gpt-4-turbo": 128000,
        "gpt-4": 8192,
        "gpt-3.5-turbo": 4096,
        # Anthropic
        "claude-3-opus": 200000,
        "claude-3-sonnet": 200000,
        "claude-3-haiku": 200000,
        "claude-2.1": 100000,
        "claude-2": 100000,
        # Meta Llama
        "llama-2": 4096,
        "llama-2-70b": 4096,
        "llama-3": 8192,
        "llama-3-70b": 8192,
        # Mistral
        "mistral-7b": 32768,
        "mistral-large": 32768,
        # Google
        "gemini-pro": 32768,
        "gemini-2.0-flash": 1000000,
        # Default fallback
        "default": 4096,
    }

    @staticmethod
    def estimate_tokens(text: str, model_name: Optional[str] = None) -> int:
        """Estimate token count for text.

        Args:
            text: The text to estimate tokens for
            model_name: Optional model name for family-specific estimation

        Returns:
            Approximate token count
        """
        if not text:
            return 0

        # Determine ratio based on model family
        ratio = TokenCounter.TOKENS_PER_CHAR.get("default", 0.27)
        if model_name:
            model_lower = model_name.lower()
            for family, family_ratio in TokenCounter.TOKENS_PER_CHAR.items():
                if family in model_lower:
                    ratio = family_ratio
                    break

        return max(1, int(len(text) * ratio))

    @staticmethod
    def estimate_message_tokens(messages: List[dict], model_name: Optional[str] = None) -> int:
        """Estimate total token count for message list.

        Accounts for message structure overhead.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            model_name: Optional model name for family-specific estimation

        Returns:
            Approximate total token count including formatting
        """
        if not messages:
            return 0

        total_tokens = 0
        # Account for message structure overhead (~4 tokens per message)
        total_tokens += len(messages) * 4

        for msg in messages:
            if isinstance(msg, dict):
                content = msg.get("content", "")
                if isinstance(content, str):
                    total_tokens += TokenCounter.estimate_tokens(content, model_name)
                elif isinstance(content, list):
                    # For multimodal content
                    for item in content:
                        if isinstance(item, dict):
                            if item.get("type") == "text":
                                total_tokens += TokenCounter.estimate_tokens(item.get("text", ""), model_name)
                            elif item.get("type") == "image_url":
                                # Image tokens vary; rough estimate
                                total_tokens += 85
                            elif item.get("type") == "image":
                                total_tokens += 85

        return total_tokens

    @staticmethod
    def _tokenize(s: str):
        """Split a model name into tokens on non-alphanumeric separators."""
        import re

        return [t for t in re.split(r"[^a-z0-9]+", s.lower()) if t]

    @staticmethod
    def get_model_context_window(model_name: Optional[str]) -> int:
        """Get context window size for a model.

        Args:
            model_name: Name of the model

        Returns:
            Context window in tokens, or default if unknown
        """
        if not model_name:
            return TokenCounter.MODEL_CONTEXT_WINDOWS["default"]

        model_name_lower = model_name.lower()

        # Exact match first
        if model_name_lower in TokenCounter.MODEL_CONTEXT_WINDOWS:
            return TokenCounter.MODEL_CONTEXT_WINDOWS[model_name_lower]

        # Token-based matching: find the key with maximum token overlap
        # e.g., "my-claude-3-custom" tokens: [my, claude, 3, custom]
        #       "claude-3-opus" tokens: [claude, 3, opus]
        #       overlap: [claude, 3] -> score 2
        model_tokens = set(TokenCounter._tokenize(model_name_lower))

        best_key = None
        best_score = 0
        best_key_len = 0

        for key in TokenCounter.MODEL_CONTEXT_WINDOWS:
            if key == "default":
                continue

            key_tokens = set(TokenCounter._tokenize(key.lower()))
            if not key_tokens:
                continue

            # Calculate token overlap
            overlap = len(model_tokens & key_tokens)
            if overlap == 0:
                continue

            # Prefer higher overlap, tie-break on longer key (more specific)
            if overlap > best_score or (overlap == best_score and len(key) > best_key_len):
                best_score = overlap
                best_key = key
                best_key_len = len(key)

        if best_key:
            return TokenCounter.MODEL_CONTEXT_WINDOWS[best_key]

        # Default fallback
        return TokenCounter.MODEL_CONTEXT_WINDOWS["default"]

    @staticmethod
    def validate_context_length(
        prompt: Union[str, List[dict]],
        model_name: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> None:
        """Validate that prompt fits within model context window.

        Args:
            prompt: The prompt (string or message list) to validate
            model_name: Name of the model (for context window lookup)
            max_tokens: Override context window size

        Raises:
            ContextLengthExceededError: If prompt exceeds context window
        """
        if isinstance(prompt, str):
            prompt_tokens = TokenCounter.estimate_tokens(prompt, model_name)
        elif isinstance(prompt, list):
            prompt_tokens = TokenCounter.estimate_message_tokens(prompt, model_name)
        else:
            return  # Can't validate unknown type

        # Determine context window
        if max_tokens is None:
            max_tokens = TokenCounter.get_model_context_window(model_name)

        # Validate (reserve 10% for safety margin and output tokens)
        safety_threshold = int(max_tokens * 0.9)

        if prompt_tokens > safety_threshold:
            raise ContextLengthExceededError(
                f"Prompt exceeds model context length. "
                f"Prompt tokens: {prompt_tokens}, "
                f"Model context window: {max_tokens} "
                f"(using 90% threshold: {safety_threshold} tokens). "
                f"Context length exceeded by {prompt_tokens - safety_threshold} tokens. "
                f"Please reduce prompt length or use a model with larger context window.",
                prompt_tokens=prompt_tokens,
                max_tokens=max_tokens,
                model_name=model_name,
            )

        log.debug(
            f"Prompt token validation passed: {prompt_tokens}/{safety_threshold} tokens "
            f"(model: {model_name or 'unknown'})"
        )


def validate_context_length(
    prompt: Union[str, List[dict]],
    model_name: Optional[str] = None,
    max_tokens: Optional[int] = None,
) -> None:
    """Convenience function to validate context length.

    Args:
        prompt: The prompt to validate
        model_name: Name of the model
        max_tokens: Override context window size

    Raises:
        ContextLengthExceededError: If prompt exceeds context window
    """
    TokenCounter.validate_context_length(prompt, model_name, max_tokens)
