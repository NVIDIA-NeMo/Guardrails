# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prompt injection detection and prevention module.

Detects common prompt injection attack patterns including:
- System prompt override attempts
- Instruction delimiter injection
- Role-switching and jailbreak patterns
- Token smuggling
"""

import re
from typing import List, Optional, Union


class PromptInjectionDetectedError(ValueError):
    """Raised when a prompt injection attack is detected."""

    def __init__(self, message: str, injection_pattern: Optional[str] = None):
        self.injection_pattern = injection_pattern
        super().__init__(message)


class PromptInjectionDetector:
    """Detects prompt injection attempts in user inputs."""

    # Patterns that indicate injection attempts
    INJECTION_PATTERNS = [
        # System prompt overrides
        (r'\bignore\s+(?:the\s+)?previous\b', 'ignore_previous'),
        (r'\bignore\s+all\s+(?:previous\s+)?instructions\b', 'ignore_instructions'),
        (r'\bforget\s+(?:the\s+)?previous\b', 'forget_previous'),
        (r'\bsystem\s*[:=]\s*', 'system_override'),
        (r'\b[Ii]nstructions?\s*[:=]', 'instruction_override'),
        (r'\b(?:system|admin|root)\s+(?:prompt|message|instruction)', 'privilege_claim'),

        # Instruction delimiter injection
        (r'^#+\s*(?:system|admin|instruction|new task)', 'delimiter_system'),
        (r'[-=]{3,}\s*(?:system|admin|instruction)', 'delimiter_instruction'),
        (r'\[(?:SYSTEM|ADMIN|INSTRUCTION|JAILBREAK)\]', 'bracket_delimiter'),

        # Role-switching and jailbreak
        (r'\b(?:you\s+are\s+now|pretend\s+(?:you\s+)?are|act\s+as|playing\s+the\s+role)', 'role_switch'),
        (r'\b(?:new\s+mode|special\s+mode|secret\s+mode)', 'mode_switch'),
        (r'\b(?:jailbreak|bypass|override)\s+(?:the\s+)?guardrails?\b', 'explicit_jailbreak'),

        # Nested prompt injection
        (r'(?:<!--.*?-->)|(?:\\[.*?\\])', 'nested_comment'),
        (r'\$\{.*?\}|\$\(.*?\)', 'variable_expansion'),

        # Token smuggling
        (r'(?:Base64|base64)\s+(?:decode|encoded)', 'token_smuggling'),
        (r'eval\s*\(|exec\s*\(', 'code_execution'),

        # Continuation patterns
        (r'\"\s*(?:\+|,)\s*\"', 'string_continuation'),
        (r"'\s*(?:\+|,)\s*'", 'string_continuation'),
    ]

    def __init__(self, sensitivity: str = 'medium'):
        """Initialize the detector with specified sensitivity level.

        Args:
            sensitivity: 'low' (minimal detection), 'medium' (default), 'high' (strict)
        """
        self.sensitivity = sensitivity
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Compile regex patterns for faster matching."""
        self.compiled_patterns = []
        for pattern, name in self.INJECTION_PATTERNS:
            flags = re.IGNORECASE | re.MULTILINE
            try:
                compiled = re.compile(pattern, flags)
                self.compiled_patterns.append((compiled, name))
            except re.error as e:
                raise ValueError(f"Invalid regex pattern '{pattern}': {e}")

    def detect(self, text: str, raise_error: bool = True) -> Optional[str]:
        """Detect prompt injection attempts in text.

        Args:
            text: The text to check for injection patterns
            raise_error: If True, raise PromptInjectionDetectedError on detection

        Returns:
            The name of the detected injection pattern, or None if clean

        Raises:
            PromptInjectionDetectedError: If injection is detected and raise_error=True
        """
        if not text or not isinstance(text, str):
            return None

        # Clean whitespace for analysis
        text_normalized = text.strip()

        for compiled_pattern, pattern_name in self.compiled_patterns:
            match = compiled_pattern.search(text_normalized)
            if match:
                if raise_error:
                    raise PromptInjectionDetectedError(
                        f"Prompt injection detected: {pattern_name}. "
                        f"User input contains instructions that attempt to override guardrails. "
                        f"Pattern: '{match.group()}'",
                        injection_pattern=pattern_name,
                    )
                return pattern_name

        return None

    def detect_in_messages(
        self, messages: List[dict], raise_error: bool = True
    ) -> Optional[dict]:
        """Detect injection attempts in message list.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            raise_error: If True, raise error on detection

        Returns:
            Dict with details of detected injection, or None if clean

        Raises:
            PromptInjectionDetectedError: If injection is detected and raise_error=True
        """
        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                continue

            content = msg.get('content')
            if not content or not isinstance(content, str):
                continue

            # Check all user-like messages for injection
            role = msg.get('role', '').lower()
            if role in ('user', 'human', 'input'):
                pattern = self.detect(content, raise_error=False)
                if pattern:
                    if raise_error:
                        raise PromptInjectionDetectedError(
                            f"Prompt injection detected in message {i} (role: {role}): {pattern}. "
                            f"Message content: '{content[:100]}...'",
                            injection_pattern=pattern,
                        )
                    return {
                        'message_index': i,
                        'role': role,
                        'pattern': pattern,
                        'content_preview': content[:100],
                    }

        return None


def validate_prompt_safety(
    prompt: Optional[str] = None,
    messages: Optional[List[dict]] = None,
    sensitivity: str = 'medium',
) -> None:
    """Validate prompt for injection attacks.

    Args:
        prompt: Single prompt string to validate
        messages: List of message dicts to validate
        sensitivity: Detection sensitivity ('low', 'medium', 'high')

    Raises:
        PromptInjectionDetectedError: If injection is detected
    """
    detector = PromptInjectionDetector(sensitivity=sensitivity)

    if prompt is not None:
        detector.detect(prompt, raise_error=True)

    if messages is not None:
        detector.detect_in_messages(messages, raise_error=True)
