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

"""Sensitive data redaction for logging.

Redacts PII and sensitive patterns from logs to prevent data leaks.
Supports custom redaction patterns and configurable masking strategies.
"""

import re
from typing import Any, Callable, Dict, List, Optional, Union

# Default sensitive patterns to redact.
# ORDER MATTERS — patterns are applied sequentially; more specific patterns must come first
# to prevent partial matches by broader patterns (e.g. credit-card digits being swallowed by
# the phone pattern, or URL credentials being matched as an email address).
DEFAULT_REDACTION_PATTERNS = {
    # URL-with-creds must precede email: "password@host.example.com" would otherwise be
    # treated as an email address before the full credential URL is matched.
    "url_with_creds": (r"(?:https?://)?(?:[a-zA-Z0-9_-]+):(?:[a-zA-Z0-9_-]+)@[^\s]+", "[URL_WITH_CREDS]"),
    "email": (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[EMAIL]"),
    "ssn": (r"\b\d{3}-\d{2}-\d{4}\b", "[SSN]"),
    # Credit-card must precede phone: the phone pattern can partially match the first
    # 8 digits of a 16-digit card number (e.g. "1234-5678" in "1234-5678-9012-3456").
    "credit_card": (r"\b(?:\d{4}[-\s]?){3}\d{4}\b|\b\d{16}\b", "[CREDIT_CARD]"),
    "phone": (r"\b(?:\+?1[-.]?)?(?:\(?[0-9]{3}\)?[-.]?)?[0-9]{3}[-.]?[0-9]{4}\b", "[PHONE]"),
    "api_key": (
        r'(?:api[\s_-]?key|apikey|api_secret|secret)["\']?\s*[:=]\s*["\']?([A-Za-z0-9_\-]{8,})["\']?',
        "[API_KEY]",
    ),
    "password": (r'(?:password|passwd|pwd)["\']?\s*[:=]\s*["\']?([^"\'\s,}\]]+)["\']?', "[PASSWORD]"),
    "token": (r'(?:token|auth_token|access_token|bearer)["\']?\s*[:=]\s*["\']?([A-Za-z0-9_\-\.]+)["\']?', "[TOKEN]"),
    "aws_key": (r"AKIA[0-9A-Z]{16}", "[AWS_KEY]"),
    "ip_address": (
        r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b",
        "[IP_ADDRESS]",
    ),
}

# Sensitive keywords that indicate sensitive content
SENSITIVE_KEYWORDS = {
    "password",
    "secret",
    "token",
    "key",
    "credential",
    "private",
    "ssn",
    "social_security",
    "credit_card",
    "card_number",
    "api_key",
    "auth",
    "authorization",
    "access_token",
    "bearer",
    "api_secret",
    "client_secret",
    "private_key",
    "aws_secret",
    "gcp_key",
    "azure_key",
}


class SensitiveDataRedactor:
    """Redacts sensitive information from text."""

    def __init__(
        self,
        patterns: Optional[Dict[str, tuple]] = None,
        custom_patterns: Optional[Dict[str, tuple]] = None,
        custom_redactor: Optional[Callable[[str], str]] = None,
    ):
        """Initialize the redactor.

        Args:
            patterns: Redaction patterns (regex, replacement) dict
            custom_patterns: Additional custom patterns
            custom_redactor: Custom redaction function
        """
        self.patterns = patterns or DEFAULT_REDACTION_PATTERNS.copy()
        if custom_patterns:
            self.patterns.update(custom_patterns)

        self.custom_redactor = custom_redactor
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Compile regex patterns for efficiency."""
        self.compiled_patterns: List[tuple] = []
        for pattern_name, (regex_str, replacement) in self.patterns.items():
            try:
                compiled = re.compile(regex_str, re.IGNORECASE)
                self.compiled_patterns.append((compiled, replacement, pattern_name))
            except re.error as e:
                raise ValueError(f"Invalid regex pattern '{regex_str}': {e}")

    def redact(self, text: str) -> str:
        """Redact sensitive data from text.

        Args:
            text: The text to redact

        Returns:
            Text with sensitive data replaced with placeholders
        """
        if not text or not isinstance(text, str):
            return text

        redacted = text
        for compiled_pattern, replacement, pattern_name in self.compiled_patterns:
            redacted = compiled_pattern.sub(replacement, redacted)

        # Apply custom redactor if provided
        if self.custom_redactor:
            redacted = self.custom_redactor(redacted)

        return redacted

    def should_redact_value(self, key: str, value: Any) -> bool:
        """Determine if a key-value pair should be redacted.

        Args:
            key: The key name
            value: The value

        Returns:
            True if value should be redacted
        """
        if not isinstance(key, str):
            return False

        key_lower = key.lower()
        return any(keyword in key_lower for keyword in SENSITIVE_KEYWORDS)

    def redact_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Redact sensitive values in a dictionary.

        Args:
            data: Dictionary to redact

        Returns:
            Dictionary with sensitive values redacted
        """
        if not isinstance(data, dict):
            return data

        redacted = {}
        for key, value in data.items():
            if self.should_redact_value(key, value) and value is not None:
                redacted[key] = f"[{key.upper()}]"
            elif isinstance(value, str):
                redacted[key] = self.redact(value)
            elif isinstance(value, dict):
                redacted[key] = self.redact_dict(value)
            elif isinstance(value, (list, tuple)):
                redacted[key] = type(value)(
                    self.redact(item)
                    if isinstance(item, str)
                    else self.redact_dict(item)
                    if isinstance(item, dict)
                    else item
                    for item in value
                )
            else:
                redacted[key] = value

        return redacted

    def redact_list(self, data: Union[List[Any], tuple]) -> Union[List[Any], tuple]:
        """Redact sensitive data in a list or tuple.

        Args:
            data: List or tuple to redact

        Returns:
            List or tuple with sensitive data redacted
        """
        if isinstance(data, tuple):
            return tuple(
                self.redact(item)
                if isinstance(item, str)
                else self.redact_dict(item)
                if isinstance(item, dict)
                else item
                for item in data
            )
        elif isinstance(data, list):
            return [
                self.redact(item)
                if isinstance(item, str)
                else self.redact_dict(item)
                if isinstance(item, dict)
                else item
                for item in data
            ]
        else:
            return data


def create_sensitive_redactor(
    patterns: Optional[Dict[str, tuple]] = None,
    custom_patterns: Optional[Dict[str, tuple]] = None,
) -> SensitiveDataRedactor:
    """Factory function to create a configured redactor.

    Args:
        patterns: Override default patterns
        custom_patterns: Add custom patterns

    Returns:
        Configured SensitiveDataRedactor instance
    """
    return SensitiveDataRedactor(patterns=patterns, custom_patterns=custom_patterns)


# Global redactor instance
_global_redactor: Optional[SensitiveDataRedactor] = None


def get_redactor() -> SensitiveDataRedactor:
    """Get or create the global redactor instance."""
    global _global_redactor
    if _global_redactor is None:
        _global_redactor = SensitiveDataRedactor()
    return _global_redactor


def redact_text(text: str) -> str:
    """Redact sensitive data from text using global redactor.

    Args:
        text: Text to redact

    Returns:
        Redacted text
    """
    return get_redactor().redact(text)


def redact_value(value: Any) -> Any:
    """Redact sensitive data from any value.

    Handles strings, dicts, lists, and nested structures.

    Args:
        value: Value to redact

    Returns:
        Redacted value
    """
    redactor = get_redactor()

    if isinstance(value, str):
        return redactor.redact(value)
    elif isinstance(value, dict):
        return redactor.redact_dict(value)
    elif isinstance(value, (list, tuple)):
        return redactor.redact_list(value)
    else:
        return value
