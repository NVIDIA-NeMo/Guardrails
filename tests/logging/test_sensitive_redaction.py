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

"""Tests for sensitive data redaction in logging."""

import logging

import pytest

from nemoguardrails.logging.redactor import (
    SensitiveDataRedactor,
    get_redactor,
    redact_text,
    redact_value,
)
from nemoguardrails.logging.sensitive_filter import SensitiveDataFilter


class TestSensitiveDataRedactor:
    """Test suite for SensitiveDataRedactor."""

    @pytest.fixture
    def redactor(self):
        """Create a redactor instance."""
        return SensitiveDataRedactor()

    def test_redact_email(self, redactor):
        """Email addresses should be redacted."""
        text = "Contact us at john@example.com for support"
        redacted = redactor.redact(text)
        assert "john@example.com" not in redacted
        assert "[EMAIL]" in redacted

    def test_redact_phone(self, redactor):
        """Phone numbers should be redacted."""
        text = "Call us at 555-123-4567 during business hours"
        redacted = redactor.redact(text)
        assert "555-123-4567" not in redacted
        assert "[PHONE]" in redacted

    def test_redact_ssn(self, redactor):
        """SSN should be redacted."""
        text = "SSN: 123-45-6789"
        redacted = redactor.redact(text)
        assert "123-45-6789" not in redacted
        assert "[SSN]" in redacted

    def test_redact_credit_card(self, redactor):
        """Credit card numbers should be redacted."""
        text = "Card: 1234-5678-9012-3456"
        redacted = redactor.redact(text)
        assert "1234-5678-9012-3456" not in redacted
        assert "[CREDIT_CARD]" in redacted

    def test_redact_api_key(self, redactor):
        """API keys should be redacted."""
        text = 'api_key="sk_live_1234567890abcdefghij"'
        redacted = redactor.redact(text)
        assert "sk_live_1234567890abcdefghij" not in redacted
        assert "[API_KEY]" in redacted

    def test_redact_password(self, redactor):
        """Passwords should be redacted."""
        text = 'password="super_secret_password123"'
        redacted = redactor.redact(text)
        assert "super_secret_password123" not in redacted
        assert "[PASSWORD]" in redacted

    def test_redact_token(self, redactor):
        """Tokens should be redacted."""
        text = 'token="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"'
        redacted = redactor.redact(text)
        assert "[TOKEN]" in redacted

    def test_redact_aws_key(self, redactor):
        """AWS keys should be redacted."""
        text = "AWS Key: AKIAIOSFODNN7EXAMPLE"
        redacted = redactor.redact(text)
        assert "AKIAIOSFODNN7EXAMPLE" not in redacted
        assert "[AWS_KEY]" in redacted

    def test_redact_ip_address(self, redactor):
        """IP addresses should be redacted."""
        text = "Server at 192.168.1.100 is down"
        redacted = redactor.redact(text)
        assert "192.168.1.100" not in redacted
        assert "[IP_ADDRESS]" in redacted

    def test_redact_url_with_creds(self, redactor):
        """URLs with embedded credentials should be redacted."""
        text = "Database: https://user:password@db.example.com/prod"
        redacted = redactor.redact(text)
        assert "user:password" not in redacted
        assert "[URL_WITH_CREDS]" in redacted

    def test_clean_text_unchanged(self, redactor):
        """Clean text without sensitive data should be unchanged."""
        text = "What is the capital of France?"
        redacted = redactor.redact(text)
        assert redacted == text

    def test_redact_dict_with_sensitive_keys(self, redactor):
        """Dict values with sensitive keys should be redacted."""
        data = {
            "username": "john",
            "password": "secret123",
            "api_key": "sk_live_xyz",
        }
        redacted = redactor.redact_dict(data)
        assert redacted["password"] == "[PASSWORD]"
        assert redacted["api_key"] == "[API_KEY]"
        assert redacted["username"] == "john"

    def test_redact_dict_with_sensitive_values(self, redactor):
        """Dict values containing sensitive data should be redacted."""
        data = {
            "contact": "john@example.com",
            "phone": "555-123-4567",
        }
        redacted = redactor.redact_dict(data)
        assert "[EMAIL]" in redacted["contact"]
        assert "[PHONE]" in redacted["phone"]

    def test_redact_nested_dict(self, redactor):
        """Nested dicts should be recursively redacted."""
        data = {
            "user": {
                "email": "john@example.com",
                "password": "secret",
            }
        }
        redacted = redactor.redact_dict(data)
        assert "[EMAIL]" in redacted["user"]["email"]
        assert redacted["user"]["password"] == "[PASSWORD]"

    def test_redact_list(self, redactor):
        """Lists should be redacted."""
        data = [
            "john@example.com",
            "555-123-4567",
            "normal text",
        ]
        redacted = redactor.redact_list(data)
        assert "[EMAIL]" in redacted[0]
        assert "[PHONE]" in redacted[1]
        assert redacted[2] == "normal text"

    def test_should_redact_value_sensitive_keys(self, redactor):
        """Sensitive keys should be identified."""
        sensitive_keys = ["password", "api_key", "secret", "token"]
        for key in sensitive_keys:
            assert redactor.should_redact_value(key, "some_value") is True

    def test_should_redact_value_non_sensitive_keys(self, redactor):
        """Non-sensitive keys should not be redacted."""
        non_sensitive_keys = ["username", "email_address", "phone_number"]
        for key in non_sensitive_keys:
            assert redactor.should_redact_value(key, "some_value") is False

    def test_redact_none_values(self, redactor):
        """None values should be handled gracefully."""
        data = {
            "password": None,
            "api_key": None,
        }
        redacted = redactor.redact_dict(data)
        assert redacted["password"] is None
        assert redacted["api_key"] is None

    def test_convenience_function_redact_text(self):
        """Convenience function should work."""
        text = "Email: john@example.com"
        redacted = redact_text(text)
        assert "[EMAIL]" in redacted

    def test_convenience_function_redact_value(self):
        """Convenience function should handle various types."""
        # String
        assert "[EMAIL]" in redact_value("john@example.com")

        # Dict
        redacted_dict = redact_value({"password": "secret"})
        assert redacted_dict["password"] == "[PASSWORD]"

        # List
        redacted_list = redact_value(["john@example.com"])
        assert "[EMAIL]" in redacted_list[0]

    def test_get_redactor_singleton(self):
        """get_redactor should return consistent instance."""
        r1 = get_redactor()
        r2 = get_redactor()
        assert r1 is r2

    def test_redact_multiple_patterns_in_text(self, redactor):
        """Multiple sensitive patterns should be redacted."""
        text = "User: john@example.com, Phone: 555-123-4567, API Key: sk_live_xyz"
        redacted = redactor.redact(text)
        assert "[EMAIL]" in redacted
        assert "[PHONE]" in redacted
        assert "[API_KEY]" in redacted

    def test_case_insensitive_redaction(self, redactor):
        """Redaction should be case insensitive."""
        text1 = "API_KEY=secret123"
        text2 = "api_key=secret123"
        redacted1 = redactor.redact(text1)
        redacted2 = redactor.redact(text2)
        # Both should be redacted (patterns are case-insensitive)
        assert redacted1 == redacted2 or "[" in redacted1


class TestSensitiveDataFilter:
    """Test suite for SensitiveDataFilter logging filter."""

    def test_filter_redacts_message(self):
        """Filter should redact main log message."""
        filter_instance = SensitiveDataFilter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="User email: john@example.com",
            args=(),
            exc_info=None,
        )
        filter_instance.filter(record)
        assert "[EMAIL]" in record.msg

    def test_filter_redacts_args(self):
        """Filter should redact message arguments."""
        filter_instance = SensitiveDataFilter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="User: %s",
            args=("john@example.com",),
            exc_info=None,
        )
        filter_instance.filter(record)
        assert "[EMAIL]" in record.args[0]

    def test_filter_redacts_dict_args(self):
        """Filter should redact sensitive values in dict-style log record args."""
        filter_instance = SensitiveDataFilter()
        # Use dict-style args with named % placeholders — the canonical Python
        # logging pattern for dict args. Two keys avoids a Python 3.13 edge case
        # where LogRecord crashes on a single-key dict via args[0] access.
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Config: %(api_key)s (env: %(env)s)",
            args={"api_key": "secret", "env": "prod"},
            exc_info=None,
        )
        filter_instance.filter(record)
        assert record.args["api_key"] == "[API_KEY]"

    def test_filter_returns_true(self):
        """Filter should always return True to allow logging."""
        filter_instance = SensitiveDataFilter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        result = filter_instance.filter(record)
        assert result is True

    def test_filter_handles_none_values(self):
        """Filter should handle None values gracefully."""
        filter_instance = SensitiveDataFilter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg=None,
            args=None,
            exc_info=None,
        )
        result = filter_instance.filter(record)
        assert result is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
