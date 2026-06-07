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

    def test_custom_patterns_in_constructor(self):
        """Custom patterns are merged into self.patterns (line 96)."""
        custom = {"zip_code": (r"\b\d{5}(?:-\d{4})?\b", "[ZIP]")}
        r = SensitiveDataRedactor(custom_patterns=custom)
        result = r.redact("Zip: 90210")
        assert "[ZIP]" in result

    def test_invalid_regex_raises_value_error(self):
        """Invalid regex in patterns dict raises ValueError (lines 108-109)."""
        bad_patterns = {"bad": (r"[invalid(", "[BAD]")}
        with pytest.raises(ValueError, match="Invalid regex pattern"):
            SensitiveDataRedactor(patterns=bad_patterns)

    def test_redact_non_string_input_returns_input(self):
        """Non-string passed to redact() returns unchanged (line 121)."""
        r = SensitiveDataRedactor()
        assert r.redact(42) == 42
        assert r.redact(None) is None
        assert r.redact([]) == []

    def test_custom_redactor_applied(self, redactor):
        """Custom redactor function is applied after pattern redaction (line 129)."""
        marker = []

        def custom_fn(text):
            marker.append(True)
            return text.replace("foo", "[FOO]")

        r = SensitiveDataRedactor(custom_redactor=custom_fn)
        result = r.redact("foo bar")
        assert "[FOO]" in result
        assert marker  # custom_fn was called

    def test_should_redact_non_string_key_returns_false(self, redactor):
        """Non-string key returns False (line 144)."""
        assert redactor.should_redact_value(123, "secret") is False
        assert redactor.should_redact_value(None, "token") is False

    def test_redact_dict_list_value_with_nested_dict(self, redactor):
        """Dict elements inside list values are recursively redacted (line 170)."""
        data = {
            "items": [
                {"password": "secret", "name": "alice"},
                "plain text",
            ]
        }
        result = redactor.redact_dict(data)
        assert result["items"][0]["password"] == "[PASSWORD]"
        assert result["items"][1] == "plain text"

    def test_redact_list_tuple_input_returns_tuple(self, redactor):
        """redact_list with tuple input returns a tuple (line 193)."""
        data = ("john@example.com", "normal")
        result = redactor.redact_list(data)
        assert isinstance(result, tuple)
        assert "[EMAIL]" in result[0]
        assert result[1] == "normal"

    def test_redact_dict_non_dict_input_returns_unchanged(self, redactor):
        """redact_dict with a non-dict argument returns it unchanged (line 159)."""
        assert redactor.redact_dict("a string") == "a string"
        assert redactor.redact_dict(42) == 42
        assert redactor.redact_dict(None) is None

    def test_redact_list_non_iterable_returns_as_is(self, redactor):
        """redact_list with non-list/tuple returns unchanged (line 211)."""
        result = redactor.redact_list(42)
        assert result == 42

    def test_create_sensitive_redactor_factory(self):
        """create_sensitive_redactor factory function (line 227)."""
        from nemoguardrails.logging.redactor import create_sensitive_redactor

        r = create_sensitive_redactor()
        assert isinstance(r, SensitiveDataRedactor)

    def test_redact_value_non_redactable_type(self):
        """redact_value with int/etc returns value unchanged (line 274)."""
        result = redact_value(42)
        assert result == 42
        result = redact_value(3.14)
        assert result == 3.14


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

    def test_filter_redacts_dict_msg(self):
        """Filter should redact sensitive values when record.msg is a dict (lines 53-54)."""
        filter_instance = SensitiveDataFilter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg={"password": "supersecret", "user": "alice"},
            args=None,
            exc_info=None,
        )
        filter_instance.filter(record)
        assert record.msg["password"] == "[PASSWORD]"
        assert record.msg["user"] == "alice"

    def test_filter_tuple_args_with_dict_item(self):
        """Filter should redact dicts inside tuple args (lines 65-66)."""
        filter_instance = SensitiveDataFilter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Log entry: %s and %s",
            args=({"password": "secret123", "env": "prod"}, {"user": "alice", "env": "dev"}),
            exc_info=None,
        )
        filter_instance.filter(record)
        assert record.args[0]["password"] == "[PASSWORD]"
        assert record.args[1]["user"] == "alice"

    def test_filter_tuple_args_with_non_string_item(self):
        """Non-string, non-dict args items are passed through unchanged (line 68)."""
        filter_instance = SensitiveDataFilter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Count: %s",
            args=(42,),
            exc_info=None,
        )
        filter_instance.filter(record)
        assert record.args[0] == 42

    def test_filter_exc_info_redacts_exception_string(self):
        """Filter should redact sensitive data in exception args (lines 73-76)."""
        filter_instance = SensitiveDataFilter()
        exc = ValueError("password=supersecret connection failed")
        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=1,
            msg="An error occurred",
            args=None,
            exc_info=(type(exc), exc, None),
        )
        filter_instance.filter(record)
        # The exception args should be updated with redacted string
        assert "supersecret" not in str(exc.args[0])
        assert "[PASSWORD]" in str(exc.args[0])

    def test_filter_exc_info_frozen_args_handled(self):
        """Filter handles exc_value.args assignment failure gracefully (lines 78-81)."""
        filter_instance = SensitiveDataFilter()

        class _FrozenArgsExc:
            """Exception-like object with read-only args property."""

            def __str__(self):
                return "password=topsecret"

            @property
            def args(self):
                return ("password=topsecret",)

            @args.setter
            def args(self, value):
                raise AttributeError("args is read-only")

            def __bool__(self):
                return True

        frozen_exc = _FrozenArgsExc()
        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=1,
            msg="Error",
            args=None,
            exc_info=(Exception, frozen_exc, None),
        )
        # Should not raise even though args assignment fails
        result = filter_instance.filter(record)
        assert result is True


class TestSetupSensitiveDataFilter:
    """Tests for setup_sensitive_data_filter and setup_all_loggers."""

    def test_setup_sensitive_data_filter_defaults_to_root_logger(self):
        """When logger=None, setup_sensitive_data_filter uses the root logger (line 100)."""
        from nemoguardrails.logging.sensitive_filter import setup_sensitive_data_filter

        f = setup_sensitive_data_filter()
        assert isinstance(f, SensitiveDataFilter)
        root = logging.getLogger()
        assert any(isinstance(fl, SensitiveDataFilter) for fl in root.filters)

    def test_setup_sensitive_data_filter_returns_existing(self):
        """When filter already exists on logger, return the same instance (line 104)."""
        from nemoguardrails.logging.sensitive_filter import setup_sensitive_data_filter

        test_logger = logging.getLogger("test.setup_filter.idempotent")
        test_logger.filters = []
        try:
            first = setup_sensitive_data_filter(test_logger)
            second = setup_sensitive_data_filter(test_logger)
            assert second is first
            assert len([f for f in test_logger.filters if isinstance(f, SensitiveDataFilter)]) == 1
        finally:
            test_logger.filters = []

    def test_setup_all_loggers_adds_filters(self):
        """setup_all_loggers adds filter to root and named loggers (lines 117-123)."""
        from nemoguardrails.logging.sensitive_filter import setup_all_loggers

        # Just verify it runs without error and the root logger gets a filter
        setup_all_loggers()
        root_logger = logging.getLogger()
        assert any(isinstance(f, SensitiveDataFilter) for f in root_logger.filters)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
