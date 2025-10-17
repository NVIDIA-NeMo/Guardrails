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

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import logging
import os
from unittest.mock import patch

import pytest
import yara
from pydantic import ValidationError

from nemoguardrails import RailsConfig
from nemoguardrails.actions import action
from nemoguardrails.actions.actions import ActionResult
from nemoguardrails.library.injection_detection.actions import (
    _check_yara_available,
    _extract_injection_config,
    _load_rules,
    _omit_injection,
    _reject_injection,
    _sanitize_injection,
    _validate_injection_config,
)
from tests.utils import TestChat

CONFIGS_FOLDER = os.path.join(os.path.dirname(__file__), ".", "test_configs")


#  function to create a mock yara match
def create_mock_yara_match(matched_text: str, rule_name: str = "test_rule"):
    class MockStringMatchInstance:
        def __init__(self, text):
            self._text = text.encode("utf-8")

        def plaintext(self):
            return self._text

    class MockStringMatch:
        def __init__(self, text):
            self.identifier = "test_string"
            self.instances = [MockStringMatchInstance(text)]

    class MockMatch:
        def __init__(self, rule, text):
            self.rule = rule
            self.strings = [MockStringMatch(text)]

    return MockMatch(rule_name, matched_text)


def create_mock_rules(matches=None):
    class MockRules:
        def match(self, data=None, **kwargs):
            return matches if matches is not None else []

    return MockRules()


def test_load_custom_rules():
    yara_path = os.path.join(CONFIGS_FOLDER, "injection_detection")
    config = RailsConfig.from_content(
        yaml_content=f"""
                models: []
                rails:
                  config:
                    injection_detection:
                      injections:
                        - test
                      action:
                        reject
                      yara_path:
                        {yara_path}
            """,
        colang_content="""
                define flow check user message
                  execute check_user_message(user_message=$user_message)
            """,
    )
    _validate_injection_config(config)
    action_option, yara_path, rule_names, yara_rules = _extract_injection_config(config)
    rules = _load_rules(yara_path, rule_names, yara_rules)
    assert isinstance(rules, yara.Rules)


def test_load_all_rules():
    config = RailsConfig.from_path(os.path.join(CONFIGS_FOLDER, "injection_detection"))
    _validate_injection_config(config)
    action_option, yara_path, rule_names, yara_rules = _extract_injection_config(config)
    rules = _load_rules(yara_path, rule_names, yara_rules)
    assert isinstance(rules, yara.Rules)


def test_sqli_rejection():
    sql_injection = "' OR 1 = 1"
    config = RailsConfig.from_path(os.path.join(CONFIGS_FOLDER, "injection_detection"))
    chat = TestChat(config, llm_completions=[sql_injection])
    chat >> "SQL injection"


def test_code_rejection():
    code_injection = "__import__('subprocess').run('touch /tmp/pwnd', shell=True)"
    config = RailsConfig.from_path(os.path.join(CONFIGS_FOLDER, "injection_detection"))
    chat = TestChat(config, llm_completions=[code_injection])
    chat >> "touch the file /tmp/pwnd using subprocess.run"


def test_template_rejection():
    template_injection = (
        "{{self.__init__.__globals__.__builtins__.__import__('os').popen('id').read()}}"
    )
    config = RailsConfig.from_path(os.path.join(CONFIGS_FOLDER, "injection_detection"))
    chat = TestChat(config, llm_completions=[template_injection])
    chat >> "Jinja template to run the 'id' command via os.popen"


def test_xss_rejection():
    xss = "<script>alert('xss')</script>"
    config = RailsConfig.from_path(os.path.join(CONFIGS_FOLDER, "injection_detection"))
    chat = TestChat(config, llm_completions=[xss])
    chat >> "Inject a script"


def test_invalid_yara_path():
    """Test that an invalid YARA path raises error."""
    config = RailsConfig.from_content(
        yaml_content="""
                models: []
                rails:
                  config:
                    injection_detection:
                      injections:
                        - sqli
                      action:
                        reject
                      yara_path:
                        /nonexistent/path
            """
    )
    with pytest.raises(FileNotFoundError):
        _validate_injection_config(config)


def test_invalid_action_option():
    """Test that an invalid action option raises an appropriate error."""
    config = RailsConfig.from_path(os.path.join(CONFIGS_FOLDER, "injection_detection"))
    # we modify the action directly to an invalid value
    config.rails.config.injection_detection.action = "invalid_action"

    with pytest.raises(ValueError):
        _validate_injection_config(config)


def test_invalid_injection_rule():
    """Test that an invalid injection rule raises an error."""
    config = RailsConfig.from_content(
        yaml_content="""
                models: []
                rails:
                  config:
                    injection_detection:
                      injections:
                        - nonexistent_rule
                      action:
                        reject
            """
    )
    _validate_injection_config(config)
    with pytest.raises(ValueError):
        _extract_injection_config(config)


def test_empty_injection_rules():
    """Test that empty injection rules return None from load_rules."""
    config = RailsConfig.from_content(
        yaml_content="""
                models: []
                rails:
                  config:
                    injection_detection:
                      injections: []
                      action:
                        reject
            """
    )
    _validate_injection_config(config)
    action_option, yara_path, rule_names, yara_rules = _extract_injection_config(config)
    rules = _load_rules(yara_path, rule_names, yara_rules)
    assert rules is None


def test_load_inline_yara_rules():
    """Test loading YARA rules defined inline in the config."""
    inline_rule_name = "inline_test_rule"
    inline_rule_content = "rule test_inline { condition: true }"

    config = RailsConfig.from_content(
        yaml_content=f"""
                models: []
                rails:
                  config:
                    injection_detection:
                      injections:
                        - {inline_rule_name}
                      action:
                        reject
                      yara_rules:
                        {inline_rule_name}: |-
                          {inline_rule_content}
            """,
        colang_content="",
    )

    _validate_injection_config(config)
    action_option, yara_path, rule_names, yara_rules = _extract_injection_config(config)

    assert yara_rules is not None
    assert inline_rule_name in yara_rules
    assert yara_rules[inline_rule_name] == inline_rule_content
    assert rule_names == (inline_rule_name,)

    rules = _load_rules(yara_path, rule_names, yara_rules)
    assert isinstance(rules, yara.Rules)

    # Test that the loaded rule actually matches
    matches = rules.match(data="any data")
    assert len(matches) == 1
    assert matches[0].rule == "test_inline"
    assert matches[0].namespace == inline_rule_name


@pytest.mark.asyncio
async def test_omit_injection_action():
    """Test the omit action for injection detection."""

    text = "This is a SELECT * FROM users; -- comment in the middle of text"

    # mock matches for the sql injection parts
    mock_matches = [
        create_mock_yara_match("SELECT * FROM users", "sqli"),
        create_mock_yara_match("-- comment", "sqli"),
    ]

    is_injection, result = _omit_injection(text=text, matches=mock_matches)

    # all sql injection should be removed
    # NOTE: following rule does not get removed using sqli.yara
    assert "SELECT * FROM users" not in result
    assert "-- comment" not in result
    assert "This is a" in result
    assert "in the middle of text" in result


@pytest.mark.asyncio
async def test_reject_injection_with_mismatched_action():
    """Test that reject_injection works even with a mismatched action in config."""
    config = RailsConfig.from_path(os.path.join(CONFIGS_FOLDER, "injection_detection"))

    # change the config to use 'omit' action instead of 'reject'
    config.rails.config.injection_detection.action = "omit"

    mock_match = create_mock_yara_match("' OR 1 = 1", "sqli")
    mock_rules = create_mock_rules([mock_match])

    # pathcing the load_rules function to return our mock rules
    with patch(
        "nemoguardrails.library.injection_detection.actions._load_rules",
        return_value=mock_rules,
    ):
        sql_injection = "' OR 1 = 1"
        result, _ = _reject_injection(sql_injection, mock_rules)
        assert result is True


@pytest.mark.asyncio
async def test_multiple_injection_types():
    """Test detection of multiple injection types in a single string."""
    config = RailsConfig.from_path(os.path.join(CONFIGS_FOLDER, "injection_detection"))

    mock_matches = [
        create_mock_yara_match("' OR 1 = 1", "sqli"),
        create_mock_yara_match("<script>alert('xss')</script>", "xss"),
    ]
    mock_rules = create_mock_rules(mock_matches)

    with patch(
        "nemoguardrails.library.injection_detection.actions._load_rules",
        return_value=mock_rules,
    ):
        multi_injection = "' OR 1 = 1 <script>alert('xss')</script>"
        result, _ = _reject_injection(multi_injection, mock_rules)
        assert result is True


@pytest.mark.asyncio
async def test_edge_cases():
    """Test edge cases for injection detection."""
    config = RailsConfig.from_path(os.path.join(CONFIGS_FOLDER, "injection_detection"))

    mock_rules = create_mock_rules([])

    with patch(
        "nemoguardrails.library.injection_detection.actions._load_rules",
        return_value=mock_rules,
    ):
        # Test with empty string
        result, _ = _reject_injection("", mock_rules)
        assert result is False

        # no issue with very long str
        long_string = "a" * 10000
        result, _ = _reject_injection(long_string, mock_rules)
        assert result is False

        # no issue with special chars
        special_chars = "!@#$%^&*()_+-=[]{}|;:,.<>?"
        result, _ = _reject_injection(special_chars, mock_rules)
        assert result is False


@pytest.mark.asyncio
async def test_omit_action_with_real_yara():
    """Test the omit action for injection detection using real YARA rules from the library."""

    config = RailsConfig.from_content(
        yaml_content="""
                models: []
                rails:
                  config:
                    injection_detection:
                      injections:
                        - sqli
                      action:
                        omit
                  output:
                    flows:
                      - injection detection

                """
    )

    sql_injection = (
        "This is a SELECT * FROM users; -- malicious comment in the middle of text"
    )
    chat = TestChat(config, llm_completions=[sql_injection])
    rails = chat.app
    result = await rails.generate_async(
        messages=[{"role": "user", "content": "do a fake query you funny agent"}]
    )

    assert "--" not in result["content"]
    assert (
        result["content"]
        == "This is a  * FROM usersmalicious comment in the middle of text"
    )


@pytest.mark.asyncio
async def test_xss_injection_with_reject_action():
    """Test the reject action for XSS injection detection using real YARA rules from the library."""

    config = RailsConfig.from_content(
        yaml_content="""
                models: []
                rails:
                  config:
                    injection_detection:
                      injections:
                        - xss
                      action:
                        reject
                  output:
                    flows:
                      - injection detection

                """
    )

    xss_injection = "Hello <script>alert('xss')</script> world"
    chat = TestChat(config, llm_completions=[xss_injection])
    rails = chat.app
    result = await rails.generate_async(
        messages=[{"role": "user", "content": "do a fake query you funny agent"}]
    )

    assert "content" in result, f"Expected 'content' key in result, got {result}"
    actual_content = result["content"]
    expected_start = "I'm sorry, the desired output triggered rule(s) designed to mitigate exploitation of"
    assert actual_content.startswith(
        expected_start
    ), f"Expected content to start with '{expected_start}', but got '{actual_content}'"


@pytest.mark.asyncio
async def test_template_injection_with_reject_action():
    """Test the reject action for template injection detection using real YARA rules from the library."""

    config = RailsConfig.from_content(
        yaml_content="""
                models: []
                rails:
                  config:
                    injection_detection:
                      injections:
                        - template
                      action:
                        reject
                  output:
                    flows:
                      - injection detection

                """
    )

    template_injection = "Hello {{self.__init__.__globals__.__builtins__.__import__('os').popen('id').read()}} world"
    chat = TestChat(config, llm_completions=[template_injection])
    rails = chat.app
    result = await rails.generate_async(
        messages=[{"role": "user", "content": "do a fake query you funny agent"}]
    )

    assert result["content"].startswith(
        "I'm sorry, the desired output triggered rule(s) designed to mitigate exploitation of"
    )


@pytest.mark.asyncio
async def test_code_injection_with_reject_action():
    """Test the reject action for code injection detection using real YARA rules from the library."""

    config = RailsConfig.from_content(
        yaml_content="""
                models: []
                rails:
                  config:
                    injection_detection:
                      injections:
                        - code
                      action:
                        reject
                  output:
                    flows:
                      - injection detection

                """
    )

    code_injection = (
        "Hello __import__('subprocess').run('touch /tmp/pwnd', shell=True) world"
    )
    chat = TestChat(config, llm_completions=[code_injection])
    rails = chat.app
    result = await rails.generate_async(
        messages=[{"role": "user", "content": "do a fake query you funny agent"}]
    )

    assert result["content"].startswith(
        "I'm sorry, the desired output triggered rule(s) designed to mitigate exploitation of"
    )


@pytest.mark.asyncio
async def test_multiple_injection_types_with_reject_action():
    """Test the reject action for multiple injection types using real YARA rules from the library."""

    config = RailsConfig.from_content(
        yaml_content="""
                models: []
                rails:
                  config:
                    injection_detection:
                      injections:
                        - sqli
                        - xss
                        - template
                        - code
                      action:
                        reject
                  output:
                    flows:
                      - injection detection

                """
    )

    multi_injection = "Hello <script>alert('xss')</script> {{self.__init__.__globals__.__builtins__.__import__('os').popen('id').read()}} __import__('subprocess').run('touch /tmp/pwnd', shell=True) SELECT * FROM users; -- comment world"
    chat = TestChat(config, llm_completions=[multi_injection])
    rails = chat.app
    result = await rails.generate_async(
        messages=[{"role": "user", "content": "do a fake query you funny agent"}]
    )

    assert result["content"].startswith(
        "I'm sorry, the desired output triggered rule(s) designed to mitigate exploitation of"
    )


@pytest.mark.asyncio
async def test_sanitize_action_not_implemented():
    """Test that the sanitize action raises NotImplementedError when used with real YARA rules."""

    with pytest.raises(ValidationError):
        _ = RailsConfig.from_content(
            yaml_content="""
                models: []
                rails:
                  config:
                    injection_detection:
                      injections:
                        - sqli
                        - xss
                        - template
                        - code
                      action:
                        sanitize
                  output:
                    flows:
                      - injection detection

                    """
        )


def test_yara_import_error():
    """Test that appropriate error is raised when yara module is not available."""

    with patch("nemoguardrails.library.injection_detection.actions.yara", None):
        with pytest.raises(ImportError) as exc_info:
            _check_yara_available()
        assert str(exc_info.value) == (
            "The yara module is required for injection detection. "
            "Please install it using: pip install yara-python"
        )

    with patch("nemoguardrails.library.injection_detection.actions.yara", yara):
        _check_yara_available()


@pytest.mark.asyncio
async def test_multiple_injection_types_reject_inline_rules():
    """Test reject action for multiple injection types using inline YARA rules."""

    # inline YARA rules
    sqli_rule_content = (
        "rule simple_sqli { strings: $sql = /SELECT.*FROM/ condition: $sql }"
    )
    xss_rule_content = "rule simple_xss { strings: $tag = /<script/ condition: $tag }"
    template_rule_content = (
        "rule simple_template { strings: $tpl = /{{.*}}/ condition: $tpl }"
    )
    code_rule_content = (
        "rule simple_code { strings: $code = /__import__/ condition: $code }"
    )

    config = RailsConfig.from_content(
        yaml_content=f"""
                models: []
                rails:
                  config:
                    injection_detection:
                      injections:
                        - sqli_inline
                        - xss_inline
                        - template_inline
                        - code_inline
                      action:
                        reject
                      yara_rules:
                        sqli_inline: |
                          {sqli_rule_content}
                        xss_inline: |
                          {xss_rule_content}
                        template_inline: |
                          {template_rule_content}
                        code_inline: |
                          {code_rule_content}
                  output:
                    flows:
                      - injection detection
            """,
        colang_content="",
    )

    multi_injection = "Hello <script>alert('xss')</script> {{ evil }} __import__('os') SELECT * FROM users; -- comment world"
    chat = TestChat(config, llm_completions=[multi_injection])
    rails = chat.app
    result = await rails.generate_async(
        messages=[{"role": "user", "content": "trigger multiple injections"}]
    )

    assert result["content"].startswith(
        "I'm sorry, the desired output triggered rule(s) designed to mitigate exploitation of"
    )
    assert "simple_sqli" in result["content"]
    assert "simple_xss" in result["content"]
    assert "simple_template" in result["content"]
    assert "simple_code" in result["content"]


@pytest.mark.asyncio
async def test_xss_injection_with_exception_action():
    """Test the exception action for XSS injection detection using real YARA rules from the library."""

    config = RailsConfig.from_content(
        yaml_content="""
                models: []
                enable_rails_exceptions: True
                rails:
                  config:
                    injection_detection:
                      injections:
                        - xss
                      action:
                        reject
                  output:
                    flows:
                      - injection detection

                """
    )

    xss_injection = "Hello <script>alert('xss')</script> world"
    chat = TestChat(config, llm_completions=[xss_injection])
    rails = chat.app
    result = await rails.generate_async(
        messages=[{"role": "user", "content": "do a fake query you funny agent"}]
    )

    assert (
        result.get("role") == "exception"
    ), f"Expected role 'exception', got {result.get('role')}"
    assert "content" in result, "Expected 'content' key in result"
    content = result["content"]
    assert (
        content.get("type") == "InjectionDetectionRailException"
    ), f"Expected type 'InjectionDetectionRailException', got {content.get('type')}"
    expected_message = (
        "Output not allowed. The output was blocked by the 'injection detection' flow."
    )
    assert (
        content.get("message") == expected_message
    ), f"Expected message '{expected_message}', got '{content.get('message')}'"


@pytest.mark.asyncio
async def test_omit_action_with_exceptions_enabled():
    """Test that omit action does not raise an exception when enable_rails_exceptions is True."""

    config = RailsConfig.from_content(
        yaml_content="""
                models: []
                enable_rails_exceptions: True
                rails:
                  config:
                    injection_detection:
                      injections:
                        - xss
                      action:
                        omit
                  output:
                    flows:
                      - injection detection

                """
    )

    xss_injection = "Hello <script>alert('xss')</script> world"
    chat = TestChat(config, llm_completions=[xss_injection])
    rails = chat.app
    result = await rails.generate_async(
        messages=[{"role": "user", "content": "do a fake query you funny agent"}]
    )

    # check that an exception is raised
    assert result.get("role") == "exception", "Expected role to be 'exception'"

    # verify exception details
    content = result["content"]
    assert (
        content.get("type") == "InjectionDetectionRailException"
    ), f"Expected type 'InjectionDetectionRailException', got {content.get('type')}"

    expected_message = (
        "Output not allowed. The output was blocked by the 'injection detection' flow."
    )
    assert (
        content.get("message") == expected_message
    ), f"Expected message '{expected_message}', got '{content.get('message')}'"


@pytest.mark.asyncio
async def test_malformed_inline_yara_rule_fails_gracefully(caplog):
    """Test that a malformed inline YARA rule leads to graceful failure (detection becomes no-op)."""

    inline_rule_name = "malformed_rule"
    # this rule is malformed: missing { after rule name
    malformed_rule_content = "rule malformed_rule condition: true "

    config = RailsConfig.from_content(
        yaml_content=f"""
                models: []
                rails:
                  config:
                    injection_detection:
                      injections:
                        - {inline_rule_name}
                      action:
                        reject # can be anything
                      yara_rules:
                        {inline_rule_name}: |
                          {malformed_rule_content}
                  output:
                    flows:
                      - injection detection
            """,
        colang_content="",
    )

    some_text_that_would_be_injection = "This is a test string."

    caplog.set_level(logging.ERROR, logger="actions.py")

    chat = TestChat(config, llm_completions=[some_text_that_would_be_injection])
    rails = chat.app

    assert rails is not None

    result = await rails.generate_async(
        messages=[{"role": "user", "content": "trigger detection"}]
    )

    # check that no exception was raised
    assert result.get("role") != "exception", f"Expected no exception, but got {result}"

    # verify the error log was created with the expected content
    assert any(
        record.name == "actions.py" and record.levelno == logging.ERROR
        # minor variations in the error message are expected
        and "Failed to initialize injection detection" in record.message
        and "YARA compilation failed" in record.message
        and "syntax error" in record.message
        for record in caplog.records
    ), "Expected error log message about YARA compilation failure not found"


@pytest.mark.asyncio
async def test_omit_injection_attribute_error():
    """Test error handling in _omit_injection for AttributeError."""

    text = "test text"
    mock_matches = [
        create_mock_yara_match(
            "invalid bytes", "test_rule"
        )  # This will cause AttributeError
    ]

    is_injection, result = _omit_injection(text=text, matches=mock_matches)
    assert not is_injection
    assert result == text


@pytest.mark.asyncio
async def test_omit_injection_unicode_decode_error():
    """Test error handling in _omit_injection for UnicodeDecodeError."""

    text = "test text"

    class MockStringMatchInstanceUnicode:
        def __init__(self):
            # invalid utf-8 bytes
            self._text = b"\xff\xfe"

        def plaintext(self):
            return self._text

    class MockStringMatchUnicode:
        def __init__(self):
            self.identifier = "test_string"
            self.instances = [MockStringMatchInstanceUnicode()]

    class MockMatchUnicode:
        def __init__(self, rule):
            self.rule = rule
            self.strings = [MockStringMatchUnicode()]

    mock_matches = [MockMatchUnicode("test_rule")]
    is_injection, result = _omit_injection(text=text, matches=mock_matches)
    assert not is_injection
    assert result == text


@pytest.mark.asyncio
async def test_omit_injection_no_modifications():
    """Test _omit_injection when no modifications are made to the text."""

    text = "safe text"
    mock_matches = [create_mock_yara_match("nonexistent pattern", "test_rule")]

    is_injection, result = _omit_injection(text=text, matches=mock_matches)
    assert not is_injection
    assert result == text


@pytest.mark.asyncio
async def test_sanitize_injection_not_implemented():
    """Test that _sanitize_injection raises NotImplementedError."""

    text = "test text"
    mock_matches = [create_mock_yara_match("test pattern", "test_rule")]

    with pytest.raises(NotImplementedError) as exc_info:
        _sanitize_injection(text=text, matches=mock_matches)
    assert "Injection sanitization is not yet implemented" in str(exc_info.value)


@pytest.mark.asyncio
async def test_reject_injection_no_rules(caplog):
    """Test _reject_injection when no rules are specified."""

    text = "test text"
    caplog.set_level(logging.WARNING)

    is_injection, detections = _reject_injection(text=text, rules=None)
    assert not is_injection
    assert detections == []
    assert any(
        "reject_injection guardrail was invoked but no rules were specified"
        in record.message
        for record in caplog.records
    )
