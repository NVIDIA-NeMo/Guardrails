# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    _load_rules,
    _omit_injection,
    _reject_injection,
    _validate_unpack_config,
    injection_detection,
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
    _action_option, yara_path, rule_names = _validate_unpack_config(config)
    rules = _load_rules(yara_path, rule_names)
    assert isinstance(rules, yara.Rules)


def test_load_all_rules():
    config = RailsConfig.from_path(os.path.join(CONFIGS_FOLDER, "injection_detection"))
    _action_option, yara_path, rule_names = _validate_unpack_config(config)
    rules = _load_rules(yara_path, rule_names)
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
        _validate_unpack_config(config)


def test_invalid_action_option():
    """Test that an invalid action option raises an appropriate error."""
    config = RailsConfig.from_path(os.path.join(CONFIGS_FOLDER, "injection_detection"))
    # we modify the action directly to an invalid value
    config.rails.config.injection_detection.action = "invalid_action"

    with pytest.raises(ValueError):
        _validate_unpack_config(config)


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
    with pytest.raises(ValueError):
        _validate_unpack_config(config)


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
    _action_option, yara_path, rule_names = _validate_unpack_config(config)
    rules = _load_rules(yara_path, rule_names)
    assert rules is None


@pytest.mark.asyncio
async def test_omit_injection_action():
    """Test the omit action for injection detection."""

    text = "This is a SELECT * FROM users; -- comment in the middle of text"

    # mock matches for the sql injection parts
    mock_matches = [
        create_mock_yara_match("SELECT * FROM users", "sqli"),
        create_mock_yara_match("-- comment", "sqli"),
    ]

    result = _omit_injection(text=text, matches=mock_matches)

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

    assert result["content"].startswith(
        "I'm sorry, the desired output triggered rule(s) designed to mitigate exploitation of"
    )


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
