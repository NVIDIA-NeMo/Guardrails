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

import pytest

from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.imports import check_optional_dependency
from tests.utils import FakeLLM, TestChat

GUARDRAILS_AVAILABLE = check_optional_dependency("guardrails")
REGEX_MATCH_AVAILABLE = False
VALID_LENGTH_AVAILABLE = False

if GUARDRAILS_AVAILABLE:
    REGEX_MATCH_AVAILABLE = check_optional_dependency("guardrails.hub")
    VALID_LENGTH_AVAILABLE = check_optional_dependency("guardrails.hub")
else:
    GUARDRAILS_AVAILABLE = False
    REGEX_MATCH_AVAILABLE = False
    VALID_LENGTH_AVAILABLE = False


INPUT_RAILS_ONLY_CONFIG_EXCEPTION = """
models:
  - type: main
    engine: fake
    model: fake

enable_rails_exceptions: true

rails:
  config:
    guardrails_ai:
      validators:
        - name: regex_match
          parameters:
            regex: "^[A-Z].*"
          metadata: {}

  input:
    flows:
      - guardrailsai check input $validator="regex_match"
"""

INPUT_RAILS_ONLY_CONFIG_REFUSE = """
models:
  - type: main
    engine: fake
    model: fake

enable_rails_exceptions: false

rails:
  config:
    guardrails_ai:
      validators:
        - name: regex_match
          parameters:
            regex: "^[A-Z].*"
          metadata: {}

  input:
    flows:
      - guardrailsai check input $validator="regex_match"
"""

OUTPUT_RAILS_ONLY_CONFIG_EXCEPTION = """
models:
  - type: main
    engine: fake
    model: fake

enable_rails_exceptions: true

rails:
  config:
    guardrails_ai:
      validators:
        - name: valid_length
          parameters:
            min: 1
            max: 20
          metadata: {}

  output:
    flows:
      - guardrailsai check output $validator="valid_length"
"""

OUTPUT_RAILS_ONLY_CONFIG_REFUSE = """
models:
  - type: main
    engine: fake
    model: fake

enable_rails_exceptions: false

rails:
  config:
    guardrails_ai:
      validators:
        - name: valid_length
          parameters:
            min: 1
            max: 20
          metadata: {}

  output:
    flows:
      - guardrailsai check output $validator="valid_length"
"""

INPUT_AND_OUTPUT_RAILS_CONFIG_EXCEPTION = """
models:
  - type: main
    engine: fake
    model: fake

enable_rails_exceptions: true

rails:
  config:
    guardrails_ai:
      validators:
        - name: regex_match
          parameters:
            regex: "^[A-Z].*"
          metadata: {}
        - name: valid_length
          parameters:
            min: 1
            max: 30
          metadata: {}

  input:
    flows:
      - guardrailsai check input $validator="regex_match"

  output:
    flows:
      - guardrailsai check output $validator="valid_length"
"""

COLANG_CONTENT = """
define user express greeting
  "hello"
  "hi"
  "hey"

define bot express greeting
  "Hello! How can I help you today?"

define bot refuse to respond
  "I can't help with that request."

define flow greeting
  user express greeting
  bot express greeting
"""

OUTPUT_RAILS_COLANG_CONTENT = """
define user express greeting
  "hello"
  "hi"
  "hey"

define bot refuse to respond
  "I can't help with that request."

define flow greeting
  user express greeting
  # No predefined bot response - will be LLM generated
"""


class TestGuardrailsAIBlockingBehavior:
    @pytest.mark.skipif(
        not GUARDRAILS_AVAILABLE or not REGEX_MATCH_AVAILABLE,
        reason="Guardrails or RegexMatch validator not installed",
    )
    def test_input_rails_only_validation_passes(self):
        """Test input rails when validation passes - conversation continues normally."""
        config = RailsConfig.from_content(
            colang_content=COLANG_CONTENT,
            yaml_content=INPUT_RAILS_ONLY_CONFIG_EXCEPTION,
        )

        chat = TestChat(
            config,
            llm_completions=["  express greeting", "Hello! How can I help you today?"],
        )

        chat.user("Hello there!")
        chat.bot("Hello! How can I help you today?")

        assert len(chat.history) == 2
        assert chat.history[0]["role"] == "user"
        assert chat.history[0]["content"] == "Hello there!"
        assert chat.history[1]["role"] == "assistant"
        assert "Hello" in chat.history[1]["content"]

    @pytest.mark.skipif(
        not GUARDRAILS_AVAILABLE or not REGEX_MATCH_AVAILABLE,
        reason="Guardrails or RegexMatch validator not installed",
    )
    def test_input_rails_only_validation_blocks_with_exception(self):
        """Test input rails when validation fails - blocked with exception."""
        config = RailsConfig.from_content(
            colang_content=COLANG_CONTENT,
            yaml_content=INPUT_RAILS_ONLY_CONFIG_EXCEPTION,
        )

        llm = FakeLLM(responses=["  express greeting", "Hello! How can I help you today?"])

        rails = LLMRails(config=config, llm=llm)

        result = rails.generate(messages=[{"role": "user", "content": "hello there!"}])

        assert result["role"] == "exception"
        assert result["content"]["type"] == "GuardrailsAIException"
        assert "Guardrails AI regex_match validation failed" in result["content"]["message"]

    @pytest.mark.skipif(
        not GUARDRAILS_AVAILABLE or not REGEX_MATCH_AVAILABLE,
        reason="Guardrails or RegexMatch validator not installed",
    )
    def test_input_rails_only_validation_blocks_with_refuse(self):
        """Test input rails when validation fails - blocked with bot refuse."""
        config = RailsConfig.from_content(colang_content=COLANG_CONTENT, yaml_content=INPUT_RAILS_ONLY_CONFIG_REFUSE)

        chat = TestChat(
            config,
            llm_completions=["  express greeting", "Hello! How can I help you today?"],
        )

        chat.user("hello there!")
        chat.bot("I can't help with that request.")

        assert len(chat.history) == 2
        assert chat.history[0]["role"] == "user"
        assert chat.history[0]["content"] == "hello there!"
        assert chat.history[1]["role"] == "assistant"
        assert "can't" in chat.history[1]["content"].lower()

    @pytest.mark.skipif(
        not GUARDRAILS_AVAILABLE or not VALID_LENGTH_AVAILABLE,
        reason="Guardrails or ValidLength validator not installed",
    )
    def test_output_rails_only_validation_passes(self):
        """Test output rails when validation passes - response is allowed."""
        config = RailsConfig.from_content(
            colang_content=OUTPUT_RAILS_COLANG_CONTENT,
            yaml_content=OUTPUT_RAILS_ONLY_CONFIG_EXCEPTION,
        )

        chat = TestChat(
            config,
            llm_completions=["  express greeting", "general response", "Hi!"],
        )

        chat.user("Hello")
        chat.bot("Hi!")

        assert len(chat.history) == 2
        assert chat.history[0]["role"] == "user"
        assert chat.history[0]["content"] == "Hello"
        assert chat.history[1]["role"] == "assistant"
        assert chat.history[1]["content"] == "Hi!"

    @pytest.mark.skipif(
        not GUARDRAILS_AVAILABLE or not VALID_LENGTH_AVAILABLE,
        reason="Guardrails or ValidLength validator not installed",
    )
    def test_output_rails_only_validation_blocks_with_exception(self):
        """Test output rails when validation fails - blocked with exception."""
        config = RailsConfig.from_content(
            colang_content=OUTPUT_RAILS_COLANG_CONTENT,
            yaml_content=OUTPUT_RAILS_ONLY_CONFIG_EXCEPTION,
        )

        llm = FakeLLM(
            responses=[
                "  express greeting",
                "general response",
                "This is a very long response that exceeds the maximum length limit set in the validator configuration",
            ]
        )

        rails = LLMRails(config=config, llm=llm)

        result = rails.generate(messages=[{"role": "user", "content": "Hello"}])

        assert result["role"] == "exception"
        assert result["content"]["type"] == "GuardrailsAIException"
        assert "Guardrails AI valid_length validation failed" in result["content"]["message"]

    @pytest.mark.skipif(
        not GUARDRAILS_AVAILABLE or not VALID_LENGTH_AVAILABLE,
        reason="Guardrails or ValidLength validator not installed",
    )
    def test_output_rails_only_validation_blocks_with_refuse(self):
        """Test output rails when validation fails - blocked with bot refuse."""
        config = RailsConfig.from_content(
            colang_content=OUTPUT_RAILS_COLANG_CONTENT,
            yaml_content=OUTPUT_RAILS_ONLY_CONFIG_REFUSE,
        )

        chat = TestChat(
            config,
            llm_completions=[
                "  express greeting",
                "general response",
                "This is a very long response that exceeds the maximum length limit set in the validator configuration",
            ],
        )

        chat.user("Hello")
        chat.bot("I can't help with that request.")

        assert len(chat.history) == 2
        assert chat.history[0]["role"] == "user"
        assert chat.history[0]["content"] == "Hello"
        assert chat.history[1]["role"] == "assistant"
        assert "can't" in chat.history[1]["content"].lower()

    @pytest.mark.skipif(
        not GUARDRAILS_AVAILABLE or not REGEX_MATCH_AVAILABLE or not VALID_LENGTH_AVAILABLE,
        reason="Guardrails, RegexMatch, or ValidLength validator not installed",
    )
    def test_input_and_output_rails_both_pass(self):
        """Test input+output rails when both validations pass - conversation flows normally."""
        config = RailsConfig.from_content(
            colang_content=OUTPUT_RAILS_COLANG_CONTENT,
            yaml_content=INPUT_AND_OUTPUT_RAILS_CONFIG_EXCEPTION,
        )

        chat = TestChat(
            config,
            llm_completions=[
                "  express greeting",
                "general response",
                "Hello! How are you?",
            ],
        )

        chat.user("Hello there!")
        chat.bot("Hello! How are you?")

        assert len(chat.history) == 2
        assert chat.history[0]["role"] == "user"
        assert chat.history[0]["content"] == "Hello there!"
        assert chat.history[1]["role"] == "assistant"
        assert chat.history[1]["content"] == "Hello! How are you?"

    @pytest.mark.skipif(
        not GUARDRAILS_AVAILABLE or not REGEX_MATCH_AVAILABLE,
        reason="Guardrails or RegexMatch validator not installed",
    )
    def test_input_and_output_rails_input_blocks_with_exception(self):
        """Test input+output rails when input validation fails - blocked at input with exception."""
        config = RailsConfig.from_content(
            colang_content=OUTPUT_RAILS_COLANG_CONTENT,
            yaml_content=INPUT_AND_OUTPUT_RAILS_CONFIG_EXCEPTION,
        )

        llm = FakeLLM(responses=["  express greeting", "general response", "Hello! How are you?"])

        rails = LLMRails(config=config, llm=llm)

        result = rails.generate(messages=[{"role": "user", "content": "hello there!"}])

        assert result["role"] == "exception"
        assert result["content"]["type"] == "GuardrailsAIException"
        assert "Guardrails AI regex_match validation failed" in result["content"]["message"]

    @pytest.mark.skipif(
        not GUARDRAILS_AVAILABLE or not REGEX_MATCH_AVAILABLE or not VALID_LENGTH_AVAILABLE,
        reason="Guardrails, RegexMatch, or ValidLength validator not installed",
    )
    def test_input_and_output_rails_output_blocks_with_exception(self):
        """Test input+output rails when output validation fails - blocked at output with exception."""
        config = RailsConfig.from_content(
            colang_content=OUTPUT_RAILS_COLANG_CONTENT,
            yaml_content=INPUT_AND_OUTPUT_RAILS_CONFIG_EXCEPTION,
        )

        llm = FakeLLM(
            responses=[
                "  express greeting",
                "general response",
                "This is a very long response that definitely exceeds the maximum length limit",
            ]
        )

        rails = LLMRails(config=config, llm=llm)

        result = rails.generate(messages=[{"role": "user", "content": "Hello there!"}])

        assert result["role"] == "exception"
        assert result["content"]["type"] == "GuardrailsAIException"
        assert "Guardrails AI valid_length validation failed" in result["content"]["message"]

    def test_config_structures_are_valid(self):
        """Test that all config structures parse correctly."""

        input_config = RailsConfig.from_content(
            colang_content=COLANG_CONTENT,
            yaml_content=INPUT_RAILS_ONLY_CONFIG_EXCEPTION,
        )
        assert input_config.rails.config.guardrails_ai is not None
        assert len(input_config.rails.input.flows) == 1
        assert len(input_config.rails.output.flows) == 0

        output_config = RailsConfig.from_content(
            colang_content=COLANG_CONTENT,
            yaml_content=OUTPUT_RAILS_ONLY_CONFIG_EXCEPTION,
        )
        assert output_config.rails.config.guardrails_ai is not None
        assert len(output_config.rails.input.flows) == 0
        assert len(output_config.rails.output.flows) == 1

        both_config = RailsConfig.from_content(
            colang_content=COLANG_CONTENT,
            yaml_content=INPUT_AND_OUTPUT_RAILS_CONFIG_EXCEPTION,
        )
        assert both_config.rails.config.guardrails_ai is not None
        assert len(both_config.rails.input.flows) == 1
        assert len(both_config.rails.output.flows) == 1

    def test_validator_configurations_are_accessible(self):
        """Test that validator configurations can be accessed properly."""

        config = RailsConfig.from_content(
            colang_content=COLANG_CONTENT,
            yaml_content=INPUT_AND_OUTPUT_RAILS_CONFIG_EXCEPTION,
        )

        guardrails_config = config.rails.config.guardrails_ai

        regex_validator = guardrails_config.get_validator_config("regex_match")
        assert regex_validator.name == "regex_match"
        assert regex_validator.parameters["regex"] == "^[A-Z].*"

        length_validator = guardrails_config.get_validator_config("valid_length")
        assert length_validator.name == "valid_length"
        assert length_validator.parameters["min"] == 1
        assert length_validator.parameters["max"] == 30


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
