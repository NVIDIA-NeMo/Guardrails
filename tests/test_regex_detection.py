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

import pytest
from pydantic import ValidationError

from nemoguardrails import RailsConfig
from nemoguardrails.actions import action
from nemoguardrails.actions.actions import ActionResult
from tests.utils import TestChat


@pytest.mark.unit
def test_regex_detection_input_blocks_matching_pattern():
    """Test that input containing a matching regex pattern is blocked."""
    config = RailsConfig.from_content(
        yaml_content="""
            models: []
            rails:
              config:
                regex_detection:
                  input:
                    patterns:
                      - "\\\\d{3}-\\\\d{2}-\\\\d{4}"
              input:
                flows:
                  - regex check input
        """,
        colang_content="""
            define user express greeting
              "hi"

            define flow
              user express greeting
              bot express greeting

            define bot refuse to respond
              "I'm sorry, I can't respond to that."
        """,
    )

    chat = TestChat(
        config,
        llm_completions=["  express greeting", '  "Hello there!"'],
    )

    # This should trigger the input rail (contains SSN pattern)
    chat >> "My SSN is 123-45-6789"
    chat << "I'm sorry, I can't respond to that."


@pytest.mark.unit
def test_regex_detection_input_allows_non_matching():
    """Test that input not matching any regex pattern is allowed."""
    config = RailsConfig.from_content(
        yaml_content="""
            models: []
            rails:
              config:
                regex_detection:
                  input:
                    patterns:
                      - "\\\\d{3}-\\\\d{2}-\\\\d{4}"
              input:
                flows:
                  - regex check input
        """,
        colang_content="""
            define user express greeting
              "hi"

            define flow
              user express greeting
              bot express greeting

            define bot refuse to respond
              "I'm sorry, I can't respond to that."
        """,
    )

    chat = TestChat(
        config,
        llm_completions=["  express greeting", '  "Hello there!"'],
    )

    # This should NOT trigger the input rail (no SSN pattern)
    chat >> "Hi there!"
    chat << "Hello there!"


@pytest.mark.unit
def test_regex_detection_output_blocks_matching_pattern():
    """Test that output containing a matching regex pattern is blocked."""
    config = RailsConfig.from_content(
        yaml_content="""
            models: []
            rails:
              config:
                regex_detection:
                  output:
                    patterns:
                      - "\\\\bconfidential\\\\b"
              output:
                flows:
                  - regex check output
        """,
        colang_content="""
            define user express greeting
              "hi"

            define flow
              user express greeting
              bot express greeting

            define bot refuse to respond
              "I'm sorry, I can't respond to that."
        """,
    )

    chat = TestChat(
        config,
        llm_completions=["  express greeting", '  "This is confidential information."'],
    )

    # The LLM output contains "confidential" which should be blocked
    chat >> "Hi!"
    chat << "I'm sorry, I can't respond to that."


@pytest.mark.unit
def test_regex_detection_case_insensitive():
    """Test that case insensitive matching works correctly."""
    config = RailsConfig.from_content(
        yaml_content="""
            models: []
            rails:
              config:
                regex_detection:
                  input:
                    patterns:
                      - "\\\\bpassword\\\\b"
                    case_insensitive: true
              input:
                flows:
                  - regex check input
        """,
        colang_content="""
            define user express greeting
              "hi"

            define flow
              user express greeting
              bot express greeting

            define bot refuse to respond
              "I'm sorry, I can't respond to that."
        """,
    )

    chat = TestChat(
        config,
        llm_completions=["  express greeting", '  "Hello!"'],
    )

    # Should match regardless of case
    chat >> "My PASSWORD is secret"
    chat << "I'm sorry, I can't respond to that."


@pytest.mark.unit
def test_regex_detection_case_sensitive():
    """Test that case sensitive matching works correctly (default behavior)."""
    config = RailsConfig.from_content(
        yaml_content="""
            models: []
            rails:
              config:
                regex_detection:
                  input:
                    patterns:
                      - "\\\\bpassword\\\\b"
                    case_insensitive: false
              input:
                flows:
                  - regex check input
        """,
        colang_content="""
            define user express greeting
              "hi"

            define flow
              user express greeting
              bot express greeting

            define bot refuse to respond
              "I'm sorry, I can't respond to that."
        """,
    )

    chat = TestChat(
        config,
        llm_completions=["  express greeting", '  "Hello!"'],
    )

    # Should NOT match because case is different
    chat >> "My PASSWORD is secret"
    chat << "Hello!"


@pytest.mark.unit
def test_regex_detection_multiple_patterns():
    """Test that multiple regex patterns work correctly."""
    config = RailsConfig.from_content(
        yaml_content="""
            models: []
            rails:
              config:
                regex_detection:
                  input:
                    patterns:
                      - "\\\\bsecret\\\\b"
                      - "\\\\bpassword\\\\b"
                      - "\\\\bapi[_-]?key\\\\b"
              input:
                flows:
                  - regex check input
        """,
        colang_content="""
            define user express greeting
              "hi"

            define flow
              user express greeting
              bot express greeting

            define bot refuse to respond
              "I'm sorry, I can't respond to that."
        """,
    )

    chat = TestChat(
        config,
        llm_completions=[
            "  express greeting",
            '  "Hello!"',
            "  express greeting",
            '  "Hello!"',
            "  express greeting",
            '  "Hello!"',
        ],
    )

    # First pattern should match
    chat >> "This is a secret message"
    chat << "I'm sorry, I can't respond to that."

    # Second pattern should match
    chat >> "My password is 12345"
    chat << "I'm sorry, I can't respond to that."

    # Third pattern should match (with hyphen variant)
    chat >> "The api-key is abc123"
    chat << "I'm sorry, I can't respond to that."


@pytest.mark.unit
def test_regex_detection_empty_patterns_allows_all():
    """Test that empty patterns list allows all input."""
    config = RailsConfig.from_content(
        yaml_content="""
            models: []
            rails:
              config:
                regex_detection:
                  input:
                    patterns: []
              input:
                flows:
                  - regex check input
        """,
        colang_content="""
            define user express greeting
              "hi"

            define flow
              user express greeting
              bot express greeting

            define bot refuse to respond
              "I'm sorry, I can't respond to that."
        """,
    )

    chat = TestChat(
        config,
        llm_completions=["  express greeting", '  "Hello!"'],
    )

    # Should pass because no patterns are defined
    chat >> "My password is secret and my SSN is 123-45-6789"
    chat << "Hello!"


@pytest.mark.unit
def test_regex_detection_input_and_output():
    """Test regex detection on both input and output."""
    config = RailsConfig.from_content(
        yaml_content="""
            models: []
            rails:
              config:
                regex_detection:
                  input:
                    patterns:
                      - "\\\\bpassword\\\\b"
                  output:
                    patterns:
                      - "\\\\bsecret\\\\b"
              input:
                flows:
                  - regex check input
              output:
                flows:
                  - regex check output
        """,
        colang_content="""
            define user express greeting
              "hi"

            define flow
              user express greeting
              bot express greeting

            define bot refuse to respond
              "I'm sorry, I can't respond to that."
        """,
    )

    chat = TestChat(
        config,
        llm_completions=[
            "  express greeting",
            '  "Here is a secret!"',
            "  express greeting",
            '  "Hello!"',
        ],
    )

    # Input should pass, but output should be blocked
    chat >> "Hi there!"
    chat << "I'm sorry, I can't respond to that."

    # Input should be blocked
    chat >> "My password is 12345"
    chat << "I'm sorry, I can't respond to that."


@pytest.mark.unit
def test_regex_detection_complex_patterns():
    """Test regex detection with complex patterns like email and phone."""
    config = RailsConfig.from_content(
        yaml_content="""
            models: []
            rails:
              config:
                regex_detection:
                  input:
                    patterns:
                      - "[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\\\.[a-zA-Z]{2,}"
                      - "\\\\(?\\\\d{3}\\\\)?[-.\\\\s]?\\\\d{3}[-.\\\\s]?\\\\d{4}"
              input:
                flows:
                  - regex check input
        """,
        colang_content="""
            define user express greeting
              "hi"

            define flow
              user express greeting
              bot express greeting

            define bot refuse to respond
              "I'm sorry, I can't respond to that."
        """,
    )

    chat = TestChat(
        config,
        llm_completions=[
            "  express greeting",
            '  "Hello!"',
            "  express greeting",
            '  "Hello!"',
        ],
    )

    # Email pattern should match
    chat >> "Contact me at john.doe@example.com"
    chat << "I'm sorry, I can't respond to that."

    # Phone pattern should match
    chat >> "Call me at (555) 123-4567"
    chat << "I'm sorry, I can't respond to that."


@pytest.mark.unit
def test_regex_detection_invalid_pattern_raises_at_config_load():
    """Invalid regex patterns are caught at config load time via model_validator, not at runtime."""
    with pytest.raises(ValidationError) as excinfo:
        RailsConfig.from_content(
            yaml_content="""
                models: []
                rails:
                  config:
                    regex_detection:
                      input:
                        patterns:
                          - "[unclosed"
                          - "\\\\bvalid\\\\b"
                  input:
                    flows:
                      - regex check input
            """,
            colang_content="""
                define user express greeting
                  "hi"

                define flow
                  user express greeting
                  bot express greeting

                define bot inform answer unknown
                  "I can't answer that."
            """,
        )
    assert "Invalid regex pattern" in str(excinfo.value)
    assert "[unclosed" in str(excinfo.value)


@pytest.mark.unit
def test_regex_detection_word_boundary():
    """Test that word boundaries work correctly."""
    config = RailsConfig.from_content(
        yaml_content="""
            models: []
            rails:
              config:
                regex_detection:
                  input:
                    patterns:
                      - "\\\\bpass\\\\b"
              input:
                flows:
                  - regex check input
        """,
        colang_content="""
            define user express greeting
              "hi"

            define flow
              user express greeting
              bot express greeting

            define bot refuse to respond
              "I'm sorry, I can't respond to that."
        """,
    )

    chat = TestChat(
        config,
        llm_completions=["  express greeting", '  "Hello!"'],
    )

    # "password" should NOT match because we're looking for "pass" as a whole word
    chat >> "My password is secret"
    chat << "Hello!"


@pytest.mark.unit
def test_regex_detection_word_boundary_matches():
    """Test that word boundaries match correctly when the word is standalone."""
    config = RailsConfig.from_content(
        yaml_content="""
            models: []
            rails:
              config:
                regex_detection:
                  input:
                    patterns:
                      - "\\\\bpass\\\\b"
              input:
                flows:
                  - regex check input
        """,
        colang_content="""
            define user express greeting
              "hi"

            define flow
              user express greeting
              bot express greeting

            define bot refuse to respond
              "I'm sorry, I can't respond to that."
        """,
    )

    chat = TestChat(
        config,
        llm_completions=["  express greeting", '  "Hello!"'],
    )

    # "pass" as standalone word should match
    chat >> "Please pass the test"
    chat << "I'm sorry, I can't respond to that."


@pytest.mark.unit
def test_regex_detection_retrieval_clears_matching_chunks():
    """Test that retrieved chunks matching a regex pattern are cleared.

    Retrieval rails run inside bot message generation, so they cannot emit a new
    bot message.  Instead the matched chunks are cleared (set to empty string) so
    the LLM never sees the flagged content.  A second retrieval flow verifies the
    chunks were actually cleared.
    """
    config = RailsConfig.from_content(
        yaml_content="""
            models: []
            rails:
              config:
                regex_detection:
                  retrieval:
                    patterns:
                      - "\\\\bclassified\\\\b"
                    case_insensitive: true
              retrieval:
                flows:
                  - regex check retrieval
                  - check relevant chunks
        """,
        colang_content="""
            define user express greeting
              "hi"

            define flow
              user express greeting
              bot express greeting

            define flow check relevant chunks
              execute check_relevant_chunks(relevant_chunks=$relevant_chunks)
        """,
    )

    chat = TestChat(
        config,
        llm_completions=["  express greeting", '  "Here is what I found."'],
    )

    @action()
    def retrieve_relevant_chunks():
        context_updates = {"relevant_chunks": "This document is classified material."}
        return ActionResult(
            return_value=context_updates["relevant_chunks"],
            context_updates=context_updates,
        )

    @action()
    def check_relevant_chunks(relevant_chunks: str):
        # After the regex retrieval rail, matched chunks should be cleared.
        assert relevant_chunks == ""

    chat.app.register_action(retrieve_relevant_chunks)
    chat.app.register_action(check_relevant_chunks)

    # The retrieved chunk contains "classified" — the rail clears it, but the
    # bot still responds (without the dangerous KB context).
    chat >> "Hi!"
    chat << "Here is what I found."


@pytest.mark.unit
def test_regex_detection_retrieval_allows_non_matching():
    """Test that retrieved chunks not matching any pattern are left untouched."""
    config = RailsConfig.from_content(
        yaml_content="""
            models: []
            rails:
              config:
                regex_detection:
                  retrieval:
                    patterns:
                      - "\\\\bclassified\\\\b"
              retrieval:
                flows:
                  - regex check retrieval
                  - check relevant chunks
        """,
        colang_content="""
            define user express greeting
              "hi"

            define flow
              user express greeting
              bot express greeting

            define flow check relevant chunks
              execute check_relevant_chunks(relevant_chunks=$relevant_chunks)
        """,
    )

    chat = TestChat(
        config,
        llm_completions=["  express greeting", '  "Here is what I found."'],
    )

    @action()
    def retrieve_relevant_chunks():
        context_updates = {"relevant_chunks": "This document is public information."}
        return ActionResult(
            return_value=context_updates["relevant_chunks"],
            context_updates=context_updates,
        )

    @action()
    def check_relevant_chunks(relevant_chunks: str):
        # No match, so chunks should be passed through unchanged.
        assert relevant_chunks == "This document is public information."

    chat.app.register_action(retrieve_relevant_chunks)
    chat.app.register_action(check_relevant_chunks)

    # The retrieved chunk does NOT contain "classified" — passes through unchanged.
    chat >> "Hi!"
    chat << "Here is what I found."
