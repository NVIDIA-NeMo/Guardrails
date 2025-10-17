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

# We detect if the environment is set up correct for SDD (presidio + downloaded spacy model)
# poetry install -e "sdd" --with dev
# python -m spacy download en_core_web_lg

import subprocess

import pytest

from nemoguardrails import RailsConfig
from nemoguardrails.actions import action
from nemoguardrails.actions.actions import ActionResult
from tests.utils import TestChat

try:
    import presidio_analyzer
    import presidio_anonymizer
    import spacy

    SDD_SETUP_PRESENT = True
except ImportError:
    SDD_SETUP_PRESENT = False


def setup_module(module):
    if not SDD_SETUP_PRESENT:
        pytest.skip("Required dependencies not found")

    try:
        # check if the model is already downloaded
        if not spacy.util.is_package("en_core_web_lg"):
            subprocess.run(
                ["python", "-m", "spacy", "download", "en_core_web_lg"],
                check=True,
                capture_output=True,
                text=True,
            )

        # ensure the model is now available

        if not spacy.util.is_package("en_core_web_lg"):
            pytest.skip("Failed to download or verify spaCy model 'en_core_web_lg'")

    except subprocess.CalledProcessError as e:
        pytest.skip(f"Error downloading spaCy model: {e.stderr}")
    except Exception as e:
        pytest.skip(f"Unexpected error during setup: {str(e)}")


def teardown_module(module):
    """No cleanup needed as the spaCy model is a persistent dependency
    that should remain available for future test runs."""
    pass


@pytest.mark.skipif(
    not SDD_SETUP_PRESENT, reason="Sensitive Data Detection setup is not present."
)
@pytest.mark.unit
def test_masking_input_output():
    config = RailsConfig.from_content(
        yaml_content="""
            models: []
            rails:
              config:
                sensitive_data_detection:
                  recognizers:
                    - name: "Titles recognizer"
                      supported_language: "en"
                      supported_entity: "TITLE"
                      deny_list:
                        - Mr.
                        - Mrs.
                        - Ms.
                        - Miss
                        - Dr.
                        - Prof.
                  input:
                    entities:
                      - PERSON
                      - TITLE
                  output:
                    entities:
                      - PERSON
              input:
                flows:
                  - mask sensitive data on input
                  - check user message
              output:
                flows:
                  - mask sensitive data on output
        """,
        colang_content="""
            define flow check user message
              execute check_user_message(user_message=$user_message)
        """,
    )

    chat = TestChat(config, llm_completions=["Hello there! My name is Michael!"])

    @action()
    def check_user_message(user_message):
        assert user_message == "Hi! I am <TITLE> <PERSON>!"

    chat.app.register_action(check_user_message)

    chat >> "Hi! I am Mr. John!"
    chat << "Hello there! My name is <PERSON>!"


@pytest.mark.skipif(
    not SDD_SETUP_PRESENT, reason="Sensitive Data Detection setup is not present."
)
@pytest.mark.unit
def test_detection_input_output():
    config = RailsConfig.from_content(
        yaml_content="""
            models: []
            rails:
              config:
                sensitive_data_detection:
                  input:
                    entities:
                      - PERSON
                  output:
                    entities:
                      - PERSON
              input:
                flows:
                  - detect sensitive data on input
              output:
                flows:
                  - detect sensitive data on output
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

    chat = TestChat(
        config,
        llm_completions=["  express greeting", '  "Hi! My name is John as well."'],
    )

    # This will trigger the input rail
    chat >> "Hi! I am Mr. John!"
    chat << "I can't answer that."

    # This will trigger only the output one
    chat >> "Hi!"
    chat << "I can't answer that."


@pytest.mark.skipif(
    not SDD_SETUP_PRESENT, reason="Sensitive Data Detection setup is not present."
)
@pytest.mark.unit
def test_masking_retrieval():
    config = RailsConfig.from_content(
        yaml_content="""
            models: []
            rails:
              config:
                sensitive_data_detection:
                  retrieval:
                    entities:
                      - PERSON
                      - TITLE
              retrieval:
                flows:
                  - mask sensitive data on retrieval
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
        llm_completions=["  express greeting", '  "Hello there!"'],
    )

    @action()
    def check_relevant_chunks(relevant_chunks: str):
        # This is where we check the masking has worked correctly.
        assert relevant_chunks == "The name of the user is <PERSON>."

    @action()
    def retrieve_relevant_chunks():
        context_updates = {"relevant_chunks": "The name of the user is John."}

        return ActionResult(
            return_value=context_updates["relevant_chunks"],
            context_updates=context_updates,
        )

    chat.app.register_action(check_relevant_chunks)
    chat.app.register_action(retrieve_relevant_chunks)

    chat >> "Hi!"
    chat << "Hello there!"


@pytest.mark.skipif(
    not SDD_SETUP_PRESENT, reason="Sensitive Data Detection setup is not present."
)
@pytest.mark.unit
def test_score_threshold():
    config = RailsConfig.from_content(
        yaml_content="""
            models: []
            rails:
              config:
                sensitive_data_detection:

                  input:
                    score_threshold: 0.4
                    entities:
                      - PERSON
                  output:

                    score_threshold: 0.4
                    entities:
                      - PERSON
              input:
                flows:
                  - detect sensitive data on input
              output:
                flows:
                  - detect sensitive data on output
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

    chat = TestChat(
        config,
        llm_completions=["  express greeting", '  "Hi! My name is John as well."'],
    )

    # This will trigger the input rail
    chat >> "Hi! I am Mr. John!"
    chat << "I can't answer that."

    # This will trigger only the output one
    chat >> "Hi!"
    chat << "I can't answer that."


@pytest.mark.skipif(
    not SDD_SETUP_PRESENT, reason="Sensitive Data Detection setup is not present."
)
@pytest.mark.unit
def test_invalid_score_threshold(caplog):
    config = RailsConfig.from_content(
        yaml_content="""
            models: []
            rails:
              config:
                sensitive_data_detection:

                  input:
                    score_threshold: -0.4
                    entities:
                      - PERSON
                  output:

                    score_threshold: -0.4
                    entities:
                      - PERSON
              input:
                flows:
                  - detect sensitive data on input
              output:
                flows:
                  - detect sensitive data on output
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

    chat = TestChat(
        config,
        llm_completions=["  express greeting", '  "Hi! My name is John as well."'],
    )

    rails = chat.app

    messages = [
        {"role": "user", "content": "Hi! I am Mr. John!"},
    ]

    _ = rails.generate(messages=messages)

    assert "score_threshold must be a float between 0 and 1 (inclusive)." in caplog.text


@pytest.mark.skipif(
    not SDD_SETUP_PRESENT, reason="Sensitive Data Detection setup is not present."
)
@pytest.mark.unit
def test_invalid_score_threshold_chat_message():
    config = RailsConfig.from_content(
        yaml_content="""
            models: []
            rails:
              config:
                sensitive_data_detection:

                  input:
                    score_threshold: 50
                    entities:
                      - PERSON
                  output:

                    score_threshold: 200
                    entities:
                      - PERSON
              input:
                flows:
                  - detect sensitive data on input
              output:
                flows:
                  - detect sensitive data on output
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

    chat = TestChat(
        config,
        llm_completions=["  express greeting", '  "Hi! My name is John as well."'],
    )

    # This will trigger the input rail
    chat >> "Hi! I am Mr. John!"
    chat << "I'm sorry, an internal error has occurred."


@pytest.mark.skipif(
    not SDD_SETUP_PRESENT, reason="Sensitive Data Detection setup is not present."
)
@pytest.mark.unit
def test_high_score_threshold_disables_rails():
    config = RailsConfig.from_content(
        yaml_content="""
            models: []
            rails:
              config:
                sensitive_data_detection:

                  input:
                    score_threshold: 1.0
                    entities:
                      - PERSON
              input:
                flows:
                  - detect sensitive data on input
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

    chat = TestChat(
        config,
        llm_completions=["  express greeting", '  "Hi! My name is John as well."'],
    )

    # This will trigger the input rail
    chat >> "Hi! I am Mr. John!"
    chat << "Hi! My name is John as well."
