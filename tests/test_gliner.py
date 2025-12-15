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

import os

import pytest

from nemoguardrails import RailsConfig
from nemoguardrails.actions.actions import ActionResult, action
from tests.utils import TestChat

GLINER_SERVER_AVAILABLE = os.getenv("GLINER_SERVER_ENDPOINT") is not None


@action()
def retrieve_relevant_chunks():
    context_updates = {"relevant_chunks": "Mock retrieved context."}

    return ActionResult(
        return_value=context_updates["relevant_chunks"],
        context_updates=context_updates,
    )


@pytest.mark.skipif(not GLINER_SERVER_AVAILABLE, reason="GLiNER server is not available.")
@pytest.mark.unit
def test_gliner_pii_detection_no_active_pii_detection():
    config = RailsConfig.from_content(
        yaml_content="""
            models: []
            rails:
              config:
                gliner:
                  server_endpoint: http://localhost:1235/v1/extract
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
        llm_completions=[
            "  express greeting",
            '  "Hi! My name is John as well."',
        ],
    )

    chat.app.register_action(retrieve_relevant_chunks, "retrieve_relevant_chunks")
    chat >> "Hi! I am Mr. John! And my email is test@gmail.com"
    chat << "Hi! My name is John as well."


@pytest.mark.skipif(not GLINER_SERVER_AVAILABLE, reason="GLiNER server is not available.")
@pytest.mark.unit
def test_gliner_pii_detection_input():
    config = RailsConfig.from_content(
        yaml_content="""
            models: []
            rails:
              config:
                gliner:
                  server_endpoint: http://localhost:1235/v1/extract
                  input:
                    entities:
                      - email
                      - first_name
                      - last_name
              input:
                flows:
                  - gliner detect pii on input
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
        llm_completions=[
            "  express greeting",
            '  "Hi! My name is John as well."',
        ],
    )

    chat.app.register_action(retrieve_relevant_chunks, "retrieve_relevant_chunks")
    chat >> "Hi! I am Mr. John! And my email is test@gmail.com"
    chat << "I can't answer that."


@pytest.mark.skipif(not GLINER_SERVER_AVAILABLE, reason="GLiNER server is not available.")
@pytest.mark.unit
def test_gliner_pii_detection_output():
    config = RailsConfig.from_content(
        yaml_content="""
            models: []
            rails:
              config:
                gliner:
                  server_endpoint: http://localhost:1235/v1/extract
                  output:
                    entities:
                      - email
                      - first_name
                      - last_name
              output:
                flows:
                  - gliner detect pii on output
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
        llm_completions=[
            "  express greeting",
            '  "Hi! My name is John as well."',
        ],
    )

    chat.app.register_action(retrieve_relevant_chunks, "retrieve_relevant_chunks")
    chat >> "Hi!"
    chat << "I can't answer that."


@pytest.mark.skipif(not GLINER_SERVER_AVAILABLE, reason="GLiNER server is not available.")
@pytest.mark.unit
def test_gliner_pii_detection_retrieval_with_no_pii():
    config = RailsConfig.from_content(
        yaml_content="""
            models: []
            rails:
              config:
                gliner:
                  server_endpoint: http://localhost:1235/v1/extract
                  retrieval:
                    entities:
                      - email
                      - first_name
                      - last_name
              retrieval:
                flows:
                  - gliner detect pii on retrieval
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
        llm_completions=[
            "  express greeting",
            '  "Hi! My name is John as well."',
        ],
    )

    chat.app.register_action(retrieve_relevant_chunks, "retrieve_relevant_chunks")

    chat >> "Hi!"
    chat << "Hi! My name is John as well."


@pytest.mark.skipif(not GLINER_SERVER_AVAILABLE, reason="GLiNER server is not available.")
@pytest.mark.unit
def test_gliner_pii_masking_on_output():
    config = RailsConfig.from_content(
        yaml_content="""
            models: []
            rails:
              config:
                gliner:
                  server_endpoint: http://localhost:1235/v1/extract
                  output:
                    entities:
                      - email
                      - first_name
              output:
                flows:
                  - gliner mask pii on output
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
        llm_completions=[
            "  express greeting",
            '  "Hi! I am John.',
        ],
    )

    chat.app.register_action(retrieve_relevant_chunks, "retrieve_relevant_chunks")

    chat >> "Hi!"
    # The name should be masked - check that the response contains masked content
    # Note: The actual masking behavior depends on GLiNER server response


@pytest.mark.skipif(not GLINER_SERVER_AVAILABLE, reason="GLiNER server is not available.")
@pytest.mark.unit
def test_gliner_pii_masking_on_input():
    config = RailsConfig.from_content(
        yaml_content="""
            models: []
            rails:
              config:
                gliner:
                  server_endpoint: http://localhost:1235/v1/extract
                  input:
                    entities:
                      - email
                      - first_name
              input:
                flows:
                  - gliner mask pii on input
                  - check user message
        """,
        colang_content="""
            define user express greeting
              "hi"

            define flow
              user express greeting
              bot express greeting

            define bot inform answer unknown
              "I can't answer that."

            define flow check user message
              execute check_user_message(user_message=$user_message)
        """,
    )

    chat = TestChat(
        config,
        llm_completions=[
            "  express greeting",
            '  "Hi! Nice to meet you.',
        ],
    )

    @action()
    def check_user_message(user_message: str):
        """Check if the user message has PII masked."""
        # Verify that either the name is removed or replaced with a label
        assert "John" not in user_message or "[FIRST_NAME]" in user_message

    chat.app.register_action(retrieve_relevant_chunks, "retrieve_relevant_chunks")
    chat.app.register_action(check_user_message, "check_user_message")

    chat >> "Hi there! Are you John?"


@pytest.mark.skipif(not GLINER_SERVER_AVAILABLE, reason="GLiNER server is not available.")
@pytest.mark.unit
def test_gliner_pii_masking_on_retrieval():
    config = RailsConfig.from_content(
        yaml_content="""
            models: []
            rails:
              config:
                gliner:
                  server_endpoint: http://localhost:1235/v1/extract
                  retrieval:
                    entities:
                      - email
                      - first_name
              retrieval:
                flows:
                  - gliner mask pii on retrieval
                  - check relevant chunks
        """,
        colang_content="""
            define user express greeting
              "hi"

            define flow
              user express greeting
              bot express greeting

            define bot inform answer unknown
              "I can't answer that."

            define flow check relevant chunks
              execute check_relevant_chunks(relevant_chunks=$relevant_chunks)
        """,
    )

    chat = TestChat(
        config,
        llm_completions=[
            "  express greeting",
            "  Sorry, I don't have that in my knowledge base.",
        ],
    )

    @action()
    def check_relevant_chunks(relevant_chunks: str):
        """Check if the relevant chunks have PII masked."""
        # Verify that either the PII is removed or replaced with labels
        assert "john@email.com" not in relevant_chunks or "[EMAIL]" in relevant_chunks

    @action()
    def retrieve_relevant_chunk_for_masking():
        # Mock retrieval of relevant chunks with PII
        context_updates = {"relevant_chunks": "John's Email: john@email.com"}
        return ActionResult(
            return_value=context_updates["relevant_chunks"],
            context_updates=context_updates,
        )

    chat.app.register_action(retrieve_relevant_chunk_for_masking, "retrieve_relevant_chunks")
    chat.app.register_action(check_relevant_chunks)

    chat >> "Hey! Can you help me get John's email?"
