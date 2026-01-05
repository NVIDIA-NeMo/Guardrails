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

from typing import Any, Dict, List, Optional

import pytest

from nemoguardrails import RailsConfig
from nemoguardrails.actions.actions import ActionResult, action
from tests.utils import TestChat


def create_gliner_mock_response(
    text: str,
    entities_to_detect: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Create a mock GLiNER response based on the input text and entities to detect.

    This simulates the GLiNER server's behavior by detecting common PII patterns.
    """
    detected_entities = []

    # Define patterns to detect (simple substring matching for mock purposes)
    entity_patterns = {
        "first_name": ["John", "Jane"],
        "last_name": ["Doe", "Smith"],
        "email": ["@gmail.com", "@email.com", "@yahoo.com", "@hotmail.com"],
    }

    for entity_type, patterns in entity_patterns.items():
        # Skip if entities_to_detect is specified and this type is not in the list
        if entities_to_detect and entity_type not in entities_to_detect:
            continue

        for pattern in patterns:
            start = 0
            while True:
                pos = text.find(pattern, start)
                if pos == -1:
                    break

                # For emails, find the full email address
                if entity_type == "email":
                    # Find the start of the email (go back to find non-space character)
                    email_start = pos
                    while email_start > 0 and text[email_start - 1] not in " \n\t,;:":
                        email_start -= 1
                    # Find the end of the email
                    email_end = pos + len(pattern)
                    value = text[email_start:email_end]
                    detected_entities.append(
                        {
                            "value": value,
                            "suggested_label": entity_type,
                            "start_position": email_start,
                            "end_position": email_end,
                            "score": 0.95,
                        }
                    )
                else:
                    detected_entities.append(
                        {
                            "value": pattern,
                            "suggested_label": entity_type,
                            "start_position": pos,
                            "end_position": pos + len(pattern),
                            "score": 0.95,
                        }
                    )

                start = pos + 1

    return {
        "entities": detected_entities,
        "total_entities": len(detected_entities),
        "tagged_text": text,  # Simplified - not actually tagging for mock
    }


def _mask_text_with_entities(text: str, entities: List[dict]) -> str:
    """
    Mask detected entities in text with their labels.

    Args:
        text: Original text
        entities: List of entity dictionaries with 'value', 'suggested_label',
                 'start_position', 'end_position' keys

    Returns:
        Text with entities replaced by [LABEL] placeholders
    """
    if not entities:
        return text

    # Sort entities by start position in reverse order to replace from end to start
    sorted_entities = sorted(entities, key=lambda x: x["start_position"], reverse=True)

    masked_text = text
    for entity in sorted_entities:
        start = entity["start_position"]
        end = entity["end_position"]
        label = entity["suggested_label"].upper()
        masked_text = masked_text[:start] + f"[{label}]" + masked_text[end:]

    return masked_text


def create_mock_gliner_detect_pii(entities_to_detect: Optional[List[str]] = None):
    """Create a mock gliner_detect_pii action that returns True when PII is detected."""

    async def mock_gliner_detect_pii(source: str, text: str, config, **kwargs):
        response = create_gliner_mock_response(text, entities_to_detect)
        return response.get("total_entities", 0) > 0

    return mock_gliner_detect_pii


def create_mock_gliner_mask_pii(entities_to_detect: Optional[List[str]] = None):
    """Create a mock gliner_mask_pii action that masks PII in text."""

    async def mock_gliner_mask_pii(source: str, text: str, config, **kwargs):
        response = create_gliner_mock_response(text, entities_to_detect)
        entities = response.get("entities", [])
        return _mask_text_with_entities(text, entities)

    return mock_gliner_mask_pii


@action()
def retrieve_relevant_chunks():
    context_updates = {"relevant_chunks": "Mock retrieved context."}

    return ActionResult(
        return_value=context_updates["relevant_chunks"],
        context_updates=context_updates,
    )


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
    # Register mock GLiNER actions (not used but prevents errors if called)
    chat.app.register_action(create_mock_gliner_detect_pii(), "gliner_detect_pii")
    chat.app.register_action(create_mock_gliner_mask_pii(), "gliner_mask_pii")

    chat >> "Hi! I am Mr. John! And my email is test@gmail.com"
    chat << "Hi! My name is John as well."


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
    # Register mock GLiNER actions with the entities to detect
    chat.app.register_action(
        create_mock_gliner_detect_pii(["email", "first_name", "last_name"]),
        "gliner_detect_pii",
    )
    chat.app.register_action(
        create_mock_gliner_mask_pii(["email", "first_name", "last_name"]),
        "gliner_mask_pii",
    )

    chat >> "Hi! I am Mr. John! And my email is test@gmail.com"
    chat << "I can't answer that."


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
    # Register mock GLiNER actions with the entities to detect
    chat.app.register_action(
        create_mock_gliner_detect_pii(["email", "first_name", "last_name"]),
        "gliner_detect_pii",
    )
    chat.app.register_action(
        create_mock_gliner_mask_pii(["email", "first_name", "last_name"]),
        "gliner_mask_pii",
    )

    chat >> "Hi!"
    chat << "I can't answer that."


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
    # Register mock GLiNER actions with the entities to detect
    chat.app.register_action(
        create_mock_gliner_detect_pii(["email", "first_name", "last_name"]),
        "gliner_detect_pii",
    )
    chat.app.register_action(
        create_mock_gliner_mask_pii(["email", "first_name", "last_name"]),
        "gliner_mask_pii",
    )

    chat >> "Hi!"
    chat << "Hi! My name is John as well."


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
    # Register mock GLiNER actions with the entities to detect
    chat.app.register_action(
        create_mock_gliner_detect_pii(["email", "first_name"]),
        "gliner_detect_pii",
    )
    chat.app.register_action(
        create_mock_gliner_mask_pii(["email", "first_name"]),
        "gliner_mask_pii",
    )

    chat >> "Hi!"
    # The name should be masked - response should contain [FIRST_NAME] instead of John
    response = chat.app.generate(messages=[{"role": "user", "content": "Hi!"}])
    # Verify the name was masked
    assert "John" not in response["content"] or "[FIRST_NAME]" in response["content"]


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
    # Register mock GLiNER actions with the entities to detect
    chat.app.register_action(
        create_mock_gliner_detect_pii(["email", "first_name"]),
        "gliner_detect_pii",
    )
    chat.app.register_action(
        create_mock_gliner_mask_pii(["email", "first_name"]),
        "gliner_mask_pii",
    )

    chat >> "Hi there! Are you John?"


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
    # Register mock GLiNER actions with the entities to detect
    chat.app.register_action(
        create_mock_gliner_detect_pii(["email", "first_name"]),
        "gliner_detect_pii",
    )
    chat.app.register_action(
        create_mock_gliner_mask_pii(["email", "first_name"]),
        "gliner_mask_pii",
    )

    chat >> "Hey! Can you help me get John's email?"
