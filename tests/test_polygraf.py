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
from nemoguardrails.library.polygraf.actions import polygraf_detect_pii, polygraf_mask_pii
from nemoguardrails.library.polygraf.request import polygraf_request
from tests.utils import TestChat


def create_polygraf_mock_response(
    text: str,
    entities_to_detect: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Create a mock Polygraf response based on the input text and entities to detect."""
    detected_entities = []

    entity_patterns = {
        "Person": ["John"],
        "Email": ["test@gmail.com"],
    }

    for entity_type, patterns in entity_patterns.items():
        if entities_to_detect and entity_type not in entities_to_detect:
            continue

        for pattern in patterns:
            start = 0
            while True:
                pos = text.find(pattern, start)
                if pos == -1:
                    break
                detected_entities.append(
                    {
                        "entity_type": entity_type,
                        "entity_text": pattern,
                        "start": pos,
                        "end": pos + len(pattern),
                        "score": 0.99,
                    }
                )
                start = pos + 1

    return detected_entities


def create_mock_polygraf_detect_pii(entities_to_detect: Optional[List[str]] = None):
    """Create a mock polygraf_detect_pii action that returns True when PII is detected."""

    async def mock_polygraf_detect_pii(source: str, text: str, config, **kwargs):
        entities = create_polygraf_mock_response(text, entities_to_detect)
        return len(entities) > 0

    return mock_polygraf_detect_pii


def create_mock_polygraf_mask_pii(entities_to_detect: Optional[List[str]] = None):
    """Create a mock polygraf_mask_pii action that masks PII in text."""

    async def mock_polygraf_mask_pii(source: str, text: str, config, **kwargs):
        entities = create_polygraf_mock_response(text, entities_to_detect)
        if not entities:
            return text

        masked_text = text
        for entity in sorted(entities, key=lambda x: x["start"], reverse=True):
            start = entity["start"]
            end = entity["end"]
            entity_type = entity["entity_type"]
            masked_text = masked_text[:start] + f"<{entity_type}>" + masked_text[end:]

        return masked_text

    return mock_polygraf_mask_pii


@action()
def retrieve_relevant_chunks():
    context_updates = {"relevant_chunks": "Mock retrieved context."}

    return ActionResult(
        return_value=context_updates["relevant_chunks"],
        context_updates=context_updates,
    )


def _polygraf_config():
    return RailsConfig.from_content(
        yaml_content="""
            models: []
            rails:
              config:
                polygraf:
                  server_endpoint: http://localhost:8000/v1/pii/text-detect
                  input:
                    entities:
                      - Email
                      - Person
                  output:
                    entities:
                      - Email
                      - Person
                  retrieval:
                    entities:
                      - Email
                      - Person
        """,
    )


class _FakeResponse:
    def __init__(self, status=200, payload=None, text=""):
        self.status = status
        self._payload = payload if payload is not None else []
        self._text = text

    async def json(self):
        return self._payload

    async def text(self):
        return self._text


class _FakePostContextManager:
    def __init__(self, response):
        self.response = response

    async def __aenter__(self):
        return self.response

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakeSession:
    def __init__(self, response):
        self.response = response
        self.requests = []

    def post(self, server_endpoint, json, headers):
        self.requests.append(
            {
                "server_endpoint": server_endpoint,
                "json": json,
                "headers": headers,
            }
        )
        return _FakePostContextManager(self.response)


@pytest.mark.asyncio
async def test_polygraf_request_uses_shared_session_and_bearer_auth():
    session = _FakeSession(
        _FakeResponse(
            payload=[
                {
                    "entity_type": "Person",
                    "entity_text": "John",
                    "start": 0,
                    "end": 4,
                    "score": 0.99,
                }
            ]
        )
    )

    entities = await polygraf_request("John", "http://polygraf.example/pii", "secret", session=session)

    assert entities[0]["entity_type"] == "Person"
    assert session.requests[0]["headers"]["Authorization"] == "Bearer secret"
    assert session.requests[0]["json"]["detect_pid"] is True
    assert session.requests[0]["json"]["aggregate_entities"] is True


@pytest.mark.asyncio
async def test_polygraf_request_accepts_wrapped_entities_response():
    session = _FakeSession(_FakeResponse(payload={"entities": [{"entity_type": "Email"}]}))

    entities = await polygraf_request("test@gmail.com", "http://polygraf.example/pii", None, session=session)

    assert entities == [{"entity_type": "Email"}]


@pytest.mark.asyncio
async def test_polygraf_request_raises_for_invalid_response_shape():
    session = _FakeSession(_FakeResponse(payload={"unexpected": []}))

    with pytest.raises(ValueError, match="Invalid response from Polygraf service"):
        await polygraf_request("John", "http://polygraf.example/pii", None, session=session)


@pytest.mark.asyncio
async def test_polygraf_request_raises_for_non_200_response():
    session = _FakeSession(_FakeResponse(status=401, text="missing token"))

    with pytest.raises(ValueError, match="Polygraf call failed with status code 401"):
        await polygraf_request("John", "http://polygraf.example/pii", None, session=session)


@pytest.mark.asyncio
async def test_polygraf_actions_warn_when_api_key_missing(monkeypatch, caplog):
    async def mock_request(text, server_endpoint, api_key, session=None):
        assert api_key is None
        return []

    monkeypatch.delenv("POLYGRAF_API_KEY", raising=False)
    monkeypatch.setattr("nemoguardrails.library.polygraf.actions.polygraf_request", mock_request)
    caplog.set_level("WARNING")

    result = await polygraf_detect_pii("input", "John", _polygraf_config())

    assert result is False
    assert "POLYGRAF_API_KEY environment variable is not set" in caplog.text


@pytest.mark.asyncio
async def test_polygraf_mask_pii_accepts_extra_kwargs_and_shared_session(monkeypatch):
    sentinel_session = object()

    async def mock_request(text, server_endpoint, api_key, session=None):
        assert api_key == "secret"
        assert session is sentinel_session
        return [{"entity_type": "Person", "entity_text": "John", "start": 0, "end": 4, "score": 0.99}]

    monkeypatch.setenv("POLYGRAF_API_KEY", "secret")
    monkeypatch.setattr("nemoguardrails.library.polygraf.actions.polygraf_request", mock_request)

    result = await polygraf_mask_pii("input", "John", _polygraf_config(), session=sentinel_session, extra="ignored")

    assert result == "<Person>"


@pytest.mark.unit
def test_polygraf_pii_detection_no_active_pii_detection():
    config = RailsConfig.from_content(
        yaml_content="""
            models: []
            rails:
              config:
                polygraf:
                  server_endpoint: http://localhost:8000/v1/pii/text-detect
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
    chat.app.register_action(create_mock_polygraf_detect_pii(), "polygraf_detect_pii")
    chat.app.register_action(create_mock_polygraf_mask_pii(), "polygraf_mask_pii")

    chat >> "Hi! I am Mr. John! And my email is test@gmail.com"
    chat << "Hi! My name is John as well."


@pytest.mark.unit
def test_polygraf_pii_detection_input():
    config = RailsConfig.from_content(
        yaml_content="""
            models: []
            rails:
              config:
                polygraf:
                  server_endpoint: http://localhost:8000/v1/pii/text-detect
                  input:
                    entities:
                      - Email
                      - Person
              input:
                flows:
                  - polygraf detect pii on input
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
    chat.app.register_action(
        create_mock_polygraf_detect_pii(["Email", "Person"]),
        "polygraf_detect_pii",
    )
    chat.app.register_action(
        create_mock_polygraf_mask_pii(["Email", "Person"]),
        "polygraf_mask_pii",
    )

    chat >> "Hi! I am Mr. John! And my email is test@gmail.com"
    chat << "I can't answer that."


@pytest.mark.unit
def test_polygraf_pii_detection_output():
    config = RailsConfig.from_content(
        yaml_content="""
            models: []
            rails:
              config:
                polygraf:
                  server_endpoint: http://localhost:8000/v1/pii/text-detect
                  output:
                    entities:
                      - Email
                      - Person
              output:
                flows:
                  - polygraf detect pii on output
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
    chat.app.register_action(
        create_mock_polygraf_detect_pii(["Email", "Person"]),
        "polygraf_detect_pii",
    )
    chat.app.register_action(
        create_mock_polygraf_mask_pii(["Email", "Person"]),
        "polygraf_mask_pii",
    )

    chat >> "Hi!"
    chat << "I can't answer that."


@pytest.mark.unit
def test_polygraf_pii_detection_retrieval_with_no_pii():
    config = RailsConfig.from_content(
        yaml_content="""
            models: []
            rails:
              config:
                polygraf:
                  server_endpoint: http://localhost:8000/v1/pii/text-detect
                  retrieval:
                    entities:
                      - Email
                      - Person
              retrieval:
                flows:
                  - polygraf detect pii on retrieval
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
    chat.app.register_action(
        create_mock_polygraf_detect_pii(["Email", "Person"]),
        "polygraf_detect_pii",
    )
    chat.app.register_action(
        create_mock_polygraf_mask_pii(["Email", "Person"]),
        "polygraf_mask_pii",
    )

    chat >> "Hi!"
    chat << "Hi! My name is John as well."


@pytest.mark.unit
def test_polygraf_pii_masking_on_output():
    config = RailsConfig.from_content(
        yaml_content="""
            models: []
            rails:
              config:
                polygraf:
                  server_endpoint: http://localhost:8000/v1/pii/text-detect
                  output:
                    entities:
                      - Email
                      - Person
              output:
                flows:
                  - polygraf mask pii on output
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
    chat.app.register_action(
        create_mock_polygraf_detect_pii(["Email", "Person"]),
        "polygraf_detect_pii",
    )
    chat.app.register_action(
        create_mock_polygraf_mask_pii(["Email", "Person"]),
        "polygraf_mask_pii",
    )

    chat >> "Hi!"
    response = chat.app.generate(messages=[{"role": "user", "content": "Hi!"}])
    assert "John" not in response["content"] or "<NAME>" in response["content"]


@pytest.mark.unit
def test_polygraf_pii_masking_on_input():
    config = RailsConfig.from_content(
        yaml_content="""
            models: []
            rails:
              config:
                polygraf:
                  server_endpoint: http://localhost:8000/v1/pii/text-detect
                  input:
                    entities:
                      - Email
                      - Person
              input:
                flows:
                  - polygraf mask pii on input
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
        assert "John" not in user_message or "<NAME>" in user_message

    chat.app.register_action(retrieve_relevant_chunks, "retrieve_relevant_chunks")
    chat.app.register_action(check_user_message, "check_user_message")
    chat.app.register_action(
        create_mock_polygraf_detect_pii(["Email", "Person"]),
        "polygraf_detect_pii",
    )
    chat.app.register_action(
        create_mock_polygraf_mask_pii(["Email", "Person"]),
        "polygraf_mask_pii",
    )

    chat >> "Hi there! Are you John?"


@pytest.mark.unit
def test_polygraf_pii_masking_on_retrieval():
    config = RailsConfig.from_content(
        yaml_content="""
            models: []
            rails:
              config:
                polygraf:
                  server_endpoint: http://localhost:8000/v1/pii/text-detect
                  retrieval:
                    entities:
                      - Email
                      - Person
              retrieval:
                flows:
                  - polygraf mask pii on retrieval
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
        assert "test@gmail.com" not in relevant_chunks or "<Email>" in relevant_chunks

    @action()
    def retrieve_relevant_chunk_for_masking():
        context_updates = {"relevant_chunks": "John's Email: test@gmail.com"}
        return ActionResult(
            return_value=context_updates["relevant_chunks"],
            context_updates=context_updates,
        )

    chat.app.register_action(retrieve_relevant_chunk_for_masking, "retrieve_relevant_chunks")
    chat.app.register_action(check_relevant_chunks)
    chat.app.register_action(
        create_mock_polygraf_detect_pii(["Email", "Person"]),
        "polygraf_detect_pii",
    )
    chat.app.register_action(
        create_mock_polygraf_mask_pii(["Email", "Person"]),
        "polygraf_mask_pii",
    )

    chat >> "Hey! Can you help me get John's email?"
