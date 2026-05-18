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

import json
import logging
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, patch

import pytest
from aioresponses import aioresponses

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
    chat << "I'm sorry, I can't respond to that."


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
    chat << "I'm sorry, I can't respond to that."


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


def _build_gliner_config_for_api_key_tests(api_key_env_var: Optional[str] = None) -> RailsConfig:
    """Minimal RailsConfig with an `input` source_config so the actions reach the api_key
    resolution path. Optionally configures `api_key_env_var`."""
    yaml = (
        "models: []\n"
        "rails:\n"
        "  config:\n"
        "    gliner:\n"
        "      server_endpoint: http://localhost:8000/v1/chat/completions\n"
        "      input:\n"
        "        entities:\n"
        "          - email\n"
    )
    if api_key_env_var is not None:
        yaml += f"      api_key_env_var: {api_key_env_var}\n"
    return RailsConfig.from_content(yaml_content=yaml)


@pytest.mark.unit
def test_gliner_config_rejects_unsupported_engine():
    with pytest.raises(ValueError, match="rails.config.gliner.engine"):
        RailsConfig.from_content(
            yaml_content="""
                models: []
                rails:
                  config:
                    gliner:
                      engine: nvcf
                      input:
                        entities:
                          - email
            """,
        )


@pytest.mark.unit
def test_gliner_config_rejects_unknown_nested_option():
    with pytest.raises(ValueError, match="rails.config.gliner.input.unknown_option"):
        RailsConfig.from_content(
            yaml_content="""
                models: []
                rails:
                  config:
                    gliner:
                      input:
                        entities:
                          - email
                        unknown_option: true
            """,
        )


@pytest.mark.unit
def test_resolve_api_key_env_var_not_configured(monkeypatch, caplog):
    """No api_key_env_var set => returns None, no warning logged."""
    from nemoguardrails.library.gliner.actions import _resolve_api_key

    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    gliner_config = _build_gliner_config_for_api_key_tests(api_key_env_var=None).rails.config.gliner

    with caplog.at_level(logging.WARNING, logger="nemoguardrails.library.gliner.actions"):
        result = _resolve_api_key(gliner_config)

    assert result is None
    assert "api_key_env_var" not in caplog.text


@pytest.mark.unit
def test_resolve_api_key_env_var_set_and_present(monkeypatch, caplog):
    """api_key_env_var configured AND env var set => returns the env value, no warning."""
    from nemoguardrails.library.gliner.actions import _resolve_api_key

    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test-token")
    gliner_config = _build_gliner_config_for_api_key_tests(api_key_env_var="NVIDIA_API_KEY").rails.config.gliner

    with caplog.at_level(logging.WARNING, logger="nemoguardrails.library.gliner.actions"):
        result = _resolve_api_key(gliner_config)

    assert result == "nvapi-test-token"
    assert "environment variable is not set" not in caplog.text


@pytest.mark.unit
def test_resolve_api_key_env_var_set_but_missing(monkeypatch, caplog):
    """api_key_env_var configured BUT env var unset => returns None, warning names the var."""
    from nemoguardrails.library.gliner.actions import _resolve_api_key

    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    gliner_config = _build_gliner_config_for_api_key_tests(api_key_env_var="NVIDIA_API_KEY").rails.config.gliner

    with caplog.at_level(logging.WARNING, logger="nemoguardrails.library.gliner.actions"):
        result = _resolve_api_key(gliner_config)

    assert result is None
    assert "NVIDIA_API_KEY" in caplog.text
    assert "environment variable is not set" in caplog.text


@pytest.mark.unit
@pytest.mark.asyncio
async def test_gliner_detect_pii_forwards_resolved_api_key():
    """gliner_detect_pii passes _resolve_api_key's return value into gliner_request's api_key kwarg."""
    config = _build_gliner_config_for_api_key_tests(api_key_env_var="NVIDIA_API_KEY")

    with (
        patch(
            "nemoguardrails.library.gliner.actions._resolve_api_key",
            return_value="sentinel-api-key",
        ),
        patch(
            "nemoguardrails.library.gliner.actions.gliner_request",
            new=AsyncMock(return_value={"total_entities": 0, "entities": []}),
        ) as mock_request,
    ):
        from nemoguardrails.library.gliner.actions import gliner_detect_pii

        await gliner_detect_pii(source="input", text="Hello.", config=config)

    assert mock_request.await_args.kwargs["api_key"] == "sentinel-api-key"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_gliner_mask_pii_forwards_resolved_api_key():
    """gliner_mask_pii passes _resolve_api_key's return value into gliner_request's api_key kwarg."""
    config = _build_gliner_config_for_api_key_tests(api_key_env_var="NVIDIA_API_KEY")

    with (
        patch(
            "nemoguardrails.library.gliner.actions._resolve_api_key",
            return_value="sentinel-api-key",
        ),
        patch(
            "nemoguardrails.library.gliner.actions.gliner_request",
            new=AsyncMock(return_value={"entities": []}),
        ) as mock_request,
    ):
        from nemoguardrails.library.gliner.actions import gliner_mask_pii

        await gliner_mask_pii(source="input", text="Hello.", config=config)

    assert mock_request.await_args.kwargs["api_key"] == "sentinel-api-key"


NIM_ENDPOINT = "http://localhost:8000/v1/chat/completions"
CUSTOM_ENDPOINT = "http://localhost:1235/v1/extract"


def _wrap_in_chat_completions(content) -> dict:
    """Wrap a JSON-serializable content payload in a NIM chat-completions envelope."""
    inner = content if isinstance(content, str) else json.dumps(content)
    return {"choices": [{"message": {"content": inner}}]}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_gliner_request_chat_completions_normalizes_entities():
    """NIM (chat completions) endpoint: unwraps the envelope and renames entity fields."""
    from nemoguardrails.library.gliner.request import gliner_request

    nim_content = {
        "entities": [
            {"text": "John", "label": "first_name", "start": 0, "end": 4, "score": 0.95},
            {"text": "test@example.com", "label": "email", "start": 17, "end": 33, "score": 0.98},
        ],
        "total_entities": 2,
        "tagged_text": "[John](first_name) ... [test@example.com](email)",
    }
    with aioresponses() as m:
        m.post(NIM_ENDPOINT, payload=_wrap_in_chat_completions(nim_content))
        result = await gliner_request(text="Hi I'm John", server_endpoint=NIM_ENDPOINT)

    assert result["total_entities"] == 2
    assert result["tagged_text"] == nim_content["tagged_text"]
    assert result["entities"][0] == {
        "value": "John",
        "suggested_label": "first_name",
        "start_position": 0,
        "end_position": 4,
        "score": 0.95,
    }
    assert result["entities"][1]["value"] == "test@example.com"
    assert result["entities"][1]["suggested_label"] == "email"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_gliner_request_custom_endpoint_returns_raw():
    """Custom server (/v1/extract): returns the response body unchanged (no normalization)."""
    from nemoguardrails.library.gliner.request import gliner_request

    custom_response = {
        "entities": [
            {
                "value": "John",
                "suggested_label": "first_name",
                "start_position": 0,
                "end_position": 4,
                "score": 0.91,
            },
        ],
        "total_entities": 1,
        "tagged_text": "[John](first_name)",
    }
    with aioresponses() as m:
        m.post(CUSTOM_ENDPOINT, payload=custom_response)
        result = await gliner_request(text="John", server_endpoint=CUSTOM_ENDPOINT)

    assert result == custom_response


@pytest.mark.unit
@pytest.mark.asyncio
async def test_gliner_request_forwards_api_key_as_bearer_header():
    """When api_key is set, an Authorization: Bearer <key> header is sent."""
    from nemoguardrails.library.gliner.request import gliner_request

    with aioresponses() as m:
        m.post(
            NIM_ENDPOINT,
            payload=_wrap_in_chat_completions({"entities": [], "total_entities": 0, "tagged_text": ""}),
        )
        await gliner_request(text="hi", server_endpoint=NIM_ENDPOINT, api_key="nvapi-test-key")

    sent = next(iter(m.requests.values()))[0]
    headers = sent.kwargs.get("headers") or {}
    assert headers.get("Authorization") == "Bearer nvapi-test-key"
    assert headers.get("Content-Type") == "application/json"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_gliner_request_no_api_key_omits_authorization_header():
    """When api_key is None, no Authorization header is sent."""
    from nemoguardrails.library.gliner.request import gliner_request

    with aioresponses() as m:
        m.post(
            NIM_ENDPOINT,
            payload=_wrap_in_chat_completions({"entities": [], "total_entities": 0, "tagged_text": ""}),
        )
        await gliner_request(text="hi", server_endpoint=NIM_ENDPOINT)

    sent = next(iter(m.requests.values()))[0]
    headers = sent.kwargs.get("headers") or {}
    assert "Authorization" not in headers


@pytest.mark.unit
@pytest.mark.asyncio
async def test_gliner_request_non_200_status_raises():
    """Non-200 responses raise ValueError with the status code in the message."""
    from nemoguardrails.library.gliner.request import gliner_request

    with aioresponses() as m:
        m.post(NIM_ENDPOINT, status=500, body="Internal Server Error")
        with pytest.raises(ValueError, match=r"GLiNER call failed with status code 500"):
            await gliner_request(text="hi", server_endpoint=NIM_ENDPOINT)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_gliner_request_non_json_response_raises():
    """If the server returns a 200 with non-JSON Content-Type, ValueError is raised."""
    from nemoguardrails.library.gliner.request import gliner_request

    with aioresponses() as m:
        m.post(NIM_ENDPOINT, status=200, body="not json", content_type="text/plain")
        with pytest.raises(ValueError, match=r"Failed to parse GLiNER response as JSON"):
            await gliner_request(text="hi", server_endpoint=NIM_ENDPOINT)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_gliner_request_nim_content_unparseable_raises():
    """Chat completions: when message.content is not valid JSON, ValueError is raised."""
    from nemoguardrails.library.gliner.request import gliner_request

    with aioresponses() as m:
        m.post(NIM_ENDPOINT, payload=_wrap_in_chat_completions("this is not json {"))
        with pytest.raises(ValueError, match=r"Failed to parse NIM response content"):
            await gliner_request(text="hi", server_endpoint=NIM_ENDPOINT)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_gliner_request_nim_content_not_dict_raises():
    """Chat completions: when message.content parses to a non-dict, ValueError is raised."""
    from nemoguardrails.library.gliner.request import gliner_request

    with aioresponses() as m:
        m.post(NIM_ENDPOINT, payload=_wrap_in_chat_completions(["entities", "as", "list"]))
        with pytest.raises(ValueError, match=r"Expected NIM response content to be a JSON object"):
            await gliner_request(text="hi", server_endpoint=NIM_ENDPOINT)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_gliner_request_forwards_all_optional_params_to_payload():
    """All optional params flow into the JSON payload sent to the server."""
    from nemoguardrails.library.gliner.request import gliner_request

    with aioresponses() as m:
        m.post(
            NIM_ENDPOINT,
            payload=_wrap_in_chat_completions({"entities": [], "total_entities": 0, "tagged_text": ""}),
        )
        await gliner_request(
            text="hello",
            server_endpoint=NIM_ENDPOINT,
            enabled_entities=["email", "first_name"],
            threshold=0.7,
            chunk_length=512,
            overlap=64,
            flat_ner=True,
        )

    sent = next(iter(m.requests.values()))[0]
    payload = sent.kwargs.get("json") or {}
    assert payload["threshold"] == 0.7
    assert payload["chunk_length"] == 512
    assert payload["overlap"] == 64
    assert payload["flat_ner"] is True
    assert payload["labels"] == ["email", "first_name"]
