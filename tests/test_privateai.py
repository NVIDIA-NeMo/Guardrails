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
import os

import pytest

from nemoguardrails import RailsConfig
from nemoguardrails.actions.actions import ActionResult, action
from nemoguardrails.http import HTTPResponse
from nemoguardrails.library.privateai.actions import detect_pii, mask_pii
from nemoguardrails.library.privateai.request import private_ai_request
from nemoguardrails.testing import RecordingHTTPClient
from tests.utils import TestChat

PAI_API_KEY_PRESENT = os.getenv("PAI_API_KEY") is not None


def _response(payload=None, *, status: int = 200, content: bytes | None = None) -> HTTPResponse:
    return HTTPResponse(
        status_code=status,
        headers={"content-type": "application/json"},
        content=content if content is not None else json.dumps(payload).encode(),
    )


def _privateai_config() -> RailsConfig:
    return RailsConfig.from_content(
        yaml_content="""
            models: []
            rails:
              config:
                privateai:
                  server_endpoint: http://privateai.example/process
                  input:
                    entities:
                      - NAME
                  output:
                    entities:
                      - NAME
        """,
    )


@pytest.mark.asyncio
async def test_private_ai_request_uses_shared_client():
    response_payload = [{"processed_text": "Hello [NAME_1].", "entities_present": ["NAME"]}]
    client = RecordingHTTPClient([_response(response_payload)])

    result = await private_ai_request(
        "Hello John.",
        ["NAME"],
        "https://api.private-ai.com/cloud/v3/process/text",
        api_key="secret",
        http_client=client,
    )

    assert result == response_payload
    request = client.requests[0]
    assert request.method == "POST"
    assert request.url == "https://api.private-ai.com/cloud/v3/process/text"
    assert request.headers == {
        "Content-Type": "application/json",
        "x-api-key": "secret",
    }
    assert request.json == {
        "text": ["Hello John."],
        "link_batch": False,
        "entity_detection": {
            "accuracy": "high_automatic",
            "return_entity": False,
            "entity_types": [{"type": "ENABLE", "value": ["NAME"]}],
        },
    }


@pytest.mark.asyncio
async def test_private_ai_request_omits_optional_cloud_fields():
    client = RecordingHTTPClient([_response([])])

    await private_ai_request(
        "Hello.",
        [],
        "http://privateai.example/process",
        http_client=client,
    )

    request = client.requests[0]
    assert request.headers == {"Content-Type": "application/json"}
    assert "entity_types" not in request.json["entity_detection"]


@pytest.mark.asyncio
async def test_private_ai_request_requires_cloud_api_key():
    with pytest.raises(ValueError, match="'api_key' is required"):
        await private_ai_request(
            "Hello.",
            [],
            "https://api.private-ai.com/cloud/v3/process/text",
            http_client=RecordingHTTPClient(),
        )


@pytest.mark.asyncio
async def test_private_ai_request_raises_for_non_200():
    client = RecordingHTTPClient([_response({}, status=503)])

    with pytest.raises(ValueError, match="Private AI call failed with status code 503"):
        await private_ai_request(
            "Hello.",
            [],
            "http://privateai.example/process",
            http_client=client,
        )


@pytest.mark.asyncio
async def test_private_ai_request_raises_for_invalid_json():
    client = RecordingHTTPClient([_response(content=b"not-json")])

    with pytest.raises(ValueError, match="Failed to parse Private AI response as JSON"):
        await private_ai_request(
            "Hello.",
            [],
            "http://privateai.example/process",
            http_client=client,
        )


@pytest.mark.asyncio
async def test_privateai_actions_forward_shared_client():
    client = RecordingHTTPClient(
        [
            _response([{"entities_present": ["NAME"]}]),
            _response([{"processed_text": "Hello [NAME_1]."}]),
        ]
    )
    config = _privateai_config()

    detection = await detect_pii("input", "Hello John.", config, http_client=client)
    masking = await mask_pii("output", "Hello John.", config, http_client=client)

    assert detection.is_blocked
    assert masking.transforms[0].text == "Hello [NAME_1]."
    assert len(client.requests) == 2


@action()
def retrieve_relevant_chunks():
    context_updates = {"relevant_chunks": "Mock retrieved context."}

    return ActionResult(
        return_value=context_updates["relevant_chunks"],
        context_updates=context_updates,
    )


@pytest.mark.skipif(not PAI_API_KEY_PRESENT, reason="Private AI API key is not present.")
@pytest.mark.unit
def test_privateai_pii_detection_no_active_pii_detection():
    config = RailsConfig.from_content(
        yaml_content="""
            models: []
            rails:
              config:
                privateai:
                  server_endpoint: https://api.private-ai.com/cloud/v3/process/text
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


@pytest.mark.skipif(not PAI_API_KEY_PRESENT, reason="Private AI API key is not present.")
@pytest.mark.unit
def test_privateai_pii_detection_input():
    config = RailsConfig.from_content(
        yaml_content="""
            models: []
            rails:
              config:
                privateai:
                  server_endpoint: https://api.private-ai.com/cloud/v3/process/text
                  input:
                    entities:
                      - EMAIL_ADDRESS
                      - NAME
              input:
                flows:
                  - detect pii on input
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


@pytest.mark.skipif(not PAI_API_KEY_PRESENT, reason="Private AI API key is not present.")
@pytest.mark.unit
def test_privateai_pii_detection_output():
    config = RailsConfig.from_content(
        yaml_content="""
            models: []
            rails:
              config:
                privateai:
                  server_endpoint: https://api.private-ai.com/cloud/v3/process/text
                  output:
                    entities:
                      - EMAIL_ADDRESS
                      - NAME
              output:
                flows:
                  - detect pii on output
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


@pytest.mark.skip(reason="This test needs refinement.")
@pytest.mark.unit
def test_privateai_pii_detection_retrieval_with_pii():
    # TODO: @pouyanpi and @letmerecall: Find an alternative approach to test this functionality.
    config = RailsConfig.from_content(
        yaml_content="""
            models: []
            rails:
              config:
                privateai:
                  server_endpoint: https://api.private-ai.com/cloud/v3/process/text
                  retrieval:
                    entities:
                      - EMAIL_ADDRESS
                      - NAME
              retrieval:
                flows:
                  - detect pii on retrieval
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

    # When the relevant_chunks has_pii, a bot intent will get invoked via (bot inform answer unknown), which in turn
    # will invoke retrieve_relevant_chunks action.
    # With a mocked retrieve_relevant_chunks always returning something & detect_pii always returning True,
    # the process goes in an infinite loop and raises an Exception: Too many events.
    with pytest.raises(Exception, match="Too many events."):
        chat >> "Hi!"
        chat << "I can't answer that."


@pytest.mark.skipif(not PAI_API_KEY_PRESENT, reason="Private AI API key is not present.")
@pytest.mark.unit
def test_privateai_pii_detection_retrieval_with_no_pii():
    config = RailsConfig.from_content(
        yaml_content="""
            models: []
            rails:
              config:
                privateai:
                  server_endpoint: https://api.private-ai.com/cloud/v3/process/text
                  retrieval:
                    entities:
                      - EMAIL_ADDRESS
                      - NAME
              retrieval:
                flows:
                  - detect pii on retrieval
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


@pytest.mark.skipif(not PAI_API_KEY_PRESENT, reason="Private AI API key is not present.")
@pytest.mark.unit
def test_privateai_pii_masking_on_output():
    config = RailsConfig.from_content(
        yaml_content="""
            models: []
            rails:
              config:
                privateai:
                  server_endpoint: https://api.private-ai.com/cloud/v3/process/text
                  output:
                    entities:
                      - EMAIL_ADDRESS
                      - NAME
              output:
                flows:
                  - mask pii on output
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
    chat << "Hi! I am [NAME_1]."


@pytest.mark.skipif(not PAI_API_KEY_PRESENT, reason="Private AI API key is not present.")
@pytest.mark.unit
def test_privateai_pii_masking_on_input():
    config = RailsConfig.from_content(
        yaml_content="""
            models: []
            rails:
              config:
                privateai:
                  server_endpoint: https://api.private-ai.com/cloud/v3/process/text
                  input:
                    entities:
                      - EMAIL_ADDRESS
                      - NAME
              input:
                flows:
                  - mask pii on input
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
            '  "Hi! I am John.',
        ],
    )

    @action()
    def check_user_message(user_message: str):
        """Check if the user message is converted to the expected message with masked PII."""
        assert user_message == "Hi there! Are you [NAME_1]?"

    chat.app.register_action(retrieve_relevant_chunks, "retrieve_relevant_chunks")
    chat.app.register_action(check_user_message, "check_user_message")

    chat >> "Hi there! Are you John?"
    chat << "Hi! I am John."


@pytest.mark.skipif(not PAI_API_KEY_PRESENT, reason="Private AI API key is not present.")
@pytest.mark.unit
def test_privateai_pii_masking_on_retrieval():
    config = RailsConfig.from_content(
        yaml_content="""
            models: []
            rails:
              config:
                privateai:
                  server_endpoint: https://api.private-ai.com/cloud/v3/process/text
                  retrieval:
                    entities:
                      - EMAIL_ADDRESS
                      - NAME
              retrieval:
                flows:
                  - mask pii on retrieval
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
        """Check if the relevant chunks is converted to the expected message with masked PII."""
        assert relevant_chunks == "[NAME_1]'s Email: [EMAIL_ADDRESS_1]"

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
    chat << "Sorry, I don't have that in my knowledge base."
