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

"""Verify that every LLM-relevant field in AnthropicMessagesRequest reaches the
Anthropic model client's outgoing HTTP payload.

The test mocks the AnthropicClient at the HTTP layer and inspects the payload
dict that would be sent to the upstream provider. This catches cases where the
messages endpoint or model adapter silently drops a parameter.
"""

from unittest.mock import AsyncMock, patch

import pytest

pytest.importorskip("anthropic")

from nemoguardrails.llm.clients.anthropic import AnthropicClient  # noqa: E402
from nemoguardrails.llm.clients.base import HTTPResponse  # noqa: E402
from nemoguardrails.llm.models.anthropic_chat import AnthropicChatModel  # noqa: E402
from nemoguardrails.server.schemas.anthropic import AnthropicMessagesRequest  # noqa: E402
from nemoguardrails.types import ChatMessage, Role  # noqa: E402

FIELDS_NOT_FORWARDED_TO_LLM = {
    "messages",
    "model",
    "system",
    "stream",
}

FIELD_RENAMES = {
    "stop_sequences": "stop_sequences",
}

ANTHROPIC_RESPONSE = HTTPResponse(
    body={
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Hello"}],
        "model": "test-model",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 5},
    },
    headers={},
    status_code=200,
)

REQUEST_WITH_ALL_FIELDS = {
    "model": "test-model",
    "max_tokens": 256,
    "messages": [{"role": "user", "content": "Hi"}],
    "system": "You are helpful.",
    "stream": False,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 40,
    "stop_sequences": ["STOP"],
    "metadata": {"user_id": "test"},
    "tools": [
        {
            "name": "get_weather",
            "description": "Get weather",
            "input_schema": {"type": "object", "properties": {}},
        }
    ],
    "tool_choice": {"type": "auto"},
    "thinking": {"type": "adaptive"},
    "cache_control": {"type": "ephemeral"},
    "container": "container-123",
    "inference_geo": "us",
    "output_config": {"format": {"type": "json"}},
    "service_tier": "auto",
    "user_profile_id": "user-abc-123",
}


def _forwardable_fields() -> set[str]:
    """Return AnthropicMessagesRequest fields that should be forwarded to the LLM."""
    return set(AnthropicMessagesRequest.model_fields.keys()) - FIELDS_NOT_FORWARDED_TO_LLM


@pytest.mark.asyncio
async def test_all_params_reach_client_payload():
    """Every forwardable request field must appear in the client's outgoing payload."""
    captured_payload = {}

    client = AnthropicClient(base_url="http://test.local/v1", api_key="fake")
    model = AnthropicChatModel(client=client, model="test-model")

    original_build = client._build_payload

    def capturing_build(*args, **kwargs):
        result = original_build(*args, **kwargs)
        captured_payload.update(result)
        return result

    with (
        patch.object(client, "_build_payload", side_effect=capturing_build),
        patch.object(client, "_apost", new_callable=AsyncMock, return_value=ANTHROPIC_RESPONSE),
    ):
        body = AnthropicMessagesRequest(**REQUEST_WITH_ALL_FIELDS)

        params = {}
        for field in _forwardable_fields():
            value = getattr(body, field)
            if value is not None:
                wire_name = FIELD_RENAMES.get(field, field)
                params[wire_name] = value

        await model.generate_async(
            [ChatMessage(role=Role.USER, content="Hi")],
            **params,
        )

    forwardable = _forwardable_fields()
    missing = []
    for field in sorted(forwardable):
        wire_name = FIELD_RENAMES.get(field, field)
        if wire_name not in captured_payload:
            missing.append(field)

    assert not missing, (
        f"These AnthropicMessagesRequest fields were not forwarded to the client payload: {missing}. "
        f"Update the messages endpoint or model adapter to forward them."
    )


@pytest.mark.asyncio
async def test_values_match_request():
    """Forwarded values must match what was in the original request."""
    captured_payload = {}

    client = AnthropicClient(base_url="http://test.local/v1", api_key="fake")
    model = AnthropicChatModel(client=client, model="test-model")

    original_build = client._build_payload

    def capturing_build(*args, **kwargs):
        result = original_build(*args, **kwargs)
        captured_payload.update(result)
        return result

    with (
        patch.object(client, "_build_payload", side_effect=capturing_build),
        patch.object(client, "_apost", new_callable=AsyncMock, return_value=ANTHROPIC_RESPONSE),
    ):
        body = AnthropicMessagesRequest(**REQUEST_WITH_ALL_FIELDS)

        params = {}
        for field in _forwardable_fields():
            value = getattr(body, field)
            if value is not None:
                wire_name = FIELD_RENAMES.get(field, field)
                params[wire_name] = value

        await model.generate_async(
            [ChatMessage(role=Role.USER, content="Hi")],
            **params,
        )

    for field in _forwardable_fields():
        wire_name = FIELD_RENAMES.get(field, field)
        expected = REQUEST_WITH_ALL_FIELDS[field]
        assert captured_payload[wire_name] == expected, (
            f"Payload value for '{wire_name}' was {captured_payload.get(wire_name)!r}, expected {expected!r}"
        )


@pytest.mark.asyncio
async def test_system_prompt_forwarded():
    """The system field must be forwarded as a top-level param, not inside messages."""
    captured_payload = {}

    client = AnthropicClient(base_url="http://test.local/v1", api_key="fake")
    model = AnthropicChatModel(client=client, model="test-model")

    original_build = client._build_payload

    def capturing_build(*args, **kwargs):
        result = original_build(*args, **kwargs)
        captured_payload.update(result)
        return result

    with (
        patch.object(client, "_build_payload", side_effect=capturing_build),
        patch.object(client, "_apost", new_callable=AsyncMock, return_value=ANTHROPIC_RESPONSE),
    ):
        messages = [
            ChatMessage(role=Role.SYSTEM, content="You are helpful."),
            ChatMessage(role=Role.USER, content="Hi"),
        ]

        await model.generate_async(messages, max_tokens=100)

    assert captured_payload.get("system") == "You are helpful."
    for msg in captured_payload["messages"]:
        assert msg["role"] != "system", "System messages should not appear in the messages list"


def test_request_with_all_fields_covers_forwardable():
    """Ensure REQUEST_WITH_ALL_FIELDS includes every forwardable field.

    This test fails when a new field is added to AnthropicMessagesRequest but
    not to the test fixture, preventing silent gaps in coverage.
    """
    forwardable = _forwardable_fields()
    all_test_fields = set(REQUEST_WITH_ALL_FIELDS.keys()) - FIELDS_NOT_FORWARDED_TO_LLM
    missing_from_fixture = forwardable - all_test_fields
    assert not missing_from_fixture, (
        f"REQUEST_WITH_ALL_FIELDS is missing forwardable fields: {missing_from_fixture}. "
        f"Add them to the test fixture or to FIELDS_NOT_FORWARDED_TO_LLM."
    )
