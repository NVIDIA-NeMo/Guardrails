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
from typing import AsyncIterator, Union
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from nemoguardrails.server import api
from nemoguardrails.server.api import _format_streaming_response
from nemoguardrails.server.schemas.openai import GuardrailsChatCompletionRequest

LIVE_TEST_MODE = os.environ.get("LIVE_TEST_MODE") or os.environ.get("TEST_LIVE_MODE")

client = TestClient(api.app)


@pytest.fixture(scope="function", autouse=True)
def set_rails_config_path():
    api.app.rails_config_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "test_configs"))
    yield
    api.app.rails_config_path = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..", "examples", "bots")
    )


def test_get():
    response = client.get("/v1/rails/configs")
    assert response.status_code == 200

    result = response.json()
    assert len(result) > 0


def test_get_models_default_env_vars():
    """Test the OpenAI-compatible /v1/models endpoint."""
    response = client.get("/v1/models")
    assert response.status_code == 200

    result = response.json()

    # Check OpenAI models list format
    assert result["object"] == "list"
    assert "data" in result
    assert len(result["data"]) > 0

    # Check each model has the required OpenAI format
    for model in result["data"]:
        assert "id" in model
        assert "config_id" in model
        assert model["object"] == "model"
        assert "created" in model
        assert model["owned_by"] == "nemo-guardrails"
        assert model["engine"] == "nim"
        assert model["base_url"] == "https://localhost:8000/v1"
        assert model["api_key_env_var"] is None


def test_get_models_with_custom_env_vars():
    with patch.dict(
        os.environ,
        {
            "MAIN_MODEL_ENGINE": "custom-engine",
            "MAIN_MODEL_BASE_URL": "https://custom-api.example.com/v1",
            "MAIN_MODEL_API_KEY": "custom-api-key",
        },
    ):
        response = client.get("/v1/models")
        assert response.status_code == 200
        result = response.json()
        for model in result["data"]:
            assert model["engine"] == "custom-engine"
            assert model["base_url"] == "https://custom-api.example.com/v1"
            assert model["api_key_env_var"] == "custom-api-key"


@pytest.mark.skipif(
    not LIVE_TEST_MODE,
    reason="This test requires LIVE_TEST_MODE or TEST_LIVE_MODE environment variable to be set for live testing",
)
def test_chat_completion():
    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [
                {
                    "content": "Hello",
                    "role": "user",
                }
            ],
            "guardrails": {"config_id": "general"},
        },
    )
    assert response.status_code == 200
    res = response.json()
    # Check OpenAI-compatible response structure
    assert res["object"] == "chat.completion"
    assert "id" in res
    assert "created" in res
    assert "model" in res
    assert len(res["choices"]) == 1
    assert res["choices"][0]["message"]["content"]
    assert res["choices"][0]["message"]["role"] == "assistant"


@pytest.mark.skipif(
    not LIVE_TEST_MODE,
    reason="This test requires LIVE_TEST_MODE or TEST_LIVE_MODE environment variable to be set for live testing",
)
def test_chat_completion_with_default_configs():
    api.set_default_config_id("general")

    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [
                {
                    "content": "Hello",
                    "role": "user",
                }
            ],
        },
    )
    assert response.status_code == 200
    res = response.json()
    # Check OpenAI-compatible response structure
    assert res["object"] == "chat.completion"
    assert "id" in res
    assert "created" in res
    assert "model" in res
    assert len(res["choices"]) == 1
    assert res["choices"][0]["message"]["content"]
    assert res["choices"][0]["message"]["role"] == "assistant"


def test_request_body_validation():
    """Test GuardrailsChatCompletionRequest validation."""

    data = {
        "messages": [{"role": "user", "content": "Hello"}],
        "guardrails": {"config_id": "test_config"},
    }
    request_body = GuardrailsChatCompletionRequest.model_validate(data)
    assert request_body.guardrails.config_id == "test_config"
    assert request_body.guardrails.config_ids == ["test_config"]

    data = {
        "messages": [{"role": "user", "content": "Hello"}],
        "guardrails": {"config_ids": ["test_config1", "test_config2"]},
    }
    request_body = GuardrailsChatCompletionRequest.model_validate(data)
    assert request_body.guardrails.config_ids == ["test_config1", "test_config2"]

    data = {
        "messages": [{"role": "user", "content": "Hello"}],
        "guardrails": {
            "config_id": "test_config",
            "config_ids": ["test_config1", "test_config2"],
        },
    }
    with pytest.raises(ValueError, match="Only one of config_id or config_ids should be specified"):
        GuardrailsChatCompletionRequest.model_validate(data)

    data = {"messages": [{"role": "user", "content": "Hello"}]}
    request_body = GuardrailsChatCompletionRequest.model_validate(data)
    assert request_body.guardrails.config_ids is None


def test_model_field_independent_of_config_id():
    """Test that model field is independent of config_id."""

    data = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Hello"}],
        "guardrails": {"config_id": "test_config"},
    }
    request_body = GuardrailsChatCompletionRequest.model_validate(data)
    assert request_body.model == "gpt-4"
    assert request_body.guardrails.config_id == "test_config"
    assert request_body.guardrails.config_ids == ["test_config"]


def test_request_body_state():
    """Test GuardrailsChatCompletionRequest state handling."""
    data = {
        "messages": [{"role": "user", "content": "Hello"}],
        "guardrails": {
            "config_id": "test_config",
            "state": {"key": "value"},
        },
    }
    request_body = GuardrailsChatCompletionRequest.model_validate(data)
    assert request_body.guardrails.state == {"key": "value"}


def test_request_body_messages():
    """Test GuardrailsChatCompletionRequest messages validation."""
    data = {
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ],
        "guardrails": {"config_id": "test_config"},
    }
    request_body = GuardrailsChatCompletionRequest.model_validate(data)
    assert request_body.messages is not None
    assert len(request_body.messages) == 2

    data = {
        "messages": [{"content": "Hello"}],
        "guardrails": {"config_id": "test_config"},
    }
    request_body = GuardrailsChatCompletionRequest.model_validate(data)
    assert request_body.messages is not None
    assert len(request_body.messages) == 1


async def _create_test_stream(chunks: list) -> AsyncIterator[Union[str, dict]]:
    """Helper to create an async iterator for testing."""
    for chunk in chunks:
        yield chunk


@pytest.mark.asyncio
async def test_openai_sse_format_basic_chunks():
    """Test basic string chunks are properly formatted as SSE events."""
    # Create a test stream with string chunks
    stream = _create_test_stream(["Hello ", "world"])

    # Collect yielded SSE messages
    collected = []
    async for b in _format_streaming_response(stream, model_name=None):
        collected.append(b)

    # We expect three messages: two data: {json}\n\n events and final data: [DONE]\n\n
    assert len(collected) == 3
    # First two are JSON SSE events
    evt1 = collected[0]
    evt2 = collected[1]
    done = collected[2]

    assert evt1.startswith("data: ")
    j1 = json.loads(evt1[len("data: ") :].strip())
    assert j1["object"] == "chat.completion.chunk"
    assert j1["choices"][0]["delta"]["content"] == "Hello "

    assert evt2.startswith("data: ")
    j2 = json.loads(evt2[len("data: ") :].strip())
    assert j2["choices"][0]["delta"]["content"] == "world"

    assert done == "data: [DONE]\n\n"


@pytest.mark.asyncio
async def test_openai_sse_format_with_model_name():
    """Test that model name is properly included in the response."""
    stream = _create_test_stream(["Test"])
    collected = []

    async for b in _format_streaming_response(stream, model_name="gpt-4"):
        collected.append(b)

    assert len(collected) == 2
    evt = collected[0]
    j = json.loads(evt[len("data: ") :].strip())
    assert j["model"] == "gpt-4"
    assert j["choices"][0]["delta"]["content"] == "Test"
    assert collected[1] == "data: [DONE]\n\n"


@pytest.mark.asyncio
async def test_openai_sse_format_with_dict_chunk():
    """Test that dict chunks with role and content are properly formatted."""
    stream = _create_test_stream([{"role": "assistant", "content": "Hi!"}])
    collected = []

    async for b in _format_streaming_response(stream, model_name=None):
        collected.append(b)

    assert len(collected) == 2
    evt = collected[0]
    j = json.loads(evt[len("data: ") :].strip())
    assert j["object"] == "chat.completion.chunk"
    assert j["choices"][0]["delta"]["role"] == "assistant"
    assert j["choices"][0]["delta"]["content"] == "Hi!"
    assert collected[1] == "data: [DONE]\n\n"


@pytest.mark.asyncio
async def test_openai_sse_format_empty_string():
    """Test that empty strings are handled correctly."""
    stream = _create_test_stream([""])
    collected = []

    async for b in _format_streaming_response(stream, model_name=None):
        collected.append(b)

    assert len(collected) == 2
    evt = collected[0]
    j = json.loads(evt[len("data: ") :].strip())
    assert j["choices"][0]["delta"]["content"] == ""
    assert collected[1] == "data: [DONE]\n\n"


@pytest.mark.asyncio
async def test_openai_sse_format_none_triggers_done():
    """Test that None values are handled correctly."""
    stream = _create_test_stream(["Content", None])
    collected = []

    async for b in _format_streaming_response(stream, model_name=None):
        collected.append(b)

    assert len(collected) == 3  # Content chunk, None chunk, and [DONE]
    evt = collected[0]
    j = json.loads(evt[len("data: ") :].strip())
    assert j["choices"][0]["delta"]["content"] == "Content"
    assert collected[2] == "data: [DONE]\n\n"


@pytest.mark.asyncio
async def test_openai_sse_format_multiple_dict_chunks():
    """Test multiple dict chunks with different fields."""
    stream = _create_test_stream([{"role": "assistant"}, {"content": "Hello"}, {"content": " world"}])
    collected = []

    async for b in _format_streaming_response(stream, model_name="test-model"):
        collected.append(b)

    assert len(collected) == 4

    # Check first chunk (role only)
    j1 = json.loads(collected[0][len("data: ") :].strip())
    assert j1["choices"][0]["delta"]["role"] == "assistant"
    assert "content" not in j1["choices"][0]["delta"]

    # Check second chunk (content only)
    j2 = json.loads(collected[1][len("data: ") :].strip())
    assert j2["choices"][0]["delta"]["content"] == "Hello"

    # Check third chunk (content only)
    j3 = json.loads(collected[2][len("data: ") :].strip())
    assert j3["choices"][0]["delta"]["content"] == " world"

    # Check [DONE] message
    assert collected[3] == "data: [DONE]\n\n"


@pytest.mark.asyncio
async def test_openai_sse_format_special_characters():
    """Test that special characters are properly escaped in JSON."""
    stream = _create_test_stream(["Line 1\nLine 2", 'Quote: "test"'])
    collected = []

    async for b in _format_streaming_response(stream, model_name=None):
        collected.append(b)

    assert len(collected) == 3

    # Verify first chunk with newline
    j1 = json.loads(collected[0][len("data: ") :].strip())
    assert j1["choices"][0]["delta"]["content"] == "Line 1\nLine 2"

    # Verify second chunk with quotes
    j2 = json.loads(collected[1][len("data: ") :].strip())
    assert j2["choices"][0]["delta"]["content"] == 'Quote: "test"'

    assert collected[2] == "data: [DONE]\n\n"


@pytest.mark.asyncio
async def test_openai_sse_format_events():
    """Test that all events follow proper SSE format."""
    stream = _create_test_stream(["Test"])
    collected = []

    async for b in _format_streaming_response(stream, model_name=None):
        collected.append(b)

    # All events except [DONE] should be valid JSON with proper SSE format
    for event in collected[:-1]:
        assert event.startswith("data: ")
        assert event.endswith("\n\n")
        # Verify it's valid JSON
        json_str = event[len("data: ") :].strip()
        j = json.loads(json_str)
        assert "object" in j
        assert "choices" in j
        assert isinstance(j["choices"], list)
        assert len(j["choices"]) > 0

    # Last event should be [DONE]
    assert collected[-1] == "data: [DONE]\n\n"


@pytest.mark.asyncio
async def test_openai_sse_format_chunk_metadata():
    """Test that chunk metadata is properly formatted."""
    stream = _create_test_stream(["Test"])
    collected = []

    async for b in _format_streaming_response(stream, model_name="test-model"):
        collected.append(b)

    evt = collected[0]
    j = json.loads(evt[len("data: ") :].strip())

    # Verify all required fields are present
    assert "id" in j  # id should be present (UUID generated)
    assert j["object"] == "chat.completion.chunk"
    assert isinstance(j["created"], int)
    assert j["model"] == "test-model"
    assert isinstance(j["choices"], list)
    assert len(j["choices"]) == 1

    choice = j["choices"][0]
    assert "delta" in choice
    assert choice["index"] == 0
    assert choice["finish_reason"] is None


@pytest.mark.skip(reason="Should only be run locally as it needs OpenAI key.")
def test_chat_completion_with_streaming():
    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
            "guardrails": {"config_id": "general"},
        },
    )
    assert response.status_code == 200
    assert response.headers["Content-Type"] == "text/event-stream"
    for chunk in response.iter_lines():
        assert chunk.startswith("data: ")
        assert chunk.endswith("\n\n")
    assert "data: [DONE]\n\n" in response.text
