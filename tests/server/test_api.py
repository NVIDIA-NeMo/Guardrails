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

pytest.importorskip("openai", reason="openai is required for server tests")
from fastapi.testclient import TestClient

from nemoguardrails.server import api
from nemoguardrails.server.api import _format_streaming_response
from nemoguardrails.server.schemas.openai import GuardrailsChatCompletionRequest

LIVE_TEST_MODE = os.environ.get("LIVE_TEST_MODE") or os.environ.get("TEST_LIVE_MODE")

client = TestClient(api.app)


@pytest.fixture(scope="function", autouse=True)
def set_rails_config_path():
    original_path = api.app.rails_config_path
    original_engine = os.environ.get("MAIN_MODEL_ENGINE")
    api.app.rails_config_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "test_configs"))
    os.environ["MAIN_MODEL_ENGINE"] = "custom_llm"
    api.llm_rails_instances.clear()
    yield
    api.app.rails_config_path = original_path
    api.llm_rails_instances.clear()
    if original_engine is not None:
        os.environ["MAIN_MODEL_ENGINE"] = original_engine
    else:
        os.environ.pop("MAIN_MODEL_ENGINE", None)


def test_get():
    response = client.get("/v1/rails/configs")
    assert response.status_code == 200

    result = response.json()
    assert len(result) > 0


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
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hello"}],
        "guardrails": {"config_id": "test_config"},
    }
    request_body = GuardrailsChatCompletionRequest.model_validate(data)
    assert request_body.guardrails.config_id == "test_config"
    assert request_body.guardrails.config_ids == ["test_config"]

    data = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hello"}],
        "guardrails": {"config_ids": ["test_config1", "test_config2"]},
    }
    request_body = GuardrailsChatCompletionRequest.model_validate(data)
    assert request_body.guardrails.config_ids == ["test_config1", "test_config2"]

    data = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hello"}],
        "guardrails": {
            "config_id": "test_config",
            "config_ids": ["test_config1", "test_config2"],
        },
    }
    with pytest.raises(ValueError, match="Only one of config_id or config_ids should be specified"):
        GuardrailsChatCompletionRequest.model_validate(data)

    data = {"model": "gpt-4o", "messages": [{"role": "user", "content": "Hello"}]}
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
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hello"}],
        "guardrails": {
            "config_id": "test_config",
            "state": {"key": "value"},
        },
    }
    request_body = GuardrailsChatCompletionRequest.model_validate(data)
    assert request_body.guardrails.state == {"key": "value"}


def test_request_body_context():
    """Test GuardrailsChatCompletionRequest context handling."""
    data = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hello"}],
        "guardrails": {
            "config_id": "test_config",
            "context": {"user_name": "John", "session_id": "abc123"},
        },
    }
    request_body = GuardrailsChatCompletionRequest.model_validate(data)
    assert request_body.guardrails.context == {"user_name": "John", "session_id": "abc123"}


def test_request_body_messages():
    """Test GuardrailsChatCompletionRequest messages validation."""
    data = {
        "model": "gpt-4o",
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
        "model": "gpt-4o",
        "messages": [{"content": "Hello"}],
        "guardrails": {"config_id": "test_config"},
    }
    request_body = GuardrailsChatCompletionRequest.model_validate(data)
    assert request_body.messages is not None
    assert len(request_body.messages) == 1


def test_request_body_options():
    """Test GuardrailsChatCompletionRequest options handling."""
    data = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hello"}],
        "guardrails": {
            "config_id": "test_config",
            "options": {
                "rails": {"input": False, "output": True, "dialog": False},
                "llm_params": {"temperature": 0.5},
                "output_vars": ["relevant_chunks"],
                "log": {"activated_rails": True, "llm_calls": True},
            },
        },
    }
    request_body = GuardrailsChatCompletionRequest.model_validate(data)
    assert request_body.guardrails.options.rails.input is False
    assert request_body.guardrails.options.rails.output is True
    assert request_body.guardrails.options.rails.dialog is False
    assert request_body.guardrails.options.llm_params == {"temperature": 0.5}
    assert request_body.guardrails.options.output_vars == ["relevant_chunks"]
    assert request_body.guardrails.options.log.activated_rails is True
    assert request_body.guardrails.options.log.llm_calls is True


def test_request_body_options_with_rail_names():
    """Test options with specific rail names instead of booleans."""
    data = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hello"}],
        "guardrails": {
            "config_id": "test_config",
            "options": {
                "rails": {
                    "input": ["check jailbreak", "check toxicity"],
                    "output": ["output moderation"],
                },
            },
        },
    }
    request_body = GuardrailsChatCompletionRequest.model_validate(data)
    assert request_body.guardrails.options.rails.input == ["check jailbreak", "check toxicity"]
    assert request_body.guardrails.options.rails.output == ["output moderation"]


def test_guardrails_defaults_when_not_provided():
    """Test that guardrails field has proper defaults when not provided."""
    data = {"model": "gpt-4o", "messages": [{"role": "user", "content": "Hello"}]}
    request_body = GuardrailsChatCompletionRequest.model_validate(data)

    assert request_body.guardrails is not None
    assert request_body.guardrails.config_id is None
    assert request_body.guardrails.config_ids is None
    assert request_body.guardrails.thread_id is None
    assert request_body.guardrails.context is None
    assert request_body.guardrails.state is None
    assert request_body.guardrails.options is not None
    assert request_body.guardrails.options.rails.input is True
    assert request_body.guardrails.options.rails.output is True


def test_guardrails_defaults_when_empty_object():
    """Test that guardrails field has proper defaults when empty object provided."""
    data = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hello"}],
        "guardrails": {},
    }
    request_body = GuardrailsChatCompletionRequest.model_validate(data)

    assert request_body.guardrails.config_id is None
    assert request_body.guardrails.config_ids is None
    assert request_body.guardrails.options is not None


def test_guardrails_partial_fields():
    """Test that guardrails works with only some fields provided."""
    data = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hello"}],
        "guardrails": {"config_id": "test_config"},
    }
    request_body = GuardrailsChatCompletionRequest.model_validate(data)

    assert request_body.guardrails.config_id == "test_config"
    assert request_body.guardrails.context is None
    assert request_body.guardrails.state is None
    assert request_body.guardrails.options is not None


def test_default_config_id_from_env():
    """Test that DEFAULT_CONFIG_ID env var sets default config_id."""
    with patch.dict(os.environ, {"DEFAULT_CONFIG_ID": "env_config"}):
        from nemoguardrails.server.schemas.openai import GuardrailsDataInput

        guardrails = GuardrailsDataInput()
        assert guardrails.config_id == "env_config"


def test_no_config_error_returns_proper_response():
    """Test API returns proper error response when no config_id and no default."""
    api.app.default_config_id = None
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )
    assert response.status_code == 422
    res = response.json()
    assert "detail" in res
    assert "config" in res["detail"].lower()


def test_invalid_state_returns_error():
    """Test API handles invalid state gracefully instead of crashing."""
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
            "guardrails": {
                "config_id": "with_custom_llm",
                "state": {"invalid_key": "value"},
            },
        },
    )
    assert response.status_code == 422
    res = response.json()
    assert "detail" in res
    assert "state" in res["detail"].lower() or "events" in res["detail"].lower()


def test_chat_completion_response_structure():
    """Test that chat completion response includes proper structure."""
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
            "guardrails": {"config_id": "with_custom_llm"},
        },
    )
    assert response.status_code == 200
    res = response.json()

    assert res["id"].startswith("chatcmpl-")
    assert res["object"] == "chat.completion"
    assert isinstance(res["created"], int)
    assert res["created"] > 0
    assert res["model"] == "gpt-4o"
    assert len(res["choices"]) == 1
    assert res["choices"][0]["index"] == 0
    assert res["choices"][0]["finish_reason"] == "stop"
    assert res["choices"][0]["message"]["role"] == "assistant"
    assert res["choices"][0]["message"]["content"] == "Custom LLM response"
    assert res["guardrails"]["config_id"] == "with_custom_llm"


def test_chat_completion_with_context():
    """Test chat completion with context field."""
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
            "guardrails": {
                "config_id": "with_custom_llm",
                "context": {"user_id": "123", "session": "abc"},
            },
        },
    )
    assert response.status_code == 200
    res = response.json()
    assert res["object"] == "chat.completion"
    assert res["model"] == "gpt-4o"
    assert res["choices"][0]["message"]["content"] == "Custom LLM response"
    assert res["guardrails"]["config_id"] == "with_custom_llm"


def test_chat_completion_with_options():
    """Test chat completion with custom options."""
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
            "guardrails": {
                "config_id": "with_custom_llm",
                "options": {
                    "rails": {"input": False, "output": False},
                },
            },
        },
    )
    assert response.status_code == 200
    res = response.json()
    assert res["object"] == "chat.completion"
    assert res["model"] == "gpt-4o"
    assert res["choices"][0]["message"]["content"] == "Custom LLM response"
    assert res["guardrails"]["config_id"] == "with_custom_llm"


def test_chat_completion_with_all_guardrails_fields():
    """Test chat completion with all guardrails fields populated."""
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
            "guardrails": {
                "config_id": "with_custom_llm",
                "context": {"user_id": "123"},
                "options": {
                    "rails": {"input": True, "output": True},
                    "log": {"activated_rails": True},
                },
                "state": {},
            },
        },
    )
    assert response.status_code == 200
    res = response.json()

    assert res["object"] == "chat.completion"
    assert res["model"] == "gpt-4o"
    assert res["choices"][0]["message"]["content"] == "Custom LLM response"
    assert res["guardrails"]["config_id"] == "with_custom_llm"

    assert "log" in res["guardrails"]
    assert res["guardrails"]["log"] is not None
    assert "activated_rails" in res["guardrails"]["log"]
    assert isinstance(res["guardrails"]["log"]["activated_rails"], list)
    assert "stats" in res["guardrails"]["log"]
    assert isinstance(res["guardrails"]["log"]["stats"], dict)
    assert "total_duration" in res["guardrails"]["log"]["stats"]


def test_chat_completion_with_log_llm_calls():
    """Test chat completion returns llm_calls when requested."""
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
            "guardrails": {
                "config_id": "with_custom_llm",
                "options": {
                    "log": {"llm_calls": True},
                },
            },
        },
    )
    assert response.status_code == 200
    res = response.json()

    assert res["choices"][0]["message"]["content"] == "Custom LLM response"
    assert "log" in res["guardrails"]
    assert res["guardrails"]["log"] is not None
    assert "llm_calls" in res["guardrails"]["log"]
    assert isinstance(res["guardrails"]["log"]["llm_calls"], list)
    assert len(res["guardrails"]["log"]["llm_calls"]) >= 1
    llm_call = res["guardrails"]["log"]["llm_calls"][0]
    assert "prompt" in llm_call
    assert "completion" in llm_call


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
