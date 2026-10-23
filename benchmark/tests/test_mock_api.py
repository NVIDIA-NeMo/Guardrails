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
import time

import pytest
from fastapi.testclient import TestClient

from benchmark.mock_llm_server.api import app
from benchmark.mock_llm_server.config import ModelSettings, get_settings

UNSAFE_RESPONSE = "I cannot help with that request"
SAFE_RESPONSE = "This is a safe response, with many words to test streaming where we split on whitespace"


def get_test_settings():
    return ModelSettings(
        model="gpt-3.5-turbo",
        unsafe_probability=0.0,
        unsafe_text=UNSAFE_RESPONSE,
        safe_text=SAFE_RESPONSE,
        e2e_latency_min_seconds=0,
        e2e_latency_max_seconds=0,
        e2e_latency_mean_seconds=0,
        e2e_latency_std_seconds=0,
        # Streaming latency settings (set to 0 for fast tests)
        ttft_min_seconds=0,
        ttft_max_seconds=0,
        ttft_mean_seconds=0,
        ttft_std_seconds=0,
        chunk_latency_min_seconds=0,
        chunk_latency_max_seconds=0,
        chunk_latency_mean_seconds=0,
        chunk_latency_std_seconds=0,
    )


@pytest.fixture
def client():
    """Create a test client."""
    app.dependency_overrides[get_settings] = get_test_settings
    return TestClient(app)


def test_get_root_endpoint_server_data(client):
    """Test GET / endpoint returns correct server details (not including model info)"""

    model_name = get_test_settings().model

    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Mock LLM Server"
    assert data["version"] == "0.0.1"
    assert data["description"] == f"OpenAI-compatible mock LLM server for model: {model_name}"
    assert data["endpoints"] == [
        "/v1/models",
        "/v1/chat/completions",
        "/v1/completions",
    ]


def test_get_root_endpoint_model_data(client):
    """Test GET / endpoint returns correct model details"""

    response = client.get("/")
    data = response.json()
    model_data = data["model_configuration"]

    expected_model_data = get_test_settings().model_dump()
    assert model_data == expected_model_data


def test_get_health_endpoint(client):
    """Test GET /health endpoint."""
    pre_request_time = int(time.time())
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data
    assert isinstance(data["timestamp"], int)
    assert data["timestamp"] >= pre_request_time


def test_get_models_endpoint(client):
    """Test GET /v1/models endpoint."""
    pre_request_time = int(time.time())
    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert len(data["data"]) == 1

    expected_model = get_test_settings().model_dump()
    model = data["data"][0]
    assert model["id"] == expected_model["model"]
    assert model["object"] == "model"
    assert isinstance(model["created"], int)
    assert model["created"] >= pre_request_time
    assert model["owned_by"] == "system"


class TestChatCompletionsEndpoint:
    """Test the /v1/chat/completions endpoint."""

    def test_chat_completions_basic(self, client):
        """Test basic chat completion request."""
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Hello"}],
        }
        response = client.post("/v1/chat/completions", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "chat.completion"
        assert data["model"] == "gpt-3.5-turbo"
        assert "id" in data
        assert data["id"].startswith("chatcmpl-")

    def test_chat_completions_response_structure(self, client):
        """Test the structure of chat completion response."""
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Test message"}],
        }
        response = client.post("/v1/chat/completions", json=payload)
        data = response.json()

        # Check response structure
        assert "choices" in data
        assert len(data["choices"]) == 1
        choice = data["choices"][0]
        assert choice["index"] == 0
        assert "message" in choice
        assert choice["message"]["role"] == "assistant"
        assert "content" in choice["message"]
        assert choice["finish_reason"] == "stop"

    def test_chat_completions_usage(self, client):
        """Test that usage information is included."""
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Test"}],
        }
        response = client.post("/v1/chat/completions", json=payload)
        data = response.json()

        assert "usage" in data
        usage = data["usage"]
        assert "prompt_tokens" in usage
        assert "completion_tokens" in usage
        assert "total_tokens" in usage
        assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]

    def test_chat_completions_multiple_choices(self, client):
        """Test chat completion with n > 1."""
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Hello"}],
            "n": 3,
        }
        response = client.post("/v1/chat/completions", json=payload)
        data = response.json()

        assert len(data["choices"]) == 3
        for i, choice in enumerate(data["choices"]):
            assert choice["index"] == i

    def test_chat_completions_multiple_messages(self, client):
        """Test chat completion with multiple messages."""
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "How are you?"},
            ],
        }
        response = client.post("/v1/chat/completions", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "choices" in data

    def test_chat_completions_invalid_model(self, client):
        """Test chat completion with invalid model name."""
        payload = {
            "model": "invalid-model",
            "messages": [{"role": "user", "content": "Hello"}],
        }
        response = client.post("/v1/chat/completions", json=payload)
        assert response.status_code == 400
        assert "not found" in response.json()["detail"].lower()

    def test_chat_completions_missing_messages(self, client):
        """Test chat completion without messages field."""
        payload = {
            "model": "gpt-3.5-turbo",
        }
        response = client.post("/v1/chat/completions", json=payload)
        assert response.status_code == 422  # Validation error

    def test_chat_completions_empty_messages(self, client):
        """Test chat completion with empty messages list."""
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [],
        }
        response = client.post("/v1/chat/completions", json=payload)
        # Should either be 422 or 200 depending on validation
        # Let's check it doesn't crash
        assert response.status_code in [200, 422]


class TestCompletionsEndpoint:
    """Test the /v1/completions endpoint."""

    def test_completions_basic(self, client):
        """Test basic completion request."""
        payload = {
            "model": "gpt-3.5-turbo",
            "prompt": "Once upon a time",
        }
        response = client.post("/v1/completions", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "text_completion"
        assert data["model"] == "gpt-3.5-turbo"
        assert data["id"].startswith("cmpl-")

    def test_completions_response_structure(self, client):
        """Test the structure of completion response."""
        payload = {
            "model": "gpt-3.5-turbo",
            "prompt": "Test prompt",
        }
        response = client.post("/v1/completions", json=payload)
        data = response.json()

        assert "choices" in data
        assert len(data["choices"]) == 1
        choice = data["choices"][0]
        assert choice["index"] == 0
        assert "text" in choice
        assert isinstance(choice["text"], str)
        assert choice["finish_reason"] == "stop"
        assert choice["logprobs"] is None

    def test_completions_string_prompt(self, client):
        """Test completion with string prompt."""
        payload = {
            "model": "gpt-3.5-turbo",
            "prompt": "Single string prompt",
        }
        response = client.post("/v1/completions", json=payload)
        assert response.status_code == 200

    def test_completions_list_prompt(self, client):
        """Test completion with list of prompts."""
        payload = {
            "model": "gpt-3.5-turbo",
            "prompt": ["Prompt 1", "Prompt 2", "Prompt 3"],
        }
        response = client.post("/v1/completions", json=payload)
        assert response.status_code == 200
        data = response.json()
        # Should still return a response (joined prompts)
        assert "choices" in data

    def test_completions_multiple_choices(self, client):
        """Test completion with n > 1."""
        payload = {
            "model": "gpt-3.5-turbo",
            "prompt": "Test",
            "n": 5,
        }
        response = client.post("/v1/completions", json=payload)
        data = response.json()

        assert len(data["choices"]) == 5
        for i, choice in enumerate(data["choices"]):
            assert choice["index"] == i

    def test_completions_usage(self, client):
        """Test that usage information is included."""
        payload = {
            "model": "gpt-3.5-turbo",
            "prompt": "Test prompt",
        }
        response = client.post("/v1/completions", json=payload)
        data = response.json()

        assert "usage" in data
        usage = data["usage"]
        assert usage["prompt_tokens"] > 0
        assert usage["completion_tokens"] > 0
        assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]

    def test_completions_invalid_model(self, client):
        """Test completion with invalid model name."""
        payload = {
            "model": "wrong-model",
            "prompt": "Test",
        }
        response = client.post("/v1/completions", json=payload)
        assert response.status_code == 400

    def test_completions_missing_prompt(self, client):
        """Test completion without prompt field."""
        payload = {
            "model": "gpt-3.5-turbo",
        }
        response = client.post("/v1/completions", json=payload)
        assert response.status_code == 422  # Validation error


class TestMiddleware:
    """Test the HTTP logging middleware."""

    def test_middleware_logs_request(self, client):
        """Test that middleware processes requests."""
        # The middleware should not affect response
        response = client.get("/health")
        assert response.status_code == 200

    def test_middleware_with_post(self, client):
        """Test middleware with POST request."""
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Test"}],
        }
        response = client.post("/v1/chat/completions", json=payload)
        assert response.status_code == 200


class TestValidateRequestModel:
    """Test the _validate_request_model function."""

    def test_validate_request_model_valid(self, client):
        """Test validation with correct model."""
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Test"}],
        }
        response = client.post("/v1/chat/completions", json=payload)
        assert response.status_code == 200

    def test_validate_request_model_invalid(self, client):
        """Test validation with incorrect model."""
        payload = {
            "model": "nonexistent-model",
            "messages": [{"role": "user", "content": "Test"}],
        }
        response = client.post("/v1/chat/completions", json=payload)
        assert response.status_code == 400
        assert "not found" in response.json()["detail"].lower()
        assert "gpt-3.5-turbo" in response.json()["detail"]


def get_safe_only_settings():
    """Settings with unsafe_probability=0 (always safe response)."""
    return ModelSettings(
        model="gpt-3.5-turbo",
        unsafe_probability=0.0,
        unsafe_text=UNSAFE_RESPONSE,
        safe_text=SAFE_RESPONSE,
        e2e_latency_min_seconds=0,
        e2e_latency_max_seconds=0,
        e2e_latency_mean_seconds=0,
        e2e_latency_std_seconds=0,
        ttft_min_seconds=0,
        ttft_max_seconds=0,
        ttft_mean_seconds=0,
        ttft_std_seconds=0,
        chunk_latency_min_seconds=0,
        chunk_latency_max_seconds=0,
        chunk_latency_mean_seconds=0,
        chunk_latency_std_seconds=0,
    )


def get_unsafe_only_settings():
    """Settings with unsafe_probability=1.0 (always unsafe response)."""
    return ModelSettings(
        model="gpt-3.5-turbo",
        unsafe_probability=1.0,
        unsafe_text=UNSAFE_RESPONSE,
        safe_text=SAFE_RESPONSE,
        e2e_latency_min_seconds=0,
        e2e_latency_max_seconds=0,
        e2e_latency_mean_seconds=0,
        e2e_latency_std_seconds=0,
        ttft_min_seconds=0,
        ttft_max_seconds=0,
        ttft_mean_seconds=0,
        ttft_std_seconds=0,
        chunk_latency_min_seconds=0,
        chunk_latency_max_seconds=0,
        chunk_latency_mean_seconds=0,
        chunk_latency_std_seconds=0,
    )


@pytest.fixture
def safe_client():
    """Create a test client that always returns safe responses."""
    app.dependency_overrides[get_settings] = get_safe_only_settings
    yield TestClient(app)
    app.dependency_overrides[get_settings] = get_test_settings


@pytest.fixture
def unsafe_client():
    """Create a test client that always returns unsafe responses."""
    app.dependency_overrides[get_settings] = get_unsafe_only_settings
    yield TestClient(app)
    app.dependency_overrides[get_settings] = get_test_settings


class TestResponseContent:
    """Test that responses contain expected content."""

    def test_chat_response_content_type(self, client):
        """Test that response contains either safe or unsafe text."""
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Test"}],
        }
        response = client.post("/v1/chat/completions", json=payload)
        data = response.json()

        content = data["choices"][0]["message"]["content"]
        # Should be one of the configured responses
        assert content == SAFE_RESPONSE

    def test_completion_response_content_type(self, client):
        """Test that completion response contains expected text."""
        payload = {
            "model": "gpt-3.5-turbo",
            "prompt": "Test",
        }
        response = client.post("/v1/completions", json=payload)
        data = response.json()

        text = data["choices"][0]["text"]
        # Should be one of the configured responses
        assert text in {SAFE_RESPONSE, UNSAFE_RESPONSE}

    def test_chat_completions_safe_response_when_probability_zero(self, safe_client):
        """Test that chat completions returns safe response when unsafe_probability=0."""
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Test"}],
        }
        response = safe_client.post("/v1/chat/completions", json=payload)
        data = response.json()

        content = data["choices"][0]["message"]["content"]
        assert content == SAFE_RESPONSE

    def test_chat_completions_unsafe_response_when_probability_one(self, unsafe_client):
        """Test that chat completions returns unsafe response when unsafe_probability=1.0."""
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Test"}],
        }
        response = unsafe_client.post("/v1/chat/completions", json=payload)
        data = response.json()

        content = data["choices"][0]["message"]["content"]
        assert content == UNSAFE_RESPONSE

    def test_completions_safe_response_when_probability_zero(self, safe_client):
        """Test that completions returns safe response when unsafe_probability=0."""
        payload = {
            "model": "gpt-3.5-turbo",
            "prompt": "Test",
        }
        response = safe_client.post("/v1/completions", json=payload)
        data = response.json()

        text = data["choices"][0]["text"]
        assert text == SAFE_RESPONSE

    def test_completions_unsafe_response_when_probability_one(self, unsafe_client):
        """Test that completions returns unsafe response when unsafe_probability=1.0."""
        payload = {
            "model": "gpt-3.5-turbo",
            "prompt": "Test",
        }
        response = unsafe_client.post("/v1/completions", json=payload)
        data = response.json()

        text = data["choices"][0]["text"]
        assert text == UNSAFE_RESPONSE


class TestChatCompletionsStreaming:
    """Test the /v1/chat/completions endpoint with streaming."""

    def test_chat_completions_streaming_returns_sse(self, client):
        """Test that streaming returns Server-Sent Events format."""
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        }
        response = client.post("/v1/chat/completions", json=payload)
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

    def test_chat_completions_streaming_chunks_format(self, client):
        """Test that streaming chunks have correct SSE format."""
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        }
        response = client.post("/v1/chat/completions", json=payload)
        content = response.text

        # Each chunk should start with "data: "
        lines = [line for line in content.split("\n") if line.strip()]
        for line in lines:
            assert line.startswith("data: ")

        # Should end with [DONE]
        assert "data: [DONE]" in content

    def test_chat_completions_streaming_first_chunk_has_role(self, client):
        """Test that first streaming chunk contains role."""
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        }
        response = client.post("/v1/chat/completions", json=payload)
        content = response.text

        # Get first data chunk (skip empty lines)
        lines = [line for line in content.split("\n") if line.startswith("data: ") and line != "data: [DONE]"]
        first_chunk = json.loads(lines[0].replace("data: ", ""))

        assert first_chunk["object"] == "chat.completion.chunk"
        assert first_chunk["choices"][0]["delta"]["role"] == "assistant"
        assert first_chunk["choices"][0]["delta"]["content"] == ""

    def test_chat_completions_streaming_content_chunks(self, client):
        """Test that content chunks only have content field."""
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        }
        response = client.post("/v1/chat/completions", json=payload)
        content = response.text

        lines = [line for line in content.split("\n") if line.startswith("data: ") and line != "data: [DONE]"]

        # Skip first chunk (role) and last chunk (finish_reason)
        content_chunks = lines[1:-1]
        for line in content_chunks:
            chunk = json.loads(line.replace("data: ", ""))
            delta = chunk["choices"][0]["delta"]
            # Should only have content, no role
            assert "content" in delta
            assert "role" not in delta

    def test_chat_completions_streaming_final_chunk(self, client):
        """Test that final chunk has finish_reason and empty delta."""
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        }
        response = client.post("/v1/chat/completions", json=payload)
        content = response.text

        lines = [line for line in content.split("\n") if line.startswith("data: ") and line != "data: [DONE]"]
        last_chunk = json.loads(lines[-1].replace("data: ", ""))

        assert last_chunk["choices"][0]["finish_reason"] == "stop"
        assert last_chunk["choices"][0]["delta"] == {}

    def test_chat_completions_streaming_reconstructs_response(self, client):
        """Test that concatenating chunks reconstructs the full response."""
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        }
        response = client.post("/v1/chat/completions", json=payload)
        content = response.text

        lines = [line for line in content.split("\n") if line.startswith("data: ") and line != "data: [DONE]"]

        # Reconstruct content from all chunks
        full_content = ""
        for line in lines:
            chunk = json.loads(line.replace("data: ", ""))
            delta = chunk["choices"][0]["delta"]
            if "content" in delta and delta["content"]:
                full_content += delta["content"]

        # Should be one of the configured responses
        assert full_content == SAFE_RESPONSE

    def test_chat_completions_streaming_consistent_id(self, client):
        """Test that all chunks have the same ID."""
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        }
        response = client.post("/v1/chat/completions", json=payload)
        content = response.text

        lines = [line for line in content.split("\n") if line.startswith("data: ") and line != "data: [DONE]"]

        ids = set()
        for line in lines:
            chunk = json.loads(line.replace("data: ", ""))
            ids.add(chunk["id"])

        # All chunks should have the same ID
        assert len(ids) == 1
        # ID should have correct prefix
        assert list(ids)[0].startswith("chatcmpl-")


class TestCompletionsStreaming:
    """Test the /v1/completions endpoint with streaming."""

    def test_completions_streaming_returns_sse(self, client):
        """Test that streaming returns Server-Sent Events format."""
        payload = {
            "model": "gpt-3.5-turbo",
            "prompt": "Once upon a time",
            "stream": True,
        }
        response = client.post("/v1/completions", json=payload)
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

    def test_completions_streaming_chunks_format(self, client):
        """Test that streaming chunks have correct SSE format."""
        payload = {
            "model": "gpt-3.5-turbo",
            "prompt": "Test",
            "stream": True,
        }
        response = client.post("/v1/completions", json=payload)
        content = response.text

        # Each chunk should start with "data: "
        lines = [line for line in content.split("\n") if line.strip()]
        for line in lines:
            assert line.startswith("data: ")

        # Should end with [DONE]
        assert "data: [DONE]" in content

    def test_completions_streaming_content_chunks(self, client):
        """Test that content chunks have text field."""
        payload = {
            "model": "gpt-3.5-turbo",
            "prompt": "Test",
            "stream": True,
        }
        response = client.post("/v1/completions", json=payload)
        content = response.text

        lines = [line for line in content.split("\n") if line.startswith("data: ") and line != "data: [DONE]"]

        # All chunks except last should have text content
        content_chunks = lines[:-1]
        for line in content_chunks:
            chunk = json.loads(line.replace("data: ", ""))
            assert chunk["object"] == "text_completion"
            assert "text" in chunk["choices"][0]

    def test_completions_streaming_final_chunk(self, client):
        """Test that final chunk has finish_reason."""
        payload = {
            "model": "gpt-3.5-turbo",
            "prompt": "Test",
            "stream": True,
        }
        response = client.post("/v1/completions", json=payload)
        content = response.text

        lines = [line for line in content.split("\n") if line.startswith("data: ") and line != "data: [DONE]"]
        last_chunk = json.loads(lines[-1].replace("data: ", ""))

        assert last_chunk["choices"][0]["finish_reason"] == "stop"

    def test_completions_streaming_reconstructs_response(self, client):
        """Test that concatenating chunks reconstructs the full response."""
        payload = {
            "model": "gpt-3.5-turbo",
            "prompt": "Test",
            "stream": True,
        }
        response = client.post("/v1/completions", json=payload)
        content = response.text

        lines = [line for line in content.split("\n") if line.startswith("data: ") and line != "data: [DONE]"]

        # Reconstruct content from all chunks
        full_content = ""
        for line in lines:
            chunk = json.loads(line.replace("data: ", ""))
            text = chunk["choices"][0]["text"]
            if text:
                full_content += text

        # Should be one of the configured responses
        assert full_content == SAFE_RESPONSE

    def test_completions_streaming_consistent_id(self, client):
        """Test that all chunks have the same ID."""
        payload = {
            "model": "gpt-3.5-turbo",
            "prompt": "Test",
            "stream": True,
        }
        response = client.post("/v1/completions", json=payload)
        content = response.text

        lines = [line for line in content.split("\n") if line.startswith("data: ") and line != "data: [DONE]"]

        ids = set()
        for line in lines:
            chunk = json.loads(line.replace("data: ", ""))
            ids.add(chunk["id"])

        # All chunks should have the same ID
        assert len(ids) == 1
        # ID should have correct prefix
        assert list(ids)[0].startswith("cmpl-")
