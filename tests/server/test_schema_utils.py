# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from unittest.mock import AsyncMock, patch

import httpx
import pytest

pytest.importorskip("openai", reason="openai is required for server tests")
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage

from nemoguardrails.rails.llm.options import GenerationLog, GenerationResponse
from nemoguardrails.server.schemas.openai import (
    GuardrailsChatCompletion,
    OpenAIModel,
    OpenAIModelsList,
)
from nemoguardrails.server.schemas.utils import (
    PROVIDERS,
    _azure_url,
    _openai_compatible_url,
    create_error_chat_completion,
    extract_bot_message_from_response,
    fetch_models,
    format_streaming_chunk,
    format_streaming_chunk_as_sse,
    generation_response_to_chat_completion,
)

# ===== Tests for extract_bot_message_from_response =====


def test_extract_bot_message_from_string_response():
    """Test extracting bot message from a plain string response."""
    response = "Hello, how can I help you?"
    result = extract_bot_message_from_response(response)
    assert result == {"role": "assistant", "content": "Hello, how can I help you?"}


def test_extract_bot_message_from_dict_response():
    """Test extracting bot message from a dict response."""
    response = {"role": "assistant", "content": "Test response"}
    result = extract_bot_message_from_response(response)
    assert result == {"role": "assistant", "content": "Test response"}


def test_extract_bot_message_from_generation_response_with_string_content():
    """Test extracting bot message from GenerationResponse with string in list."""
    response = GenerationResponse(response=[{"role": "assistant", "content": "Hello from bot"}])
    result = extract_bot_message_from_response(response)
    assert result == {"role": "assistant", "content": "Hello from bot"}


def test_extract_bot_message_from_generation_response_with_dict():
    """Test extracting bot message from GenerationResponse containing a dict."""
    bot_msg = {"role": "assistant", "content": "Response from dict"}
    response = GenerationResponse(response=[bot_msg])
    result = extract_bot_message_from_response(response)
    assert result == {"role": "assistant", "content": "Response from dict"}


def test_extract_bot_message_from_tuple_with_dict():
    """Test extracting bot message from a tuple (message, state) with dict message."""
    response = ({"role": "assistant", "content": "Tuple response"}, {"state": "data"})
    result = extract_bot_message_from_response(response)
    assert result == {"role": "assistant", "content": "Tuple response"}


# ===== Tests for generation_response_to_chat_completion =====


def test_generation_response_to_chat_completion():
    """Test converting a full GenerationResponse to chat completion."""
    response = GenerationResponse(
        response=[{"role": "assistant", "content": "This is a response"}],
        llm_output={"llm_output": "This is an LLM output"},
        output_data={"output_data": "This is output data"},
        log=GenerationLog(),
        state={"state": "This is a state"},
    )
    result = generation_response_to_chat_completion(response=response, model="test_model", config_id="test_config_id")
    assert isinstance(result, GuardrailsChatCompletion)
    assert result.id.startswith("chatcmpl-")
    assert isinstance(result.created, int)

    assert result.object == "chat.completion"
    assert result.model == "test_model"
    assert result.guardrails is not None
    assert result.guardrails.config_id == "test_config_id"
    assert result.choices[0] == Choice(
        index=0,
        message=ChatCompletionMessage(role="assistant", content="This is a response"),
        finish_reason="stop",
        logprobs=None,
    )
    assert result.guardrails.llm_output == {"llm_output": "This is an LLM output"}
    assert result.guardrails.output_data == {"output_data": "This is output data"}
    assert result.guardrails.log is not None
    assert result.guardrails.state == {"state": "This is a state"}


def test_generation_response_to_chat_completion_with_empty_content():
    """Test converting GenerationResponse with missing content."""
    response = GenerationResponse(response=[{"role": "assistant", "content": ""}])
    result = generation_response_to_chat_completion(response=response, model="test_model")
    assert result.choices[0].message.content == ""


# ===== Tests for create_error_chat_completion =====


def test_create_error_chat_completion():
    """Test creating an error chat completion response."""
    error_message = "This is an error message"
    config_id = "test_config_id"
    result = create_error_chat_completion(model="test_model", error_message=error_message, config_id=config_id)
    assert result.choices[0].message.content == error_message
    assert result.model == "test_model"
    assert result.guardrails is not None
    assert result.guardrails.config_id == config_id
    assert result.object == "chat.completion"
    assert result.choices[0].message.role == "assistant"
    assert result.choices[0].finish_reason == "stop"


def test_create_error_chat_completion_without_config_id():
    """Test creating an error chat completion without config_id."""
    result = create_error_chat_completion(model="gpt-4", error_message="Error occurred")
    assert result.choices[0].message.content == "Error occurred"
    assert result.model == "gpt-4"
    assert result.guardrails is None


# ===== Tests for format_streaming_chunk =====


def test_format_streaming_chunk_with_dict():
    """Test formatting a dict chunk."""
    chunk = {"content": "Hello"}
    result = format_streaming_chunk(chunk, model="test_model")
    assert result["object"] == "chat.completion.chunk"
    assert result["model"] == "test_model"
    assert result["choices"][0]["delta"] == {"content": "Hello"}
    assert result["choices"][0]["index"] == 0
    assert result["choices"][0]["finish_reason"] is None
    assert "id" in result
    assert "created" in result


def test_format_streaming_chunk_with_plain_string():
    """Test formatting a plain string chunk."""
    chunk = "Hello world"
    result = format_streaming_chunk(chunk, model="test_model")
    assert result["object"] == "chat.completion.chunk"
    assert result["model"] == "test_model"
    assert result["choices"][0]["delta"]["content"] == "Hello world"
    assert result["choices"][0]["index"] == 0
    assert result["choices"][0]["finish_reason"] is None


def test_format_streaming_chunk_with_json_string():
    """Test formatting a JSON string chunk."""
    chunk_data = {"custom": "data", "value": 123}
    chunk = json.dumps(chunk_data)
    result = format_streaming_chunk(chunk, model="test_model", chunk_id="test-id")
    assert result["id"] == "test-id"
    assert result["model"] == "test_model"
    # Should parse the JSON and add missing fields
    assert result["custom"] == "data"
    assert result["value"] == 123


def test_format_streaming_chunk_with_none():
    """Test formatting a None chunk."""
    chunk = None
    result = format_streaming_chunk(chunk, model="test_model")
    assert result["choices"][0]["delta"]["content"] == "None"


# ===== Tests for format_streaming_chunk_as_sse =====


def test_format_streaming_chunk_as_sse_with_string():
    """Test formatting a string chunk as SSE."""
    chunk = "Hello SSE"
    result = format_streaming_chunk_as_sse(chunk, model="test_model")

    assert result.startswith("data: ")
    assert result.endswith("\n\n")
    json_str = result[6:-2]  # Remove "data: " and "\n\n"
    payload = json.loads(json_str)
    assert payload["object"] == "chat.completion.chunk"
    assert payload["model"] == "test_model"
    assert payload["choices"][0]["delta"]["content"] == "Hello SSE"


def test_format_streaming_chunk_as_sse_with_dict():
    """Test formatting a dict chunk as SSE."""
    chunk = {"role": "assistant", "content": "SSE response"}
    result = format_streaming_chunk_as_sse(chunk, model="test_model")
    assert result.startswith("data: ")
    assert result.endswith("\n\n")
    json_str = result[6:-2]
    payload = json.loads(json_str)
    assert payload["choices"][0]["delta"] == {
        "role": "assistant",
        "content": "SSE response",
    }


def test_format_streaming_chunk_as_sse_with_none():
    """Test creating the streaming done event."""
    result = format_streaming_chunk_as_sse(None, model="test_model")
    json_str = result[6:-2]
    payload = json.loads(json_str)
    assert payload["choices"][0]["delta"] == {
        "content": "None",
    }


def test_format_streaming_chunk_as_sse_with_empty_string():
    """Test creating the streaming done event."""
    result = format_streaming_chunk_as_sse("", model="test_model")
    json_str = result[6:-2]
    payload = json.loads(json_str)
    assert payload["choices"][0]["delta"] == {
        "content": "",
    }


# ===== Tests for _openai_compatible_url =====


def test_openai_url_basic():
    """Test building URL when base_url does not end with /v1."""
    with patch.dict(os.environ, {"MAIN_MODEL_BASE_URL": "http://localhost:8000"}):
        assert _openai_compatible_url() == "http://localhost:8000/v1/models"


def test_openai_url_already_has_v1():
    """Test building URL when base_url already ends with /v1."""
    with patch.dict(os.environ, {"MAIN_MODEL_BASE_URL": "http://localhost:8000/v1"}):
        assert _openai_compatible_url() == "http://localhost:8000/v1/models"


def test_openai_url_strips_trailing_slash():
    """Test that trailing slashes are stripped from base_url."""
    with patch.dict(os.environ, {"MAIN_MODEL_BASE_URL": "http://localhost:8000/"}):
        assert _openai_compatible_url() == "http://localhost:8000/v1/models"


def test_openai_url_v1_trailing_slash():
    """Test URL with /v1/ (trailing slash after v1)."""
    with patch.dict(os.environ, {"MAIN_MODEL_BASE_URL": "http://localhost:8000/v1/"}):
        assert _openai_compatible_url() == "http://localhost:8000/v1/models"


def test_openai_url_missing():
    """Test that missing MAIN_MODEL_BASE_URL raises ValueError."""
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("MAIN_MODEL_BASE_URL", None)
        with pytest.raises(ValueError, match="MAIN_MODEL_BASE_URL"):
            _openai_compatible_url()


def test_azure_url():
    """Test building Azure OpenAI URL from env vars."""
    with patch.dict(
        os.environ,
        {
            "AZURE_OPENAI_ENDPOINT": "https://myresource.openai.azure.com",
        },
    ):
        result = _azure_url()
        assert "myresource.openai.azure.com/openai/models" in result
        assert "api-version=2024-06-01" in result


# ===== Tests for _azure_url =====


def test_azure_url_custom_version():
    """Test Azure URL with custom api-version."""
    with patch.dict(
        os.environ,
        {
            "AZURE_OPENAI_ENDPOINT": "https://res.openai.azure.com",
            "AZURE_OPENAI_API_VERSION": "2025-01-01",
        },
    ):
        assert "api-version=2025-01-01" in _azure_url()


def test_azure_url_missing_endpoint():
    """Test that missing AZURE_OPENAI_ENDPOINT raises ValueError."""
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("AZURE_OPENAI_ENDPOINT", None)
        with pytest.raises(ValueError, match="AZURE_OPENAI_ENDPOINT"):
            _azure_url()


# ===== Tests for PROVIDERS table =====


def test_known_providers_exist():
    for engine in ("openai", "vllm", "nim", "trt_llm", "anthropic", "azure", "cohere"):
        assert engine in PROVIDERS


def test_azure_openai_alias():
    assert PROVIDERS["azure_openai"] is PROVIDERS["azure"]


# ===== Tests for fetch_models =====


def _mock_httpx(json_data, status_code=200):
    """Return a mock httpx.AsyncClient context manager."""
    response = httpx.Response(
        status_code=status_code,
        json=json_data,
        request=httpx.Request("GET", "http://test/v1/models"),
    )
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    return mock_client


@pytest.mark.asyncio
async def test_fetch_openai():
    """Test fetching models from an OpenAI-compatible endpoint."""
    upstream = {
        "data": [
            {
                "id": "gpt-4o",
                "object": "model",
                "created": 1700000000,
                "owned_by": "openai",
            },
        ]
    }
    mock = _mock_httpx(upstream)
    with patch.dict(
        os.environ,
        {"MAIN_MODEL_BASE_URL": "http://localhost:8000", "MAIN_MODEL_ENGINE": "openai"},
    ):
        with patch("httpx.AsyncClient", return_value=mock):
            models = await fetch_models("openai", {})
    assert len(models) == 1
    assert models[0].id == "gpt-4o"
    assert models[0].owned_by == "openai"


@pytest.mark.asyncio
async def test_fetch_anthropic():
    """Test fetching models from Anthropic."""
    upstream = {
        "data": [
            {"id": "claude-sonnet-4-20250514", "created_at": "2025-05-14T00:00:00Z"},
        ]
    }
    mock = _mock_httpx(upstream)
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}):
        with patch("httpx.AsyncClient", return_value=mock):
            models = await fetch_models("anthropic", {})
    assert models[0].id == "claude-sonnet-4-20250514"
    assert models[0].owned_by == "anthropic"
    call_headers = mock.get.call_args.kwargs["headers"]
    assert call_headers["x-api-key"] == "sk-ant-test"
    assert call_headers["anthropic-version"] == "2023-06-01"


@pytest.mark.asyncio
async def test_fetch_cohere():
    """Test fetching models from Cohere."""
    upstream = {
        "models": [
            {"name": "command-r-plus", "description": "..."},
            {"name": "command-r", "description": "..."},
        ]
    }
    mock = _mock_httpx(upstream)
    with patch.dict(os.environ, {"COHERE_API_KEY": "co-key"}):
        with patch("httpx.AsyncClient", return_value=mock):
            models = await fetch_models("cohere", {})
    assert len(models) == 2
    assert models[0].id == "command-r-plus"
    assert models[1].id == "command-r"
    assert models[0].owned_by == "cohere"


@pytest.mark.asyncio
async def test_fetch_azure():
    """Test fetching models from Azure OpenAI."""
    upstream = {"data": [{"id": "gpt-4", "created_at": 1700000000}]}
    mock = _mock_httpx(upstream)
    with patch.dict(
        os.environ,
        {
            "AZURE_OPENAI_ENDPOINT": "https://res.openai.azure.com",
            "AZURE_OPENAI_API_KEY": "az-key",
        },
    ):
        with patch("httpx.AsyncClient", return_value=mock):
            models = await fetch_models("azure", {})
    assert models[0].id == "gpt-4"
    assert models[0].owned_by == "azure"
    call_headers = mock.get.call_args.kwargs["headers"]
    assert call_headers["api-key"] == "az-key"


@pytest.mark.asyncio
async def test_fetch_unknown_engine_with_base_url():
    """Unknown engines fall back to OpenAI-compatible when MAIN_MODEL_BASE_URL is set."""
    mock = _mock_httpx({"data": [{"id": "custom-model"}]})
    with patch.dict(os.environ, {"MAIN_MODEL_BASE_URL": "http://localhost:5000"}):
        with patch("httpx.AsyncClient", return_value=mock):
            models = await fetch_models("my_custom", {})
    assert len(models) == 1
    assert models[0].id == "custom-model"


@pytest.mark.asyncio
async def test_fetch_unknown_engine_no_base_url():
    """Unknown engines return empty list when no MAIN_MODEL_BASE_URL."""
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("MAIN_MODEL_BASE_URL", None)
        models = await fetch_models("niche_llm", {})
    assert models == []


@pytest.mark.asyncio
async def test_fetch_auth_forwarded():
    """Incoming Authorization header is forwarded for OpenAI-compatible providers."""
    mock = _mock_httpx({"data": []})
    with patch.dict(os.environ, {"MAIN_MODEL_BASE_URL": "http://localhost:8000"}):
        with patch("httpx.AsyncClient", return_value=mock):
            await fetch_models("openai", {"Authorization": "Bearer user-token"})
    call_headers = mock.get.call_args.kwargs["headers"]
    assert call_headers["Authorization"] == "Bearer user-token"


@pytest.mark.asyncio
async def test_fetch_non_dict_items_skipped():
    """Non-dict items in the response data are skipped."""
    upstream = {"data": [{"id": "good"}, "bad", 42, None]}
    mock = _mock_httpx(upstream)
    with patch.dict(os.environ, {"MAIN_MODEL_BASE_URL": "http://localhost:8000"}):
        with patch("httpx.AsyncClient", return_value=mock):
            models = await fetch_models("openai", {})
    assert len(models) == 1
    assert models[0].id == "good"


# ===== Tests for OpenAIModel and OpenAIModelsList schemas =====


def test_openai_model_schema():
    """Test OpenAIModel schema creation."""
    model = OpenAIModel(
        id="llama-3.1-8b",
        object="model",
        created=1700000000,
        owned_by="system",
    )
    assert model.id == "llama-3.1-8b"
    assert model.object == "model"
    assert model.created == 1700000000
    assert model.owned_by == "system"


def test_openai_models_list_schema():
    """Test OpenAIModelsList schema with multiple models."""
    models = OpenAIModelsList(
        data=[
            OpenAIModel(id="test-model-1", object="model", created=1700000000, owned_by="org-a"),
            OpenAIModel(id="test-model-2", object="model", created=1700000001, owned_by="org-b"),
        ]
    )
    assert len(models.data) == 2
    assert models.data[0].id == "test-model-1"
    assert models.data[1].id == "test-model-2"


def test_openai_models_list_schema_empty():
    """Test OpenAIModelsList schema with empty list."""
    models = OpenAIModelsList(data=[])
    assert len(models.data) == 0


def test_openai_models_list_serialization():
    """Test OpenAIModelsList serializes to expected JSON structure."""
    models = OpenAIModelsList(
        data=[
            OpenAIModel(id="test-model", object="model", created=1700000000, owned_by="system"),
        ]
    )
    result = models.model_dump()
    assert result == {
        "data": [
            {
                "id": "test-model",
                "object": "model",
                "created": 1700000000,
                "owned_by": "system",
            }
        ]
    }
