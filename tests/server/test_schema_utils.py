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

import pytest

pytest.importorskip("openai", reason="openai is required for server tests")
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage

from nemoguardrails.rails.llm.options import GenerationLog, GenerationResponse
from nemoguardrails.server.schemas.openai import GuardrailsChatCompletion
from nemoguardrails.server.schemas.utils import (
    create_error_chat_completion,
    extract_bot_message_from_response,
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
