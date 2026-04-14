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

from unittest.mock import AsyncMock

import pytest

from nemoguardrails.llm.clients.openai_compatible import OpenAICompatibleClient
from nemoguardrails.types import ChatMessage, LLMResponse, Role


def _make_completion_response(
    content="Hello",
    model="gpt-4o",
    finish_reason="stop",
    tool_calls=None,
    reasoning_content=None,
    usage=None,
):
    message = {"content": content, "role": "assistant"}
    if tool_calls:
        message["tool_calls"] = tool_calls
    if reasoning_content:
        message["reasoning_content"] = reasoning_content
    response = {
        "id": "chatcmpl-123",
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
    }
    if usage:
        response["usage"] = usage
    return response


def _make_stream_chunks(deltas, model="gpt-4o", usage=None):
    chunks = []
    for i, delta in enumerate(deltas):
        finish_reason = None
        if i == len(deltas) - 1:
            finish_reason = "stop"
        chunks.append(
            {
                "id": "chatcmpl-123",
                "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
            }
        )
    if usage:
        chunks.append({"id": "chatcmpl-123", "model": model, "choices": [], "usage": usage})
    return chunks


class TestOpenAICompatibleClient:
    def test_properties(self):
        client = OpenAICompatibleClient(model="gpt-4o", base_url="https://api.openai.com/v1", api_key="sk-test")
        assert client.model_name == "gpt-4o"
        assert client.provider_name == "openai"
        assert client.provider_url == "https://api.openai.com/v1"

    def test_provider_name_nim(self):
        client = OpenAICompatibleClient(model="llama", base_url="https://integrate.api.nvidia.com/v1")
        assert client.provider_name == "nim"

    def test_provider_name_local(self):
        client = OpenAICompatibleClient(model="llama", base_url="http://localhost:11434/v1")
        assert client.provider_name == "local"

    @pytest.mark.asyncio
    async def test_generate_basic(self):
        client = OpenAICompatibleClient(model="gpt-4o", base_url="https://api.openai.com/v1", api_key="sk-test")
        response_data = _make_completion_response(
            content="Hello there!",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )
        client._apost = AsyncMock(return_value=response_data)

        result = await client.generate_async("Hi")

        assert isinstance(result, LLMResponse)
        assert result.content == "Hello there!"
        assert result.model == "gpt-4o"
        assert result.finish_reason == "stop"
        assert result.usage.input_tokens == 10
        assert result.usage.output_tokens == 5
        assert result.usage.total_tokens == 15

    @pytest.mark.asyncio
    async def test_generate_with_tool_calls(self):
        client = OpenAICompatibleClient(model="gpt-4o", base_url="https://api.openai.com/v1", api_key="sk-test")
        response_data = _make_completion_response(
            content="",
            tool_calls=[
                {
                    "id": "call_abc",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                }
            ],
            finish_reason="tool_calls",
        )
        client._apost = AsyncMock(return_value=response_data)

        result = await client.generate_async("What's the weather?")

        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].id == "call_abc"
        assert result.tool_calls[0].function.name == "get_weather"
        assert result.tool_calls[0].function.arguments == {"city": "Paris"}
        assert result.finish_reason == "tool_calls"

    @pytest.mark.asyncio
    async def test_generate_with_reasoning(self):
        client = OpenAICompatibleClient(model="o3", base_url="https://api.openai.com/v1", api_key="sk-test")
        response_data = _make_completion_response(content="42", reasoning_content="Let me think step by step...")
        client._apost = AsyncMock(return_value=response_data)

        result = await client.generate_async("What is the answer?")

        assert result.content == "42"
        assert result.reasoning == "Let me think step by step..."

    @pytest.mark.asyncio
    async def test_generate_with_chat_messages(self):
        client = OpenAICompatibleClient(model="gpt-4o", base_url="https://api.openai.com/v1", api_key="sk-test")
        response_data = _make_completion_response(content="Hi!")
        client._apost = AsyncMock(return_value=response_data)

        messages = [
            ChatMessage(role=Role.SYSTEM, content="You are helpful."),
            ChatMessage(role=Role.USER, content="Hello"),
        ]
        result = await client.generate_async(messages)

        call_payload = client._apost.call_args[0][1]
        assert call_payload["messages"][0]["role"] == "system"
        assert call_payload["messages"][1]["role"] == "user"

    @pytest.mark.asyncio
    async def test_stream_basic(self):
        client = OpenAICompatibleClient(model="gpt-4o", base_url="https://api.openai.com/v1", api_key="sk-test")
        chunks = _make_stream_chunks(
            [{"content": "Hello"}, {"content": " there"}, {"content": "!"}],
            usage={"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        )

        async def mock_stream(*args, **kwargs):
            for chunk in chunks:
                yield chunk

        client._apost_stream = mock_stream

        results = []
        async for chunk in client.stream_async("Hi"):
            results.append(chunk)

        assert len(results) == 4
        assert results[0].delta_content == "Hello"
        assert results[1].delta_content == " there"
        assert results[2].delta_content == "!"
        assert results[2].finish_reason == "stop"
        assert results[3].usage.total_tokens == 8

    @pytest.mark.asyncio
    async def test_payload_includes_stop_and_kwargs(self):
        client = OpenAICompatibleClient(
            model="gpt-4o",
            base_url="https://api.openai.com/v1",
            api_key="sk-test",
            temperature=0.7,
        )
        response_data = _make_completion_response()
        client._apost = AsyncMock(return_value=response_data)

        await client.generate_async("Hi", stop=["END"], max_tokens=100)

        payload = client._apost.call_args[0][1]
        assert payload["model"] == "gpt-4o"
        assert payload["stop"] == ["END"]
        assert payload["temperature"] == 0.7
        assert payload["max_tokens"] == 100

    @pytest.mark.asyncio
    async def test_reasoning_model_strips_temperature(self):
        client = OpenAICompatibleClient(model="o3-mini", base_url="https://api.openai.com/v1", api_key="sk-test")
        response_data = _make_completion_response(content="Hello", model="o3-mini")
        client._apost = AsyncMock(return_value=response_data)

        await client.generate_async("Hi", temperature=0.5, max_tokens=100)

        payload = client._apost.call_args[0][1]
        assert "temperature" not in payload
        assert payload["max_tokens"] == 100

    @pytest.mark.asyncio
    async def test_non_reasoning_model_keeps_temperature(self):
        client = OpenAICompatibleClient(model="gpt-4o", base_url="https://api.openai.com/v1", api_key="sk-test")
        response_data = _make_completion_response()
        client._apost = AsyncMock(return_value=response_data)

        await client.generate_async("Hi", temperature=0.5)

        payload = client._apost.call_args[0][1]
        assert payload["temperature"] == 0.5

    @pytest.mark.asyncio
    async def test_generate_cached_tokens(self):
        client = OpenAICompatibleClient(model="gpt-4o", base_url="https://api.openai.com/v1", api_key="sk-test")
        response_data = _make_completion_response(
            usage={
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "total_tokens": 120,
                "prompt_tokens_details": {"cached_tokens": 50},
                "completion_tokens_details": {"reasoning_tokens": 5},
            }
        )
        client._apost = AsyncMock(return_value=response_data)

        result = await client.generate_async("Hi")

        assert result.usage.cached_tokens == 50
        assert result.usage.reasoning_tokens == 5

    @pytest.mark.asyncio
    async def test_generate_null_token_details(self):
        client = OpenAICompatibleClient(
            model="nvidia/nemotron", base_url="https://integrate.api.nvidia.com/v1", api_key="nvapi-test"
        )
        response_data = _make_completion_response(
            usage={
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "total_tokens": 120,
                "prompt_tokens_details": None,
                "completion_tokens_details": None,
            }
        )
        client._apost = AsyncMock(return_value=response_data)

        result = await client.generate_async("Hi")

        assert result.usage.input_tokens == 100
        assert result.usage.cached_tokens is None
        assert result.usage.reasoning_tokens is None

    @pytest.mark.asyncio
    async def test_stream_single_tool_call(self):
        client = OpenAICompatibleClient(model="gpt-4o", base_url="https://api.openai.com/v1", api_key="sk-test")
        chunks = [
            {
                "id": "chatcmpl-123",
                "model": "gpt-4o",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_abc",
                                    "type": "function",
                                    "function": {"name": "get_weather", "arguments": ""},
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "chatcmpl-123",
                "model": "gpt-4o",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"ci'}}]},
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "chatcmpl-123",
                "model": "gpt-4o",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"tool_calls": [{"index": 0, "function": {"arguments": 'ty":"Paris"}'}}]},
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "chatcmpl-123",
                "model": "gpt-4o",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
            },
            {
                "id": "chatcmpl-123",
                "model": "gpt-4o",
                "choices": [],
                "usage": {"prompt_tokens": 10, "completion_tokens": 15, "total_tokens": 25},
            },
        ]

        async def mock_stream(*args, **kwargs):
            for chunk in chunks:
                yield chunk

        client._apost_stream = mock_stream

        results = []
        async for chunk in client.stream_async("What's the weather?"):
            results.append(chunk)

        assert len(results) == 5
        assert results[0].delta_tool_calls is None
        assert results[1].delta_tool_calls is None
        assert results[2].delta_tool_calls is None

        final = results[3]
        assert final.finish_reason == "tool_calls"
        assert final.delta_tool_calls is not None
        assert len(final.delta_tool_calls) == 1
        assert final.delta_tool_calls[0].id == "call_abc"
        assert final.delta_tool_calls[0].function.name == "get_weather"
        assert final.delta_tool_calls[0].function.arguments == {"city": "Paris"}

        assert results[4].usage.total_tokens == 25

    @pytest.mark.asyncio
    async def test_stream_parallel_tool_calls(self):
        client = OpenAICompatibleClient(model="gpt-4o", base_url="https://api.openai.com/v1", api_key="sk-test")
        chunks = [
            {
                "id": "chatcmpl-123",
                "model": "gpt-4o",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "get_weather", "arguments": ""},
                                },
                                {
                                    "index": 1,
                                    "id": "call_2",
                                    "type": "function",
                                    "function": {"name": "get_time", "arguments": ""},
                                },
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "chatcmpl-123",
                "model": "gpt-4o",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "function": {"arguments": '{"city":"Paris"}'}},
                                {"index": 1, "function": {"arguments": '{"city":"Paris"}'}},
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "chatcmpl-123",
                "model": "gpt-4o",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
            },
        ]

        async def mock_stream(*args, **kwargs):
            for chunk in chunks:
                yield chunk

        client._apost_stream = mock_stream

        results = []
        async for chunk in client.stream_async("Weather and time?"):
            results.append(chunk)

        final = results[-1]
        assert final.finish_reason == "tool_calls"
        assert len(final.delta_tool_calls) == 2
        assert final.delta_tool_calls[0].function.name == "get_weather"
        assert final.delta_tool_calls[0].function.arguments == {"city": "Paris"}
        assert final.delta_tool_calls[1].function.name == "get_time"
        assert final.delta_tool_calls[1].function.arguments == {"city": "Paris"}

    @pytest.mark.asyncio
    async def test_stream_tool_call_invalid_json(self):
        client = OpenAICompatibleClient(model="gpt-4o", base_url="https://api.openai.com/v1", api_key="sk-test")
        chunks = [
            {
                "id": "chatcmpl-123",
                "model": "gpt-4o",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_x",
                                    "type": "function",
                                    "function": {"name": "broken", "arguments": ""},
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "chatcmpl-123",
                "model": "gpt-4o",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"tool_calls": [{"index": 0, "function": {"arguments": "{not valid"}}]},
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "chatcmpl-123",
                "model": "gpt-4o",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
            },
        ]

        async def mock_stream(*args, **kwargs):
            for chunk in chunks:
                yield chunk

        client._apost_stream = mock_stream

        results = []
        async for chunk in client.stream_async("test"):
            results.append(chunk)

        final = results[-1]
        assert final.delta_tool_calls is not None
        assert final.delta_tool_calls[0].function.arguments == {}


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_auth_error(self):
        from nemoguardrails.exceptions import LLMAuthenticationError

        client = OpenAICompatibleClient(model="gpt-4o", base_url="https://api.openai.com/v1", api_key="sk-test")

        async def mock_post(*args, **kwargs):
            import httpx

            return httpx.Response(
                401,
                json={
                    "error": {"message": "Invalid API key", "type": "invalid_request_error", "code": "invalid_api_key"}
                },
                request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
            )

        client._get_client = AsyncMock()
        client._get_client.return_value.post = mock_post

        with pytest.raises(LLMAuthenticationError) as exc_info:
            await client.generate_async("Hi")
        assert exc_info.value.status_code == 401
        assert "Invalid API key" in exc_info.value.error_message

    @pytest.mark.asyncio
    async def test_bad_request_error(self):
        from nemoguardrails.exceptions import LLMBadRequestError

        client = OpenAICompatibleClient(model="gpt-4o", base_url="https://api.openai.com/v1", api_key="sk-test")

        async def mock_post(*args, **kwargs):
            import httpx

            return httpx.Response(
                400,
                json={"error": {"message": "Invalid temperature value", "type": "invalid_request_error"}},
                request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
            )

        client._get_client = AsyncMock()
        client._get_client.return_value.post = mock_post

        with pytest.raises(LLMBadRequestError) as exc_info:
            await client.generate_async("Hi")
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_context_window_error(self):
        from nemoguardrails.exceptions import LLMContextWindowError

        client = OpenAICompatibleClient(model="gpt-4o", base_url="https://api.openai.com/v1", api_key="sk-test")

        async def mock_post(*args, **kwargs):
            import httpx

            return httpx.Response(
                400,
                json={"error": {"message": "This model's maximum context length is 8192 tokens"}},
                request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
            )

        client._get_client = AsyncMock()
        client._get_client.return_value.post = mock_post

        with pytest.raises(LLMContextWindowError):
            await client.generate_async("Hi")

    @pytest.mark.asyncio
    async def test_error_redacts_api_key(self):
        from nemoguardrails.exceptions import LLMAuthenticationError

        client = OpenAICompatibleClient(model="gpt-4o", base_url="https://api.openai.com/v1", api_key="sk-test")

        async def mock_post(*args, **kwargs):
            import httpx

            return httpx.Response(
                401,
                json={"error": {"message": "Incorrect API key provided: sk-proj-abc123def456"}},
                request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
            )

        client._get_client = AsyncMock()
        client._get_client.return_value.post = mock_post

        with pytest.raises(LLMAuthenticationError) as exc_info:
            await client.generate_async("Hi")
        assert "sk-proj-abc123def456" not in exc_info.value.error_message
        assert "sk-***" in exc_info.value.error_message


class TestStreamLlmCallAccumulation:
    @pytest.mark.asyncio
    async def test_stream_llm_call_accumulates_tool_calls(self):
        from nemoguardrails.actions.llm.utils import _stream_llm_call
        from nemoguardrails.streaming import StreamingHandler

        client = OpenAICompatibleClient(model="gpt-4o", base_url="https://api.openai.com/v1", api_key="sk-test")
        chunks = [
            {
                "id": "chatcmpl-123",
                "model": "gpt-4o",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_abc",
                                    "type": "function",
                                    "function": {"name": "get_weather", "arguments": ""},
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "chatcmpl-123",
                "model": "gpt-4o",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"city":"Paris"}'}}]},
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "chatcmpl-123",
                "model": "gpt-4o",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
            },
            {
                "id": "chatcmpl-123",
                "model": "gpt-4o",
                "choices": [],
                "usage": {"prompt_tokens": 10, "completion_tokens": 15, "total_tokens": 25},
            },
        ]

        async def mock_stream(*args, **kwargs):
            for chunk in chunks:
                yield chunk

        client._apost_stream = mock_stream
        handler = StreamingHandler()

        result = await _stream_llm_call(client, "What's the weather?", handler, stop=None)

        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "get_weather"
        assert result.tool_calls[0].function.arguments == {"city": "Paris"}
        assert result.model == "gpt-4o"
        assert result.finish_reason == "tool_calls"
        assert result.usage is not None
        assert result.usage.total_tokens == 25

    @pytest.mark.asyncio
    async def test_stream_llm_call_accumulates_reasoning(self):
        from nemoguardrails.actions.llm.utils import _stream_llm_call
        from nemoguardrails.streaming import StreamingHandler

        client = OpenAICompatibleClient(model="o3", base_url="https://api.openai.com/v1", api_key="sk-test")
        chunks = _make_stream_chunks(
            [
                {"reasoning_content": "Let me "},
                {"reasoning_content": "think..."},
                {"content": "42"},
            ],
            usage={"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15},
        )

        async def mock_stream(*args, **kwargs):
            for chunk in chunks:
                yield chunk

        client._apost_stream = mock_stream
        handler = StreamingHandler()

        result = await _stream_llm_call(client, "What is the answer?", handler, stop=None)

        assert result.content == "42"
        assert result.reasoning == "Let me think..."
        assert result.model == "gpt-4o"
        assert result.finish_reason == "stop"
        assert result.usage.total_tokens == 15

    @pytest.mark.asyncio
    async def test_stream_llm_call_text_only(self):
        from nemoguardrails.actions.llm.utils import _stream_llm_call
        from nemoguardrails.streaming import StreamingHandler

        client = OpenAICompatibleClient(model="gpt-4o", base_url="https://api.openai.com/v1", api_key="sk-test")
        chunks = _make_stream_chunks(
            [{"content": "Hello"}, {"content": " world"}],
            usage={"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        )

        async def mock_stream(*args, **kwargs):
            for chunk in chunks:
                yield chunk

        client._apost_stream = mock_stream
        handler = StreamingHandler()

        result = await _stream_llm_call(client, "Say hello", handler, stop=None)

        assert result.content == "Hello world"
        assert result.tool_calls is None
        assert result.reasoning is None
        assert result.model == "gpt-4o"
        assert result.finish_reason == "stop"
        assert result.usage.total_tokens == 7


class TestDefaultFramework:
    def test_create_model_openai(self):
        from nemoguardrails.llm.default_framework import DefaultFramework

        fw = DefaultFramework()
        model = fw.create_model("gpt-4o", "openai", {"api_key": "sk-test"})

        assert isinstance(model, OpenAICompatibleClient)
        assert model.model_name == "gpt-4o"
        assert model.provider_url == "https://api.openai.com/v1"

    def test_create_model_nim(self):
        from nemoguardrails.llm.default_framework import DefaultFramework

        fw = DefaultFramework()
        model = fw.create_model("llama", "nim", {"api_key": "nvapi-test"})

        assert isinstance(model, OpenAICompatibleClient)
        assert model.provider_url == "https://integrate.api.nvidia.com/v1"

    def test_create_model_custom_base_url(self):
        from nemoguardrails.llm.default_framework import DefaultFramework

        fw = DefaultFramework()
        model = fw.create_model("my-model", "custom", {"base_url": "https://my.api.com/v1"})

        assert model.provider_url == "https://my.api.com/v1"

    def test_create_model_unknown_provider_no_base_url_raises(self):
        from nemoguardrails.llm.default_framework import DefaultFramework

        fw = DefaultFramework()
        with pytest.raises(ValueError, match="No default base_url"):
            fw.create_model("model", "unknown_provider", {})

    def test_get_provider_names(self):
        from nemoguardrails.llm.default_framework import DefaultFramework

        fw = DefaultFramework()
        names = fw.get_provider_names()
        assert "openai" in names
        assert "nim" in names
        assert "ollama" in names

    def test_framework_registered_in_lazy_frameworks(self):
        from nemoguardrails.llm.frameworks import _LAZY_FRAMEWORKS

        assert "default" in _LAZY_FRAMEWORKS

    def test_framework_lazy_loading(self):
        from nemoguardrails.llm.default_framework import DefaultFramework
        from nemoguardrails.llm.frameworks import _reset_frameworks, get_framework

        _reset_frameworks()
        fw = get_framework("default")
        assert isinstance(fw, DefaultFramework)
        _reset_frameworks()
