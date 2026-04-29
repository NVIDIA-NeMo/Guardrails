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

from unittest.mock import ANY, AsyncMock, MagicMock

import pytest

from nemoguardrails.exceptions import (
    LLMAuthenticationError,
    LLMClientError,
    LLMRateLimitError,
    LLMResponseValidationError,
)
from nemoguardrails.llm.clients.openai_chat_model import OpenAIChatModel, _is_openai_reasoning_model
from nemoguardrails.llm.clients.openai_compatible import OpenAICompatibleClient
from nemoguardrails.types import ChatMessage, LLMResponse, Role, ToolCall, ToolCallFunction
from tests.llm.clients._helpers import make_client, mock_httpx_post, stream_client


def _mock_client():
    client = MagicMock(spec=OpenAICompatibleClient)
    client.provider_name = "openai"
    client.provider_url = "https://api.openai.com/v1"
    return client


def _model(mock_client=None, model="gpt-4o", **kwargs):
    if mock_client is None:
        mock_client = _mock_client()
    return OpenAIChatModel(client=mock_client, model=model, **kwargs)


def _response(
    content="Hello",
    model="gpt-4o",
    finish_reason="stop",
    tool_calls=None,
    reasoning_content=None,
    usage=None,
    extra_fields=None,
):
    message = {"content": content, "role": "assistant"}
    if tool_calls:
        message["tool_calls"] = tool_calls
    if reasoning_content:
        message["reasoning_content"] = reasoning_content
    resp = {
        "id": "chatcmpl-123",
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
    }
    if usage:
        resp["usage"] = usage
    if extra_fields:
        resp.update(extra_fields)
    return resp


def _stream_chunks(deltas, model="gpt-4o", usage=None):
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


class TestGenerate:
    @pytest.mark.asyncio
    async def test_text(self):
        mc = _mock_client()
        mc.chat_completion = AsyncMock(
            return_value=_response(usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})
        )
        m = _model(mc)

        result = await m.generate_async("Hi")

        assert isinstance(result, LLMResponse)
        assert result.content == "Hello"
        assert result.model == "gpt-4o"
        assert result.finish_reason == "stop"
        assert result.usage.input_tokens == 10
        mc.chat_completion.assert_called_once_with("gpt-4o", ANY)

    @pytest.mark.asyncio
    async def test_tool_calls(self):
        mc = _mock_client()
        mc.chat_completion = AsyncMock(
            return_value=_response(
                content="",
                tool_calls=[
                    {
                        "id": "call_abc",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                    },
                ],
                finish_reason="tool_calls",
            )
        )
        m = _model(mc)

        result = await m.generate_async("Weather?")

        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].id == "call_abc"
        assert result.tool_calls[0].function.name == "get_weather"
        assert result.tool_calls[0].function.arguments == {"city": "Paris"}
        assert result.finish_reason == "tool_calls"

    @pytest.mark.asyncio
    async def test_reasoning(self):
        mc = _mock_client()
        mc.chat_completion = AsyncMock(return_value=_response(content="42", reasoning_content="Let me think..."))
        m = _model(mc)

        result = await m.generate_async("What?")

        assert result.content == "42"
        assert result.reasoning == "Let me think..."

    @pytest.mark.asyncio
    async def test_cached_tokens(self):
        mc = _mock_client()
        mc.chat_completion = AsyncMock(
            return_value=_response(
                usage={
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                    "total_tokens": 120,
                    "prompt_tokens_details": {"cached_tokens": 50},
                    "completion_tokens_details": {"reasoning_tokens": 5},
                }
            )
        )
        m = _model(mc)

        result = await m.generate_async("Hi")

        assert result.usage.cached_tokens == 50
        assert result.usage.reasoning_tokens == 5

    @pytest.mark.asyncio
    async def test_null_token_details(self):
        mc = _mock_client()
        mc.chat_completion = AsyncMock(
            return_value=_response(
                usage={
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                    "total_tokens": 120,
                    "prompt_tokens_details": None,
                    "completion_tokens_details": None,
                }
            )
        )
        m = _model(mc)

        result = await m.generate_async("Hi")

        assert result.usage.cached_tokens is None
        assert result.usage.reasoning_tokens is None

    @pytest.mark.asyncio
    async def test_provider_metadata(self):
        mc = _mock_client()
        mc.chat_completion = AsyncMock(
            return_value=_response(extra_fields={"system_fingerprint": "fp_abc", "service_tier": "default"})
        )
        m = _model(mc)

        result = await m.generate_async("Hi")

        assert result.provider_metadata is not None
        assert result.provider_metadata["system_fingerprint"] == "fp_abc"

    @pytest.mark.asyncio
    async def test_response_headers_in_metadata(self):
        mc = _mock_client()
        mc.chat_completion = AsyncMock(
            return_value=_response(extra_fields={"_response_headers": {"x-request-id": "req-abc"}})
        )
        m = _model(mc)

        result = await m.generate_async("Hi")

        assert result.provider_metadata["response_headers"]["x-request-id"] == "req-abc"

    @pytest.mark.asyncio
    async def test_content_alongside_tool_calls(self):
        mc = _mock_client()
        mc.chat_completion = AsyncMock(
            return_value=_response(
                content="Let me check.",
                tool_calls=[{"id": "call_1", "type": "function", "function": {"name": "fn", "arguments": "{}"}}],
                finish_reason="tool_calls",
            )
        )
        m = _model(mc)

        result = await m.generate_async("Hi")

        assert result.content == "Let me check."
        assert result.tool_calls is not None


class TestStream:
    @pytest.mark.asyncio
    async def test_text(self):
        mc = _mock_client()

        async def mock_stream(*args, **kwargs):
            for c in _stream_chunks(
                [{"content": "Hello"}, {"content": " world"}],
                usage={"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
            ):
                yield c

        mc.stream_chat_completion = mock_stream
        m = _model(mc)

        results = []
        async for chunk in m.stream_async("Hi"):
            results.append(chunk)

        assert len(results) == 3
        assert results[0].delta_content == "Hello"
        assert results[1].delta_content == " world"
        assert results[2].usage.total_tokens == 7

    @pytest.mark.asyncio
    async def test_tool_call_accumulation(self):
        mc = _mock_client()

        async def mock_stream(*args, **kwargs):
            for c in [
                {
                    "id": "c",
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
                                    }
                                ]
                            },
                            "finish_reason": None,
                        }
                    ],
                },
                {
                    "id": "c",
                    "model": "gpt-4o",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"city":"Paris"}'}}]},
                            "finish_reason": None,
                        }
                    ],
                },
                {"id": "c", "model": "gpt-4o", "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
            ]:
                yield c

        mc.stream_chat_completion = mock_stream
        m = _model(mc)

        results = []
        async for chunk in m.stream_async("Weather?"):
            results.append(chunk)

        final = results[-1]
        assert final.finish_reason == "tool_calls"
        assert len(final.delta_tool_calls) == 1
        assert final.delta_tool_calls[0].function.name == "get_weather"
        assert final.delta_tool_calls[0].function.arguments == {"city": "Paris"}

    @pytest.mark.asyncio
    async def test_parallel_tool_calls(self):
        mc = _mock_client()

        async def mock_stream(*args, **kwargs):
            for c in [
                {
                    "id": "c",
                    "model": "gpt-4o",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "c1",
                                        "type": "function",
                                        "function": {"name": "get_weather", "arguments": ""},
                                    },
                                    {
                                        "index": 1,
                                        "id": "c2",
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
                    "id": "c",
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
                {"id": "c", "model": "gpt-4o", "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
            ]:
                yield c

        mc.stream_chat_completion = mock_stream
        m = _model(mc)

        results = []
        async for chunk in m.stream_async("test"):
            results.append(chunk)

        tcs = results[-1].delta_tool_calls
        assert len(tcs) == 2
        assert tcs[0].function.name == "get_weather"
        assert tcs[1].function.name == "get_time"

    @pytest.mark.asyncio
    async def test_invalid_json_fallback(self):
        mc = _mock_client()

        async def mock_stream(*args, **kwargs):
            for c in [
                {
                    "id": "c",
                    "model": "gpt-4o",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "c1",
                                        "type": "function",
                                        "function": {"name": "fn", "arguments": ""},
                                    }
                                ]
                            },
                            "finish_reason": None,
                        }
                    ],
                },
                {
                    "id": "c",
                    "model": "gpt-4o",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"tool_calls": [{"index": 0, "function": {"arguments": "{bad json"}}]},
                            "finish_reason": None,
                        }
                    ],
                },
                {"id": "c", "model": "gpt-4o", "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
            ]:
                yield c

        mc.stream_chat_completion = mock_stream
        m = _model(mc)

        results = []
        async for chunk in m.stream_async("test"):
            results.append(chunk)

        assert results[-1].delta_tool_calls[0].function.arguments == {}


class TestParams:
    @pytest.mark.asyncio
    async def test_reasoning_model_strips_temperature(self):
        mc = _mock_client()
        mc.chat_completion = AsyncMock(return_value=_response())
        m = _model(mc, model="o3-mini")

        await m.generate_async("Hi", temperature=0.5, max_tokens=100)

        call = mc.chat_completion.call_args
        assert "temperature" not in call.kwargs
        assert call.kwargs["max_tokens"] == 100

    @pytest.mark.asyncio
    async def test_non_reasoning_keeps_temperature(self):
        mc = _mock_client()
        mc.chat_completion = AsyncMock(return_value=_response())
        m = _model(mc, model="gpt-4o")

        await m.generate_async("Hi", temperature=0.5)

        assert mc.chat_completion.call_args.kwargs["temperature"] == 0.5

    @pytest.mark.asyncio
    async def test_stop_passed_through(self):
        mc = _mock_client()
        mc.chat_completion = AsyncMock(return_value=_response())
        m = _model(mc)

        await m.generate_async("Hi", stop=["END"])

        assert mc.chat_completion.call_args.kwargs["stop"] == ["END"]

    @pytest.mark.asyncio
    async def test_reasoning_model_strips_stop(self):
        mc = _mock_client()
        mc.chat_completion = AsyncMock(return_value=_response())
        m = _model(mc, model="o3-mini")

        await m.generate_async("Hi", stop=["END"], temperature=0.5)

        call = mc.chat_completion.call_args
        assert "stop" not in call.kwargs
        assert "temperature" not in call.kwargs

    @pytest.mark.asyncio
    async def test_reasoning_model_strips_default_stop(self):
        mc = _mock_client()
        mc.chat_completion = AsyncMock(return_value=_response())
        m = _model(mc, model="o3-mini", stop=["END"])

        await m.generate_async("Hi")

        assert "stop" not in mc.chat_completion.call_args.kwargs

    @pytest.mark.asyncio
    async def test_default_kwargs_merged(self):
        mc = _mock_client()
        mc.chat_completion = AsyncMock(return_value=_response())
        m = _model(mc, temperature=0.7)

        await m.generate_async("Hi")

        assert mc.chat_completion.call_args.kwargs["temperature"] == 0.7

    @pytest.mark.asyncio
    async def test_per_call_overrides_default(self):
        mc = _mock_client()
        mc.chat_completion = AsyncMock(return_value=_response())
        m = _model(mc, temperature=0.7)

        await m.generate_async("Hi", temperature=0.9)

        assert mc.chat_completion.call_args.kwargs["temperature"] == 0.9


class TestIsReasoningModel:
    @pytest.mark.parametrize(
        "model_name",
        [
            "o1",
            "o1-mini",
            "o1-preview",
            "o1-2024-12-17",
            "o3",
            "o3-mini",
            "o3-mini-2025-01-31",
            "o4",
            "o4-mini",
            "gpt-5",
            "gpt-5-mini",
            "gpt-5-nano",
            "gpt-5.1",
            "gpt-5.4",
            "gpt-5.5",
            "gpt-6",
            "gpt-6-mini",
            "gpt-6.1",
            "O3-MINI",
            "GPT-5",
        ],
    )
    def test_reasoning_models(self, model_name):
        assert _is_openai_reasoning_model(model_name) is True

    @pytest.mark.parametrize(
        "model_name",
        [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4",
            "gpt-4-turbo",
            "gpt-3.5-turbo",
            "llama-3",
            "claude-3-opus",
            "o10-turbo",
            "o11",
            "o100",
            "o2",
            "oxford",
            "gpt-50",
            "gpt-5x",
            "gpt-500",
            "gpt-5-chat",
            "gpt-5-chat-latest",
        ],
    )
    def test_non_reasoning_models(self, model_name):
        assert _is_openai_reasoning_model(model_name) is False


class TestMessageSerialization:
    @pytest.mark.asyncio
    async def test_string_prompt(self):
        mc = _mock_client()
        mc.chat_completion = AsyncMock(return_value=_response())
        m = _model(mc)

        await m.generate_async("Hello")

        messages = mc.chat_completion.call_args[0][1]
        assert messages == [{"role": "user", "content": "Hello"}]

    @pytest.mark.asyncio
    async def test_chat_messages(self):
        mc = _mock_client()
        mc.chat_completion = AsyncMock(return_value=_response())
        m = _model(mc)

        await m.generate_async(
            [
                ChatMessage(role=Role.SYSTEM, content="Be helpful."),
                ChatMessage(role=Role.USER, content="Hello"),
            ]
        )

        messages = mc.chat_completion.call_args[0][1]
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    @pytest.mark.asyncio
    async def test_tool_call_arguments_serialized_as_json_string(self):
        mc = _mock_client()
        mc.chat_completion = AsyncMock(return_value=_response())
        m = _model(mc)

        await m.generate_async(
            [
                ChatMessage(role=Role.USER, content="Weather?"),
                ChatMessage(
                    role=Role.ASSISTANT,
                    content="",
                    tool_calls=[ToolCall(id="c1", function=ToolCallFunction(name="fn", arguments={"city": "Paris"}))],
                ),
                ChatMessage(role=Role.TOOL, content='{"temp": 22}', tool_call_id="c1"),
            ]
        )

        messages = mc.chat_completion.call_args[0][1]
        assert messages[1]["tool_calls"][0]["function"]["arguments"] == '{"city": "Paris"}'
        assert messages[2]["role"] == "tool"
        assert messages[2]["tool_call_id"] == "c1"

    @pytest.mark.asyncio
    async def test_provider_metadata_not_sent(self):
        mc = _mock_client()
        mc.chat_completion = AsyncMock(return_value=_response())
        m = _model(mc)

        msg = ChatMessage.from_dict({"role": "user", "content": "Hi", "extra_field": "x"})
        assert msg.provider_metadata == {"extra_field": "x"}

        await m.generate_async([msg])

        messages = mc.chat_completion.call_args[0][1]
        assert "provider_metadata" not in messages[0]
        assert "extra_field" not in messages[0]


class TestValidation:
    @pytest.mark.asyncio
    async def test_no_choices(self):
        mc = _mock_client()
        mc.chat_completion = AsyncMock(return_value={"id": "c", "model": "gpt-4o"})
        m = _model(mc)

        with pytest.raises(LLMResponseValidationError, match="choices"):
            await m.generate_async("Hi")

    @pytest.mark.asyncio
    async def test_empty_choices(self):
        mc = _mock_client()
        mc.chat_completion = AsyncMock(return_value={"id": "c", "model": "gpt-4o", "choices": []})
        m = _model(mc)

        with pytest.raises(LLMResponseValidationError, match="choices"):
            await m.generate_async("Hi")

    @pytest.mark.asyncio
    async def test_no_message(self):
        mc = _mock_client()
        mc.chat_completion = AsyncMock(return_value={"id": "c", "model": "gpt-4o", "choices": [{"index": 0}]})
        m = _model(mc)

        with pytest.raises(LLMResponseValidationError, match="message"):
            await m.generate_async("Hi")


class TestErrorEnrichment:
    @pytest.mark.asyncio
    async def test_model_enriches_client_errors_with_provider_and_model(self):
        mc = _mock_client()
        mc.chat_completion = AsyncMock(side_effect=LLMClientError(401, "Unauthorized"))
        m = _model(mc, model="gpt-4o")

        with pytest.raises(LLMClientError) as exc_info:
            await m.generate_async("Hi")

        assert exc_info.value.model_name == "gpt-4o"
        assert exc_info.value.provider_name == "openai"
        assert exc_info.value.base_url == "https://api.openai.com/v1"

    @pytest.mark.asyncio
    async def test_model_enriches_stream_errors(self):
        mc = _mock_client()

        async def _failing_stream(*args, **kwargs):
            raise LLMClientError(503, "overloaded")
            yield  # pragma: no cover

        mc.stream_chat_completion = _failing_stream
        m = _model(mc, model="gpt-4o")

        with pytest.raises(LLMClientError) as exc_info:
            async for _ in m.stream_async("Hi"):
                pass

        assert exc_info.value.model_name == "gpt-4o"
        assert exc_info.value.provider_name == "openai"

    @pytest.mark.asyncio
    async def test_validation_error_enriched_with_context(self):
        mc = _mock_client()
        mc.chat_completion = AsyncMock(return_value={"id": "x", "model": "gpt-4o"})
        m = _model(mc, model="gpt-4o")

        with pytest.raises(LLMResponseValidationError) as exc_info:
            await m.generate_async("Hi")

        assert exc_info.value.model_name == "gpt-4o"
        assert exc_info.value.provider_name == "openai"
        assert exc_info.value.base_url == "https://api.openai.com/v1"

    @pytest.mark.asyncio
    async def test_http_error_enriched_through_real_client_path(self):
        client = make_client()
        mock_httpx_post(client, [(401, {"error": {"message": "Invalid key"}}, {})])
        m = OpenAIChatModel(client=client, model="gpt-4o")
        with pytest.raises(LLMAuthenticationError) as exc_info:
            await m.generate_async("hi")
        assert exc_info.value.provider_name == "openai"
        assert exc_info.value.base_url == "https://api.openai.com/v1"
        assert exc_info.value.model_name == "gpt-4o"

    @pytest.mark.asyncio
    async def test_http_error_enriched_with_nim_url_lookup(self):
        client = OpenAICompatibleClient(base_url="https://integrate.api.nvidia.com/v1", api_key="nvapi-x")
        mock_httpx_post(client, [(401, {"error": {"message": "Invalid key"}}, {})])
        m = OpenAIChatModel(client=client, model="llama")
        with pytest.raises(LLMAuthenticationError) as exc_info:
            await m.generate_async("hi")
        assert exc_info.value.provider_name == "nim"
        assert exc_info.value.base_url == "https://integrate.api.nvidia.com/v1"

    @pytest.mark.asyncio
    async def test_http_error_explicit_provider_override_wins(self):
        client = OpenAICompatibleClient(base_url="http://my-internal-nim.corp/v1", api_key="x")
        mock_httpx_post(client, [(401, {"error": {"message": "Invalid key"}}, {})])
        m = OpenAIChatModel(client=client, model="llama", provider_name="nim")
        with pytest.raises(LLMAuthenticationError) as exc_info:
            await m.generate_async("hi")
        assert exc_info.value.provider_name == "nim"
        assert exc_info.value.base_url == "http://my-internal-nim.corp/v1"

    @pytest.mark.asyncio
    async def test_mid_stream_error_enriched_through_real_client(self):
        client = stream_client(
            [
                'data: {"error": {"message": "err", "type": "rate_limit_error"}}',
                "",
            ]
        )
        m = OpenAIChatModel(client=client, model="gpt-4o")
        with pytest.raises(LLMRateLimitError) as exc_info:
            async for _ in m.stream_async("hi"):
                pass
        assert exc_info.value.provider_name == "openai"
        assert exc_info.value.base_url == "https://api.openai.com/v1"
        assert exc_info.value.model_name == "gpt-4o"

    @pytest.mark.asyncio
    async def test_validation_error_status_code_is_zero(self):
        mc = _mock_client()
        mc.chat_completion = AsyncMock(return_value={"id": "x", "model": "gpt-4o"})
        m = _model(mc, model="gpt-4o")

        with pytest.raises(LLMResponseValidationError) as exc_info:
            await m.generate_async("Hi")

        assert exc_info.value.status_code == 0


class TestProperties:
    def test_model_name(self):
        m = _model(model="gpt-4o-mini")
        assert m.model_name == "gpt-4o-mini"

    def test_provider_name_url_lookup_openai(self):
        mc = _mock_client()
        mc.provider_url = "https://api.openai.com/v1"
        m = _model(mc)
        assert m.provider_name == "openai"

    def test_provider_name_url_lookup_nim(self):
        mc = _mock_client()
        mc.provider_url = "https://integrate.api.nvidia.com/v1"
        m = _model(mc)
        assert m.provider_name == "nim"

    def test_provider_name_explicit_override_wins(self):
        mc = _mock_client()
        mc.provider_url = "http://my-internal-nim.corp/v1"
        m = OpenAIChatModel(client=mc, model="gpt-4o", provider_name="nim")
        assert m.provider_name == "nim"

    def test_provider_name_unknown_url_defaults_to_openai(self):
        mc = _mock_client()
        mc.provider_url = "https://corp-gateway.example/v1"
        m = _model(mc)
        assert m.provider_name == "openai"

    def test_provider_url_delegated(self):
        mc = _mock_client()
        mc.provider_url = "https://example.com/v1"
        m = _model(mc)
        assert m.provider_url == "https://example.com/v1"
