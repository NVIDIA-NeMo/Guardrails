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

"""HTTP tool streams reuse generation and rail checks before releasing SSE."""

import asyncio
import json
from unittest.mock import AsyncMock

import httpx
import pytest
import pytest_asyncio
from openai import AsyncOpenAI
from openai.lib.streaming.chat import ChatCompletionStreamState
from openai.types.chat import ChatCompletionChunk

from nemoguardrails import Guardrails, RailsConfig
from nemoguardrails.rails import LLMRails
from nemoguardrails.rails.llm.options import GenerationResponse
from nemoguardrails.server import api
from nemoguardrails.server.datastore.memory_store import MemoryStore
from nemoguardrails.testing.fake_model import FakeLLMModel
from nemoguardrails.types import LLMResponse, ToolCall, ToolCallFunction

pytestmark = pytest.mark.asyncio

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
        },
    }
]
REQUEST = {
    "model": "test-model",
    "messages": [{"role": "user", "content": "Weather?"}],
    "tools": TOOLS,
    "stream": True,
    "guardrails": {"config_id": "tools"},
}
CALLS = [
    ToolCall(id=f"call_{city}", function=ToolCallFunction(name="weather", arguments={"city": city}))
    for city in ("Paris", "Boston")
]


def events(response):
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/event-stream")
    data = [line[6:] for line in response.text.splitlines() if line.startswith("data: ")]
    assert data[-1] == "[DONE]"
    assert data.count("[DONE]") == 1
    return [json.loads(line) for line in data[:-1]]


def assembled_message(frames):
    state = ChatCompletionStreamState()
    for frame in frames:
        state.handle_chunk(ChatCompletionChunk.model_validate(frame))
    message = state.get_final_completion().choices[0].message.model_dump(exclude_none=True)
    for call in message.get("tool_calls", []):
        call.pop("index", None)
    return message


@pytest_asyncio.fixture
async def client():
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url="http://testserver") as client:
        yield client


@pytest_asyncio.fixture(params=["llmrails", "llmrails-nonpassthrough", "iorails"])
async def engine(request, monkeypatch):
    model = FakeLLMModel(
        llm_responses=[
            LLMResponse(content="Checking both cities.", tool_calls=CALLS),
            LLMResponse(content="Both cities are sunny."),
        ]
    )
    if request.param.startswith("llmrails"):
        rails = LLMRails(
            RailsConfig.from_content(config={"models": [], "passthrough": request.param == "llmrails"}), llm=model
        )
    else:
        config = RailsConfig.from_content(
            config={
                "models": [
                    {"type": "main", "engine": "openai", "model": "test-model", "parameters": {"api_key": "test-key"}}
                ],
                "rails": {
                    "tool_call": {"flows": ["tool call validation"]},
                    "tool_result": {"flows": ["tool result validation"]},
                },
            }
        )
        rails = Guardrails(config, require_iorails=True)
        registry = rails.rails_engine.engine_registry
        monkeypatch.setattr(registry, "start", AsyncMock())

        async def model_call(model_type, messages, **kwargs):
            return await model.generate_async(messages, **kwargs)

        monkeypatch.setattr(registry, "model_call", model_call)
    monkeypatch.setattr(api, "_get_rails", AsyncMock(return_value=rails))
    monkeypatch.setattr(
        rails, "stream_async", lambda **kwargs: pytest.fail("Tool requests must use checked generation")
    )
    yield rails, model
    if request.param == "iorails":
        await rails.rails_engine.stop()


async def test_tool_stream_round_trip_with_openai_client(engine, client, monkeypatch):
    store = MemoryStore()
    monkeypatch.setattr(api, "datastore", store)
    thread = "streaming-tool-thread"
    async with AsyncOpenAI(api_key="test-key", base_url="http://testserver/v1", http_client=client) as sdk:
        stream = await sdk.chat.completions.create(
            model="test-model",
            messages=REQUEST["messages"],
            tools=TOOLS,
            tool_choice="required",
            parallel_tool_calls=True,
            stream=True,
            extra_body={"guardrails": {"config_id": "tools", "thread_id": thread}},
        )
        chunks = [chunk async for chunk in stream]
        assert len({chunk.id for chunk in chunks}) == 1
        assert len({chunk.created for chunk in chunks}) == 1
        assert all(chunk.model == "test-model" for chunk in chunks)
        expected_text = "Checking both cities." if isinstance(engine[0], Guardrails) else ""
        assert "".join(c.choices[0].delta.content or "" for c in chunks) == expected_text
        headers = [call for c in chunks for call in c.choices[0].delta.tool_calls or [] if call.id]
        calls = assembled_message([c.model_dump(exclude_none=True) for c in chunks])["tool_calls"]
        assert [call.index for call in headers] == [0, 1]
        assert [call["id"] for call in calls] == [call.id for call in CALLS]
        assert [json.loads(call["function"]["arguments"]) for call in calls] == [{"city": "Paris"}, {"city": "Boston"}]
        assert chunks[-1].choices[0].finish_reason == "tool_calls"
        stored = json.loads(await store.get("thread-" + thread))
        assert [call["id"] for call in stored[-1]["tool_calls"]] == [call.id for call in CALLS]
        results = [{"role": "tool", "tool_call_id": call.id, "content": "Sunny"} for call in CALLS]
        followup = await sdk.chat.completions.create(
            model="test-model",
            messages=results,
            tools=TOOLS,
            stream=True,
            extra_body={"guardrails": {"config_id": "tools", "thread_id": thread}},
        )
        final = [chunk async for chunk in followup]
        assert "".join(c.choices[0].delta.content or "" for c in final) == "Both cities are sunny."
        assert final[-1].choices[0].finish_reason == "stop"
        assert not any(c.choices[0].delta.tool_calls for c in final)


@pytest.mark.parametrize("params", [{"tool_choice": "auto"}, {"parallel_tool_calls": False}, {"tools": TOOLS}])
async def test_each_tool_parameter_uses_buffered_generation(client, monkeypatch, params):
    rails = AsyncMock()
    rails.config = RailsConfig.from_content(config={"models": []})
    rails.generate_async.return_value = GenerationResponse(
        response=[{"role": "assistant", "content": "No tool needed."}]
    )
    monkeypatch.setattr(api, "_get_rails", AsyncMock(return_value=rails))
    body = {k: v for k, v in REQUEST.items() if k != "tools"}
    frames = events(await client.post("/v1/chat/completions", json={**body, **params}))
    assert assembled_message(frames)["content"] == "No tool needed."
    assert frames[-1]["choices"][0]["finish_reason"] == "stop"
    rails.stream_async.assert_not_called()
    for key, value in params.items():
        assert rails.generate_async.call_args.kwargs["options"].llm_params[key] == value


@pytest.mark.parametrize("passthrough", [False, True])
async def test_llmrails_blocks_calls_before_sse(client, monkeypatch, passthrough):
    config = RailsConfig.from_content(
        config={"models": [], "passthrough": passthrough, "rails": {"tool_call": {"flows": ["block calls"]}}},
        colang_content="""
define flow block calls
  bot refuse call
  abort

define bot refuse call
  "Tool call rejected."
""",
    )
    rails = LLMRails(config, llm=FakeLLMModel(llm_responses=[LLMResponse(content="", tool_calls=CALLS)]))
    monkeypatch.setattr(api, "_get_rails", AsyncMock(return_value=rails))
    frames = events(await client.post("/v1/chat/completions", json=REQUEST))
    assert assembled_message(frames)["content"] == "Tool call rejected."
    assert frames[-1]["choices"][0]["finish_reason"] == "stop"
    assert "call_Paris" not in json.dumps(frames)
    assert "tool_calls" not in json.dumps(frames)


@pytest.mark.parametrize("engine", ["iorails"], indirect=True)
async def test_iorails_blocks_undeclared_calls(engine, client):
    rails, model = engine
    frames = events(
        await client.post(
            "/v1/chat/completions",
            json={
                **REQUEST,
                "tools": [
                    {"type": "function", "function": {"name": "different_tool", "parameters": {"type": "object"}}}
                ],
            },
        )
    )
    assert frames[-1]["choices"][0]["finish_reason"] == "stop"
    assert "call_Paris" not in json.dumps(frames)
    assert "tool_calls" not in json.dumps(frames)
    assert model.inference_count == 1


@pytest.mark.parametrize("engine", ["iorails"], indirect=True)
async def test_iorails_blocks_unlinked_results_before_generation(engine, client):
    rails, model = engine
    frames = events(
        await client.post(
            "/v1/chat/completions",
            json={
                **REQUEST,
                "messages": [*REQUEST["messages"], {"role": "tool", "tool_call_id": "unknown", "content": "Sunny"}],
            },
        )
    )
    assert frames[-1]["choices"][0]["finish_reason"] == "stop"
    assert model.inference_count == 0


async def test_no_sse_before_generation_finishes_and_cancellation_propagates(monkeypatch):
    from starlette.requests import Request

    from nemoguardrails.server.schemas.openai import GuardrailsChatCompletionRequest

    entered, cancelled = asyncio.Event(), asyncio.Event()

    async def generate(**kwargs):
        entered.set()
        try:
            await asyncio.Future()
        finally:
            cancelled.set()

    rails = AsyncMock()
    rails.config = RailsConfig.from_content(config={"models": []})
    rails.generate_async = generate
    monkeypatch.setattr(api, "_get_rails", AsyncMock(return_value=rails))
    task = asyncio.create_task(
        api.chat_completion(
            GuardrailsChatCompletionRequest.model_validate(REQUEST), Request({"type": "http", "headers": []})
        )
    )
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        assert not task.done()
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert cancelled.is_set()


@pytest.mark.parametrize("direction", ["input", "output"])
async def test_text_rails_block_before_tool_request_delivery(client, monkeypatch, direction):
    config = RailsConfig.from_content(
        config={"models": [], "passthrough": True, "rails": {direction: {"flows": ["block text"]}}},
        colang_content="""
define flow block text
  bot refuse text
  abort

define bot refuse text
  "Text rejected."
""",
    )
    model = FakeLLMModel(responses=["Unsafe answer."])
    rails = LLMRails(config, llm=model)
    monkeypatch.setattr(api, "_get_rails", AsyncMock(return_value=rails))
    frames = events(await client.post("/v1/chat/completions", json=REQUEST))
    assert assembled_message(frames)["content"] == "Text rejected."
    assert frames[-1]["choices"][0]["finish_reason"] == "stop"
    assert model.inference_count == (1 if direction == "output" else 0)
    assert "Unsafe answer" not in json.dumps(frames)


@pytest.mark.parametrize("status", [400, 429, 500, 502, 503])
async def test_generation_errors_remain_http_errors_before_sse(client, monkeypatch, status):
    from nemoguardrails.exceptions import LLMCallException, LLMClientError

    rails = AsyncMock()
    rails.config = RailsConfig.from_content(config={"models": []})
    rails.generate_async.side_effect = LLMCallException(
        LLMClientError(status, "Provider rejected request"), status=status
    )
    monkeypatch.setattr(api, "_get_rails", AsyncMock(return_value=rails))
    response = await client.post("/v1/chat/completions", json=REQUEST)
    assert response.status_code == status
    assert response.headers["content-type"] == "application/json"
    assert response.json()["error"]["message"] == "Provider rejected request"
    assert "data:" not in response.text


@pytest.mark.parametrize("wrapper", ["direct", "llm", "engine", "nested"])
async def test_response_validation_errors_are_bad_gateway(client, monkeypatch, wrapper):
    from nemoguardrails.exceptions import LLMCallException, LLMResponseValidationError
    from nemoguardrails.guardrails.model_engine import ModelEngineError

    error = LLMResponseValidationError("Invalid tool arguments")
    if wrapper in ("engine", "nested"):
        error = ModelEngineError("Provider response invalid", model_name="private-model", inner_exception=error)
    if wrapper in ("llm", "nested"):
        error = LLMCallException(error)
    rails = AsyncMock()
    rails.config = RailsConfig.from_content(config={"models": []})
    rails.generate_async.side_effect = error
    monkeypatch.setattr(api, "_get_rails", AsyncMock(return_value=rails))

    response = await client.post("/v1/chat/completions", json=REQUEST)

    assert response.status_code == 502
    assert response.headers["content-type"] == "application/json"
    assert response.json()["error"]["type"] == "server_error"
    assert response.json()["error"]["message"] == "Invalid tool arguments"
    assert "x-should-retry" not in response.headers
    assert "data:" not in response.text


async def test_unexpected_generation_error_remains_internal_error(client, monkeypatch):
    rails = AsyncMock()
    rails.config = RailsConfig.from_content(config={"models": []})
    rails.generate_async.side_effect = RuntimeError("Internal failure")
    monkeypatch.setattr(api, "_get_rails", AsyncMock(return_value=rails))

    response = await client.post("/v1/chat/completions", json=REQUEST)

    assert response.status_code == 500
    assert response.json()["error"]["message"] == "Internal server error"


async def test_iorails_overload_remains_http_error(client, monkeypatch):
    from nemoguardrails.exceptions import NonStreamingWorkQueueFullError

    rails = AsyncMock()
    rails.config = RailsConfig.from_content(config={"models": []})
    rails.generate_async.side_effect = NonStreamingWorkQueueFullError()
    monkeypatch.setattr(api, "_get_rails", AsyncMock(return_value=rails))
    response = await client.post("/v1/chat/completions", json=REQUEST)
    assert response.status_code == 503
    assert response.headers["content-type"] == "application/json"


async def test_sse_formatter_preserves_approved_fields_and_usage():
    from nemoguardrails.server.schemas.openai import GuardrailsChatCompletion
    from nemoguardrails.server.schemas.utils import chat_completion_to_sse

    completion = GuardrailsChatCompletion.model_validate(
        {
            "id": "chatcmpl-tested",
            "created": 42,
            "model": "test-model",
            "object": "chat.completion",
            "system_fingerprint": "fp_test",
            "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
            "guardrails": {"config_id": "tools", "output_data": {"approved": True}},
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Allowed text",
                        "refusal": "Approved refusal",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "weather", "arguments": '{"city":"Paris"}'},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        }
    )
    before = completion.model_dump()
    frames = list(chat_completion_to_sse(completion))
    payloads = [json.loads(frame[6:]) for frame in frames[:-1]]
    assert completion.model_dump() == before
    assert all(
        p["id"] == "chatcmpl-tested" and p["created"] == 42 and p["system_fingerprint"] == "fp_test" for p in payloads
    )
    assert assembled_message(payloads)["refusal"] == "Approved refusal"
    assert payloads[0]["guardrails"]["output_data"] == {"approved": True}
    assert payloads[-1]["choices"] == []
    assert payloads[-1]["usage"]["total_tokens"] == 5
    assert frames[-1] == "data: [DONE]\n\n"


@pytest.mark.parametrize("iorails", [False, True])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("arguments", ['{"city":"Paris"}', '{"city":', "[]", "null"])
async def test_provider_response_is_buffered_and_parsed_before_delivery(
    client, monkeypatch, httpx_mock, iorails, arguments, stream
):
    from aioresponses import CallbackResult, aioresponses

    url = "http://upstream.invalid/v1/chat/completions"
    config = RailsConfig.from_content(
        config={
            "passthrough": True,
            "models": [
                {
                    "type": "main",
                    "engine": "openai",
                    "model": "test-model",
                    "parameters": {"base_url": "http://upstream.invalid/v1", "api_key": "test-key", "max_retries": 0},
                }
            ],
        }
    )
    rails = Guardrails(config, use_iorails=iorails, require_iorails=iorails)
    monkeypatch.setattr(api, "_get_rails", AsyncMock(return_value=rails))
    payload = {
        "id": "chatcmpl-upstream",
        "model": "test-model",
        "object": "chat.completion",
        "created": 42,
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_Paris",
                            "type": "function",
                            "function": {"name": "weather", "arguments": arguments},
                        }
                    ],
                },
            }
        ],
    }
    requests = []

    def httpx_response(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=payload)

    def aiohttp_response(url, **kwargs):
        requests.append(kwargs["json"])
        return CallbackResult(payload=payload)

    try:
        with aioresponses() as upstream:
            if iorails:
                upstream.post(url, callback=aiohttp_response)
            else:
                httpx_mock.add_callback(httpx_response, url=url)
            response = await client.post("/v1/chat/completions", json={**REQUEST, "stream": stream})
        assert len(requests) == 1
        assert not requests[0].get("stream", False)
        assert requests[0]["tools"] == TOOLS
        if arguments.endswith("}"):
            assert response.status_code == 200
            message = assembled_message(events(response)) if stream else response.json()["choices"][0]["message"]
            assert message["tool_calls"][0]["id"] == "call_Paris"
        else:
            assert response.status_code == 502
            assert response.json()["error"]["type"] == "server_error"
            assert response.headers["content-type"] == "application/json"
            assert "data:" not in response.text
    finally:
        await rails.shutdown()


@pytest.mark.parametrize("text", ["", 'Quoted "text", backslash \\, newline\n and Unicode 漢😊. ' * 5])
async def test_successive_deltas_reconstruct_exact_approved_messages(text):
    from nemoguardrails.server.schemas.openai import GuardrailsChatCompletion
    from nemoguardrails.server.schemas.utils import chat_completion_to_sse

    arguments = json.dumps({"text": text, "cities": ["Zürich", "東京"]}, ensure_ascii=False)
    message = {
        "role": "assistant",
        "content": text,
        "refusal": text,
        "tool_calls": [
            {"id": f"call_{index}", "type": "function", "function": {"name": "lookup", "arguments": arguments}}
            for index in range(2)
        ],
    }
    completion = GuardrailsChatCompletion.model_validate(
        {
            "id": "chatcmpl-fragments",
            "created": 42,
            "object": "chat.completion",
            "model": "test-model",
            "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls"}],
        }
    )
    before = completion.model_dump()
    frames = [json.loads(frame[6:]) for frame in list(chat_completion_to_sse(completion))[:-1]]
    assert assembled_message(frames) == message
    assert completion.model_dump() == before
    deltas = [frame["choices"][0]["delta"] for frame in frames[:-1]]
    assert deltas[0] == {"role": "assistant"}
    assert all(frame["choices"][0]["finish_reason"] is None for frame in frames[:-1])
    assert frames[-1]["choices"][0]["finish_reason"] == "tool_calls"
    for index in range(2):
        calls = [call for delta in deltas for call in delta.get("tool_calls", []) if call["index"] == index]
        assert calls[0] == {
            "index": index,
            "id": f"call_{index}",
            "type": "function",
            "function": {"name": "lookup", "arguments": ""},
        }
        assert len(calls) > 2
        assert all(set(call) == {"index", "function"} for call in calls[1:])
        assert all(set(call["function"]) == {"arguments"} for call in calls[1:])
        assert "".join(call["function"]["arguments"] for call in calls) == arguments
    if text:
        for key in ("content", "refusal"):
            pieces = [delta[key] for delta in deltas if key in delta]
            assert len(pieces) > 1
            assert all(0 < len(piece) < len(text) for piece in pieces)
