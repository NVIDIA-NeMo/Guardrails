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

import asyncio
import json
from contextlib import suppress

import httpx
import pytest

from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.context import (
    llm_call_info_var,
    llm_response_metadata_var,
    llm_stats_var,
    reasoning_trace_var,
    tool_calls_var,
)
from nemoguardrails.exceptions import LLMCallException
from nemoguardrails.llm.call import _stream_llm_call
from nemoguardrails.llm.clients.openai_compatible import OpenAICompatibleClient
from nemoguardrails.llm.models.openai_chat import OpenAIChatModel
from nemoguardrails.server import api
from nemoguardrails.streaming import StreamingHandler
from nemoguardrails.types import ToolCallChunk, ToolCallDeltaChunk

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def reset_context():
    variables = [tool_calls_var, llm_call_info_var, llm_stats_var, llm_response_metadata_var, reasoning_trace_var]
    tokens = [(variable, variable.set(None)) for variable in variables]
    yield
    for variable, token in tokens:
        variable.reset(token)


class GatedProviderStream(httpx.AsyncByteStream):
    def __init__(self):
        self.release = asyncio.Event()
        self.paused = asyncio.Event()
        self.closed = asyncio.Event()
        self.finish_reason = "tool_calls"
        self.fail = False
        self.leading_text = None

    async def __aiter__(self):
        if self.leading_text:
            payload = {"choices": [{"index": 0, "delta": {"content": self.leading_text}, "finish_reason": None}]}
            yield ("data: " + json.dumps(payload) + "\n\n").encode()
        deltas = [
            [
                {
                    "index": 0,
                    "id": "call_paris",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city":'},
                }
            ],
            [{"index": 1, "id": "call_", "type": "function", "function": {"name": "get_", "arguments": '{"city":'}}],
            [
                {"index": 0, "function": {"arguments": '"Paris"}'}},
                {"index": 1, "id": "boston", "function": {"name": "weather", "arguments": '"Boston"}'}},
            ],
        ]
        for index, calls in enumerate(deltas):
            payload = {
                "id": "provider_response",
                "model": "test-model",
                "choices": [{"index": 0, "delta": {"tool_calls": calls}, "finish_reason": None}],
            }
            yield ("data: " + json.dumps(payload) + "\n\n").encode()
            if index == 0:
                self.paused.set()
                await self.release.wait()
                if self.fail:
                    raise httpx.ReadError("provider interrupted")
        yield (
            "data: "
            + json.dumps({"choices": [{"index": 0, "delta": {}, "finish_reason": self.finish_reason}]})
            + "\n\n"
        ).encode()
        yield b'data: {"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":6,"total_tokens":16}}\n\n'
        yield b"data: [DONE]\n\n"

    async def aclose(self):
        self.closed.set()


@pytest.fixture
def provider_stream(httpx_mock):
    stream = GatedProviderStream()
    httpx_mock.add_response(
        url="https://provider.test/v1/chat/completions",
        headers={"content-type": "text/event-stream"},
        stream=stream,
    )
    return stream


def make_model():
    return OpenAIChatModel(
        OpenAICompatibleClient(base_url="https://provider.test/v1", api_key="test", max_retries=0),
        model="test-model",
    )


@pytest.mark.parametrize("finish_reason", ["tool_calls", "stop", "length", "content_filter"])
@pytest.mark.parametrize("include_metadata", [False, True])
async def test_delta_arrives_before_completion_and_final_calls_are_not_duplicated(
    provider_stream, finish_reason, include_metadata
):
    provider_stream.finish_reason = finish_reason
    handler = StreamingHandler(include_metadata=include_metadata)
    model = make_model()
    context_token = tool_calls_var.set(None)
    task = asyncio.create_task(_stream_llm_call(model, "weather", handler, stop=["Paris"]))
    iterator = handler.__aiter__()
    emitted = []
    try:
        await asyncio.wait_for(provider_stream.paused.wait(), 2)
        while True:
            chunk = await asyncio.wait_for(anext(iterator), 2)
            emitted.append(chunk)
            if isinstance(chunk, ToolCallDeltaChunk):
                break
        assert chunk.tool_calls[0].function.arguments == '{"city":'
        assert chunk.finish_reason is None
        assert not task.done()
        assert not provider_stream.release.is_set()
        provider_stream.release.set()
        response = await asyncio.wait_for(task, 2)
        emitted.extend([chunk async for chunk in iterator])
        events = [chunk for chunk in emitted if isinstance(chunk, ToolCallDeltaChunk)]
        assert len(events) == 4
        assert not any(isinstance(chunk, ToolCallChunk) for chunk in emitted)
        assert events[-1].tool_calls == []
        assert events[-1].finish_reason == finish_reason
        assert response.finish_reason == finish_reason
        assert response.content == handler.completion == ""
        assert response.usage.total_tokens == 16
        if finish_reason in ("stop", "tool_calls"):
            assert [(call.id, call.function.name, call.function.arguments) for call in response.tool_calls] == [
                ("call_paris", "get_weather", {"city": "Paris"}),
                ("call_boston", "get_weather", {"city": "Boston"}),
            ]
        else:
            assert response.tool_calls is None
        assert provider_stream.closed.is_set()
    finally:
        provider_stream.release.set()
        task.cancel()
        with suppress(asyncio.CancelledError, LLMCallException):
            await task
        await iterator.aclose()
        tool_calls_var.reset(context_token)


async def test_interrupted_provider_does_not_emit_completed_calls(provider_stream):
    provider_stream.fail = True
    handler = StreamingHandler()
    model = make_model()
    token = tool_calls_var.set([{"id": "stale"}])
    try:
        provider_stream.release.set()
        with pytest.raises(LLMCallException):
            await _stream_llm_call(model, "weather", handler, stop=None)
        await handler.finish()
        events = [chunk async for chunk in handler if isinstance(chunk, (ToolCallChunk, ToolCallDeltaChunk))]
        assert len(events) == 1
        assert isinstance(events[0], ToolCallDeltaChunk)
        assert events[0].finish_reason is None
        assert tool_calls_var.get() is None
        assert provider_stream.closed.is_set()
    finally:
        tool_calls_var.reset(token)


async def test_cancelling_after_first_delta_closes_upstream(provider_stream):
    handler = StreamingHandler()
    task = asyncio.create_task(_stream_llm_call(make_model(), "weather", handler, stop=None))
    await asyncio.wait_for(provider_stream.paused.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert provider_stream.closed.is_set()


@pytest.mark.parametrize("cancel_consumer", [False, True])
async def test_closing_public_stream_closes_stalled_provider(provider_stream, cancel_consumer):
    rails = LLMRails(
        RailsConfig.from_content(config={"models": [], "passthrough": True, "streaming": True}),
        llm=make_model(),
    )
    stream = rails.stream_async(prompt="Weather?")
    consumer = None
    try:
        while not isinstance(await asyncio.wait_for(anext(stream), 2), ToolCallDeltaChunk):
            pass
        await asyncio.wait_for(provider_stream.paused.wait(), 2)
        if cancel_consumer:
            started = asyncio.Event()

            async def consume():
                started.set()
                return await anext(stream)

            consumer = asyncio.create_task(consume())
            await started.wait()
            consumer.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(consumer, 2)
        else:
            await asyncio.wait_for(stream.aclose(), 2)
        assert provider_stream.closed.is_set()
        assert not provider_stream.release.is_set()
        assert not rails._active_tasks
    finally:
        provider_stream.release.set()
        if consumer is not None:
            consumer.cancel()
            await asyncio.gather(consumer, return_exceptions=True)
        await stream.aclose()


@pytest.mark.parametrize("rail_type", [None, "output", "tool_call"])
@pytest.mark.parametrize("passthrough", [False, True])
@pytest.mark.parametrize("text_before_tool", [False, True])
async def test_server_sends_delta_before_upstream_finishes(
    provider_stream, monkeypatch, rail_type, passthrough, text_before_tool
):
    provider_stream.leading_text = "Checking weather. " if text_before_tool else None
    config = {"models": [], "passthrough": passthrough, "streaming": True}
    if rail_type == "output":
        config["rails"] = {"output": {"flows": ["check output"], "streaming": {"enabled": True, "stream_first": False}}}
    elif rail_type == "tool_call":
        config["rails"] = {"tool_call": {"flows": ["check output"]}}
    rails = LLMRails(
        RailsConfig.from_content(
            config=config,
            colang_content='define flow check output\n  if $bot_message == "BLOCK"\n    bot refuse to respond\n    stop\n',
        ),
        llm=make_model(),
    )

    async def get_rails(*args, **kwargs):
        return rails

    monkeypatch.setattr(api, "_get_rails", get_rails)
    request_body = json.dumps(
        {
            "model": "test-model",
            "messages": [{"role": "user", "content": "weather"}],
            "stream": True,
            "guardrails": {"config_id": "test"},
        }
    ).encode()
    delivered = asyncio.Event()
    sent = []
    received = False

    async def receive():
        nonlocal received
        if not received:
            received = True
            return {"type": "http.request", "body": request_body, "more_body": False}
        await asyncio.Event().wait()

    async def send(message):
        sent.append(message)
        if b'"tool_calls": [' in message.get("body", b""):
            delivered.set()

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/chat/completions",
        "raw_path": b"/v1/chat/completions",
        "query_string": b"",
        "root_path": "",
        "headers": [(b"content-type", b"application/json")],
        "server": ("testserver", 80),
        "client": ("127.0.0.1", 1234),
    }
    task = asyncio.create_task(api.app(scope, receive, send))
    try:
        await asyncio.wait_for(provider_stream.paused.wait(), 2)
        if rail_type is None:
            await asyncio.wait_for(delivered.wait(), 2)
            assert not task.done()
            assert not provider_stream.release.is_set()
            provider_stream.release.set()
        done, _ = await asyncio.wait({task}, timeout=5)
        assert task in done, "The server waited for the provider after rejecting its tool-call fragment"
        await task
        assert provider_stream.closed.is_set()
        body = b"".join(message.get("body", b"") for message in sent).decode()
        if rail_type is not None:
            assert not provider_stream.release.is_set()
            assert not delivered.is_set()
            assert "Tool calls are not supported with output rails or tool-call rails during streaming." in body
            if text_before_tool and rail_type == "tool_call":
                assert next(message["status"] for message in sent if message["type"] == "http.response.start") == 200
                assert body.endswith("data: [DONE]\n\n")
        else:
            frames = [json.loads(line[6:]) for line in body.splitlines() if line.startswith("data: {")]
            choices = [frame["choices"][0] for frame in frames]
            assert sum(bool(choice["delta"].get("tool_calls")) for choice in choices) == 3
            terminal = [choice for choice in choices if choice["finish_reason"]]
            assert terminal == [{"delta": {}, "index": 0, "finish_reason": "tool_calls"}]
            assert body.endswith("data: [DONE]\n\n")
    finally:
        provider_stream.release.set()
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


@pytest.mark.parametrize("mode", ["buffer", "output", "pipe", "stop"])
async def test_partial_arguments_cannot_bypass_unsupported_paths(mode):
    from nemoguardrails.exceptions import StreamingNotSupportedError
    from nemoguardrails.types import ToolCallDelta, ToolCallFunctionDelta

    handler = StreamingHandler(enable_buffer=mode == "buffer", allow_tool_calls=mode not in ("output", "pipe"))
    target = StreamingHandler()
    if mode == "pipe":
        handler.set_pipe_to(target)
    event = ToolCallDeltaChunk([ToolCallDelta(index=0, function=ToolCallFunctionDelta(arguments='{"city":'))])
    if mode == "stop":
        handler.stop = ["STOP"]
        await handler.push_chunk("STOP")
        await handler.push_chunk(event)
        await handler.finish()
        assert [chunk async for chunk in handler] == []
    elif mode == "output":
        await handler.push_chunk(event)
        with pytest.raises(StreamingNotSupportedError):
            await anext(handler)
    else:
        with pytest.raises(StreamingNotSupportedError):
            await handler.push_chunk(event)
        assert target.queue.empty()
        assert handler.buffer == ""


async def test_generation_retains_assembled_calls_after_streaming_deltas(provider_stream):
    provider_stream.release.set()
    rails = LLMRails(
        RailsConfig.from_content(config={"models": [], "passthrough": True, "streaming": True}),
        llm=make_model(),
    )
    handler = StreamingHandler()
    result = await rails.generate_async(messages=[{"role": "user", "content": "weather"}], streaming_handler=handler)
    chunks = [chunk async for chunk in handler]
    assert result["tool_calls"] == [
        {"id": "call_paris", "type": "function", "function": {"name": "get_weather", "arguments": {"city": "Paris"}}},
        {"id": "call_boston", "type": "function", "function": {"name": "get_weather", "arguments": {"city": "Boston"}}},
    ]
    assert not any(isinstance(chunk, ToolCallChunk) for chunk in chunks)
    assert sum(isinstance(chunk, ToolCallDeltaChunk) for chunk in chunks) == 4
    assert rails.explain().llm_calls[0].completion == ""
