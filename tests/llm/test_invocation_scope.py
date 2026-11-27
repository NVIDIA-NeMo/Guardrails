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

"""Shared calls isolate state while retaining complete streaming responses."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from nemoguardrails.context import (
    explain_info_var,
    llm_call_info_var,
    llm_response_metadata_var,
    llm_stats_var,
    reasoning_trace_var,
    tool_calls_var,
)
from nemoguardrails.exceptions import LLMCallException
from nemoguardrails.llm.call import llm_call
from nemoguardrails.logging.explain import ExplainInfo, LLMCallInfo
from nemoguardrails.logging.processing_log import processing_log_var
from nemoguardrails.logging.stats import LLMStats
from nemoguardrails.streaming import StreamingHandler
from nemoguardrails.testing.fake_model import FakeLLMModel
from nemoguardrails.types import LLMResponse, LLMResponseChunk, ToolCall, ToolCallChunk, ToolCallFunction, UsageInfo


@pytest.fixture(autouse=True)
def invocation_context():
    values = {
        explain_info_var: ExplainInfo(),
        llm_call_info_var: None,
        llm_response_metadata_var: None,
        llm_stats_var: LLMStats(),
        reasoning_trace_var: None,
        tool_calls_var: None,
        processing_log_var: [],
    }
    tokens = [(variable, variable.set(value)) for variable, value in values.items()]
    yield
    for variable, token in reversed(tokens):
        variable.reset(token)


def chunk_model(monkeypatch, chunks, error=None):
    model = FakeLLMModel(responses=[])
    closed = []

    async def stream(prompt, *, stop=None, **kwargs):
        model.inference_count += 1
        try:
            for chunk in chunks:
                yield chunk
            if error is not None:
                raise error
        finally:
            closed.append(True)

    monkeypatch.setattr(model, "stream_async", stream)
    return model, closed


@pytest.mark.asyncio
async def test_typed_consumer_receives_terminal_metadata_and_final_response(monkeypatch):
    calls = [ToolCall(id="call-1", function=ToolCallFunction(name="weather", arguments={"city": "Paris"}))]
    chunks = [
        LLMResponseChunk(delta_reasoning="thinking "),
        LLMResponseChunk(delta_content="Hello "),
        LLMResponseChunk(delta_content="world", provider_metadata={"first": True}),
        LLMResponseChunk(delta_tool_calls=calls, finish_reason="tool_calls"),
        LLMResponseChunk(
            model="terminal-model",
            request_id="terminal-request",
            usage=UsageInfo(input_tokens=7, output_tokens=3, total_tokens=10),
            provider_metadata={"last": True},
        ),
    ]
    model, closed = chunk_model(monkeypatch, chunks)
    delivered = []

    async def consume(chunk):
        delivered.append(chunk)

    response = await llm_call(model, "prompt", chunk_consumer=consume)

    assert delivered == chunks
    assert model.inference_count == 1
    assert closed == [True]
    assert response.content == "Hello world"
    assert response.reasoning == "thinking "
    assert response.tool_calls == calls
    assert response.model == "terminal-model"
    assert response.request_id == "terminal-request"
    assert response.finish_reason == "tool_calls"
    assert response.provider_metadata == {"first": True, "last": True}
    assert response.usage.total_tokens == 10
    assert reasoning_trace_var.get() == response.reasoning
    assert tool_calls_var.get() == [call.to_dict() for call in calls]
    assert llm_response_metadata_var.get() == response.provider_metadata
    info = llm_call_info_var.get()
    assert (info.request_id, info.completion, info.total_tokens) == ("terminal-request", "Hello world", 10)
    assert llm_stats_var.get().get_stat("total_calls") == 1
    assert llm_stats_var.get().get_stat("total_tokens") == 10


@pytest.mark.asyncio
async def test_consumer_and_handler_are_rejected_before_invocation():
    model = FakeLLMModel(responses=["unused"])

    with pytest.raises(ValueError, match="mutually exclusive"):
        await llm_call(model, "prompt", streaming_handler=StreamingHandler(), chunk_consumer=AsyncMock())

    assert model.inference_count == 0


@pytest.mark.asyncio
async def test_consumer_failure_closes_provider_and_preserves_failure(monkeypatch):
    model, closed = chunk_model(
        monkeypatch, [LLMResponseChunk(delta_content="first"), LLMResponseChunk(delta_content="last")]
    )
    error = ValueError("consumer failed")

    with pytest.raises(ValueError) as caught:
        await llm_call(model, "prompt", chunk_consumer=AsyncMock(side_effect=error))

    assert caught.value is error
    assert closed == [True]
    assert processing_log_var.get()[0]["data"].finished_at > 0


@pytest.mark.asyncio
async def test_provider_failure_closes_stream_and_keeps_exception_cause(monkeypatch):
    error = RuntimeError("provider failed")
    model, closed = chunk_model(monkeypatch, [LLMResponseChunk(delta_content="first")], error=error)

    with pytest.raises(LLMCallException) as caught:
        await llm_call(model, "prompt", chunk_consumer=AsyncMock())

    assert caught.value.__cause__ is error
    assert closed == [True]


@pytest.mark.asyncio
@pytest.mark.parametrize("use_handler", [False, True])
async def test_cancellation_closes_stream_and_restores_outer_context(monkeypatch, use_handler):
    model, closed = chunk_model(monkeypatch, [LLMResponseChunk(delta_content="first")])
    waiting = asyncio.Event()

    async def consume(*args):
        waiting.set()
        await asyncio.Event().wait()

    seed = LLMCallInfo(task="outer")
    handler = StreamingHandler()
    monkeypatch.setattr(handler, "push_chunk", consume)

    async def run():
        llm_call_info_var.set(seed)
        reasoning_trace_var.set("outer reasoning")
        tool_calls_var.set([{"id": "outer tool"}])
        llm_response_metadata_var.set({"outer": True})
        try:
            if use_handler:
                await llm_call(model, "prompt", streaming_handler=handler)
            else:
                await llm_call(model, "prompt", chunk_consumer=consume)
        except asyncio.CancelledError:
            assert llm_call_info_var.get() is seed
            assert reasoning_trace_var.get() == "outer reasoning"
            assert tool_calls_var.get() == [{"id": "outer tool"}]
            assert llm_response_metadata_var.get() == {"outer": True}
            raise

    task = asyncio.create_task(run())
    await waiting.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert closed == [True]
    assert seed.id is None


@pytest.mark.asyncio
async def test_sequential_calls_have_independent_records_without_stale_response_fields():
    llm_call_info_var.set(LLMCallInfo(task="check"))
    model = FakeLLMModel(
        llm_responses=[
            LLMResponse(
                content="first",
                request_id="request-1",
                usage=UsageInfo(input_tokens=2, output_tokens=1, total_tokens=3),
            ),
            LLMResponse(content="second"),
        ]
    )

    await llm_call(model, "first prompt")
    first = llm_call_info_var.get()
    await llm_call(model, "second prompt")
    second = llm_call_info_var.get()

    assert first is not second
    assert first.id != second.id
    assert first.completion == "first"
    assert first.request_id == "request-1"
    assert second.completion == "second"
    assert second.request_id is None
    assert second.total_tokens == 0
    assert first.task == second.task == "check"
    recorded = [entry["data"] for entry in processing_log_var.get()]
    assert [info.completion for info in recorded] == ["first", "second"]
    assert recorded[0] is not recorded[1]
    assert [info.completion for info in explain_info_var.get().llm_calls] == ["first", "second"]


@pytest.mark.asyncio
async def test_nested_calls_restore_outer_record_and_response_context(monkeypatch):
    inner = FakeLLMModel(
        llm_responses=[LLMResponse(content="inner", reasoning="inner reasoning", usage=UsageInfo(2, 1, 3))]
    )
    outer = FakeLLMModel(responses=[])

    async def generate(prompt, *, stop=None, **kwargs):
        info = llm_call_info_var.get()
        reasoning_trace_var.set("pending outer reasoning")
        tool_calls_var.set([{"id": "pending outer tool"}])
        llm_response_metadata_var.set({"pending_outer": True})
        await llm_call(inner, "inner prompt")
        assert llm_call_info_var.get() is info
        assert reasoning_trace_var.get() == "pending outer reasoning"
        assert tool_calls_var.get() == [{"id": "pending outer tool"}]
        assert llm_response_metadata_var.get() == {"pending_outer": True}
        return LLMResponse(content="outer", usage=UsageInfo(4, 2, 6))

    monkeypatch.setattr(outer, "generate_async", generate)
    await llm_call(outer, "outer prompt")

    infos = [entry["data"] for entry in processing_log_var.get()]
    assert [info.completion for info in infos] == ["inner", "outer"]
    assert infos[0].id != infos[1].id
    assert [info.total_tokens for info in infos] == [3, 6]
    assert llm_stats_var.get().get_stat("total_tokens") == 9
    assert llm_stats_var.get().get_stat("total_calls") == 2
    assert llm_call_info_var.get().completion == "outer"
    assert [info.completion for info in explain_info_var.get().llm_calls] == ["outer", "inner"]


@pytest.mark.asyncio
async def test_concurrent_calls_do_not_mutate_an_inherited_record(monkeypatch):
    inherited = LLMCallInfo(task="shared caller")
    llm_call_info_var.set(inherited)
    ready = asyncio.Event()
    arrived = 0

    async def run(label):
        model = FakeLLMModel(responses=[])

        async def generate(prompt, *, stop=None, **kwargs):
            nonlocal arrived
            arrived += 1
            if arrived == 2:
                ready.set()
            await ready.wait()
            return LLMResponse(content=label, request_id=label)

        monkeypatch.setattr(model, "generate_async", generate)
        await llm_call(model, label)
        return llm_call_info_var.get()

    first, second = await asyncio.gather(run("first"), run("second"))

    assert first is not second
    assert first.id != second.id
    assert (first.completion, second.completion) == ("first", "second")
    assert first.task == second.task == "shared caller"
    assert inherited.id is None
    assert inherited.completion is None
    assert llm_call_info_var.get() is inherited
    assert len(processing_log_var.get()) == 2


@pytest.mark.asyncio
async def test_compiled_action_collects_each_invocation_as_a_distinct_record(monkeypatch):
    from nemoguardrails.actions.rail_outcome import RailOutcome
    from nemoguardrails.guardrails.compiled_rail import RailDependencies, compile_rail
    from nemoguardrails.llm.taskmanager import LLMTaskManager
    from nemoguardrails.manifests import RailDirection
    from nemoguardrails.rails.llm.config import Model, RailsConfig

    config = RailsConfig(models=[Model(type="content_safety", engine="fake", model="safety")])
    model = FakeLLMModel(llm_responses=[LLMResponse(content="first"), LLMResponse(content="second")])
    deps = RailDependencies(llms={"content_safety": model}, llm_task_manager=LLMTaskManager(config), config=config)

    async def action(**kwargs):
        llm_call_info_var.set(LLMCallInfo(task="two model checks"))
        await llm_call(model, "first prompt")
        await llm_call(model, "second prompt")
        return RailOutcome.allow()

    monkeypatch.setattr("nemoguardrails.library.content_safety.actions.content_safety_check_input", action)
    execution = await compile_rail(
        "content safety check input $model=content_safety", RailDirection.INPUT, deps
    ).execute([{"role": "user", "content": "hello"}])

    assert [info.completion for info in execution.llm_calls] == ["first", "second"]
    assert execution.llm_calls[0].id != execution.llm_calls[1].id
    assert all(info.task == "two model checks" for info in execution.llm_calls)
    assert processing_log_var.get() == []


@pytest.mark.asyncio
async def test_logging_failure_restores_context_without_masking_provider_failure():
    class RejectingLog(list):
        def append(self, entry):
            raise RuntimeError("log sink failed")

    seed = LLMCallInfo(task="caller")
    llm_call_info_var.set(seed)
    reasoning_trace_var.set("caller reasoning")
    processing_log_var.set(RejectingLog())
    provider_error = ValueError("provider failed")

    with pytest.raises(LLMCallException) as caught:
        await llm_call(FakeLLMModel(llm_exception=provider_error), "prompt")

    assert caught.value.__cause__ is provider_error
    assert llm_call_info_var.get() is seed
    assert reasoning_trace_var.get() == "caller reasoning"
    processing_log_var.set([])
    await llm_call(FakeLLMModel(responses=["later"]), "later prompt")
    assert llm_call_info_var.get().completion == "later"


@pytest.mark.asyncio
@pytest.mark.parametrize("delivery", ["handler", "consumer"])
async def test_provider_failure_wins_over_stream_cleanup_failure(monkeypatch, delivery):
    error = ValueError("provider failed")

    class Stream:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise error

        async def aclose(self):
            raise RuntimeError("stream cleanup failed")

    model = FakeLLMModel(responses=[])
    monkeypatch.setattr(model, "stream_async", lambda *args, **kwargs: Stream())
    kwargs = {"streaming_handler": StreamingHandler()} if delivery == "handler" else {"chunk_consumer": AsyncMock()}
    with pytest.raises(LLMCallException) as caught:
        await llm_call(model, "prompt", **kwargs)
    assert caught.value.__cause__ is error


@pytest.mark.asyncio
@pytest.mark.parametrize("delivery", ["handler", "consumer"])
async def test_cancellation_wins_over_stream_cleanup_failure(monkeypatch, delivery):
    waiting = asyncio.Event()

    class Stream:
        def __aiter__(self):
            return self

        async def __anext__(self):
            waiting.set()
            await asyncio.Event().wait()

        async def aclose(self):
            raise RuntimeError("stream cleanup failed")

    model = FakeLLMModel(responses=[])
    monkeypatch.setattr(model, "stream_async", lambda *args, **kwargs: Stream())
    kwargs = {"streaming_handler": StreamingHandler()} if delivery == "handler" else {"chunk_consumer": AsyncMock()}
    task = asyncio.create_task(llm_call(model, "prompt", **kwargs))
    await waiting.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
@pytest.mark.parametrize("delivery", ["handler", "typed"])
@pytest.mark.parametrize("finish_reason", ["tool_calls", "length", "content_filter"])
@pytest.mark.parametrize("cumulative", [False, True])
async def test_completed_tool_call_batches_are_retained_in_order(monkeypatch, delivery, finish_reason, cumulative):
    calls = [
        ToolCall(id="first", function=ToolCallFunction(name="one", arguments={})),
        ToolCall(id="second", function=ToolCallFunction(name="two", arguments={})),
    ]
    first_batch = [calls[0]]
    second_batch = calls[:] if cumulative else [calls[1]]
    chunks = [
        LLMResponseChunk(delta_content="Hello"),
        LLMResponseChunk(delta_tool_calls=first_batch),
        LLMResponseChunk(delta_tool_calls=second_batch, finish_reason=finish_reason),
        LLMResponseChunk(delta_tool_calls=[], usage=UsageInfo(input_tokens=2, output_tokens=3, total_tokens=5)),
    ]
    model, closed = chunk_model(monkeypatch, chunks)
    delivered = []
    if delivery == "typed":

        async def consume(chunk):
            delivered.extend(chunk.delta_tool_calls or [])

        response = await llm_call(model, "prompt", chunk_consumer=consume)
    else:
        handler = StreamingHandler()
        response = await llm_call(model, "prompt", streaming_handler=handler)
        events = [event async for event in handler]
        delivered = [call for event in events if isinstance(event, ToolCallChunk) for call in event.tool_calls]
    assert delivered == first_batch + second_batch
    assert response.tool_calls == calls
    assert response.content == "Hello"
    assert response.finish_reason == finish_reason
    assert tool_calls_var.get() == [call.to_dict() for call in calls]
    assert first_batch == [calls[0]]
    assert second_batch == (calls if cumulative else [calls[1]])
    assert closed == [True]
    assert llm_call_info_var.get().completion == "Hello"
    assert llm_call_info_var.get().total_tokens == 5
    assert llm_stats_var.get().get_stat("total_calls") == 1
    assert llm_stats_var.get().get_stat("total_tokens") == 5


@pytest.mark.asyncio
async def test_completed_call_updates_keep_order_and_anonymous_calls(monkeypatch):
    first = ToolCall(id="first", function=ToolCallFunction(name="lookup", arguments={"version": 1}))
    updated = ToolCall(id="first", function=ToolCallFunction(name="lookup", arguments={"version": 2}))
    anonymous_a = ToolCall(id="", function=ToolCallFunction(name="a", arguments={}))
    anonymous_b = ToolCall(id="", function=ToolCallFunction(name="b", arguments={}))
    model, _ = chunk_model(
        monkeypatch,
        [
            LLMResponseChunk(delta_tool_calls=[first, anonymous_a]),
            LLMResponseChunk(delta_tool_calls=[updated, anonymous_b], finish_reason="tool_calls"),
        ],
    )

    async def consume(chunk):
        pass

    response = await llm_call(model, "prompt", chunk_consumer=consume)
    assert response.tool_calls == [updated, anonymous_a, anonymous_b]
    assert first.function.arguments == {"version": 1}
