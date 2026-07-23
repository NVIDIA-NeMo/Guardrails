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
import warnings
from unittest.mock import patch

import pytest
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from nemoguardrails.actions.llm.utils import llm_call
from nemoguardrails.guardrails import telemetry
from nemoguardrails.llm.models.instrumented import InstrumentedLLMModel, instrument_llm_model
from nemoguardrails.tracing import constants as tracing_constants
from nemoguardrails.tracing.constants import SystemConstants
from nemoguardrails.types import ChatMessage, LLMModel, LLMResponse, LLMResponseChunk, UsageInfo
from tests.guardrails.metric_helpers import collect_histogram_sum, collect_metric_points


class RecordingModel:
    model_name = "test-model"
    provider_name = "test-provider"
    provider_url = "https://example.test/v1"

    def __init__(self, response=None, chunks=None, error=None):
        self.response = response or LLMResponse(content="response")
        self.chunks = chunks or []
        self.error = error
        self.calls = []
        self.stream_closed = False

    async def generate_async(self, prompt, *, stop=None, **kwargs):
        self.calls.append((prompt, stop, kwargs))
        if self.error is not None:
            raise self.error
        return self.response

    async def stream_async(self, prompt, *, stop=None, **kwargs):
        self.calls.append((prompt, stop, kwargs))
        try:
            for chunk in self.chunks:
                yield chunk
            if self.error is not None:
                raise self.error
        finally:
            self.stream_closed = True


@pytest.fixture(autouse=True)
def reset_telemetry():
    telemetry._meter = None
    tracing_constants._llm_instruments = None
    yield
    telemetry._meter = None
    tracing_constants._llm_instruments = None


@pytest.fixture
def span_exporter():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider.get_tracer("test"), exporter


@pytest.fixture
def metric_reader():
    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    telemetry._meter = provider.get_meter(SystemConstants.SYSTEM_NAME)
    return reader


@pytest.mark.asyncio
async def test_generate_delegates_and_returns_exact_response(span_exporter):
    tracer, exporter = span_exporter
    response = LLMResponse(
        content="answer",
        model="response-model",
        request_id="request-id",
        finish_reason="stop",
        usage=UsageInfo(input_tokens=3, output_tokens=2),
    )
    model = RecordingModel(response=response)
    instrumented = InstrumentedLLMModel(model, tracer=tracer)
    prompt = [ChatMessage(role="user", content="question")]

    result = await instrumented.generate_async(prompt, stop=["END"], temperature=0.2)

    assert isinstance(instrumented, LLMModel)
    assert result is response
    assert model.calls == [(prompt, ["END"], {"temperature": 0.2})]
    attrs = dict(exporter.get_finished_spans()[0].attributes)
    assert attrs["gen_ai.request.model"] == "test-model"
    assert attrs["gen_ai.provider.name"] == "test-provider"
    assert attrs["gen_ai.request.temperature"] == 0.2
    assert list(attrs["gen_ai.request.stop_sequences"]) == ["END"]
    assert attrs["gen_ai.response.model"] == "response-model"
    assert attrs["gen_ai.response.id"] == "request-id"
    assert list(attrs["gen_ai.response.finish_reasons"]) == ["stop"]
    assert attrs["gen_ai.usage.input_tokens"] == 3
    assert attrs["gen_ai.usage.output_tokens"] == 2


@pytest.mark.asyncio
async def test_generate_preserves_exception_identity(span_exporter):
    tracer, exporter = span_exporter
    error = RuntimeError("provider failed")
    model = InstrumentedLLMModel(RecordingModel(error=error), tracer=tracer)

    with pytest.raises(RuntimeError) as exc_info:
        await model.generate_async("question")

    assert exc_info.value is error
    traceback = exc_info.value.__traceback__
    while traceback and traceback.tb_next:
        traceback = traceback.tb_next
    assert traceback is not None
    assert traceback.tb_frame.f_code is RecordingModel.generate_async.__code__
    attrs = dict(exporter.get_finished_spans()[0].attributes)
    assert attrs["error.type"] == "RuntimeError"


@pytest.mark.asyncio
async def test_llm_call_uses_instrumented_custom_model(span_exporter):
    tracer, exporter = span_exporter
    model = InstrumentedLLMModel(RecordingModel(), tracer=tracer)

    result = await llm_call(model, "question")

    assert result.content == "response"
    assert len(exporter.get_finished_spans()) == 1


@pytest.mark.asyncio
async def test_stream_preserves_chunks_and_latest_non_null_metadata(span_exporter):
    tracer, exporter = span_exporter
    chunks = [
        LLMResponseChunk(delta_content="a", model="first", request_id="id-1"),
        LLMResponseChunk(delta_reasoning="thinking", model="second"),
        LLMResponseChunk(delta_content="b", finish_reason="stop"),
        LLMResponseChunk(usage=UsageInfo(input_tokens=4, output_tokens=2)),
    ]
    underlying = RecordingModel(chunks=chunks)
    model = InstrumentedLLMModel(underlying, tracer=tracer)

    received = [chunk async for chunk in model.stream_async("question")]

    assert received == chunks
    assert all(received_chunk is source_chunk for received_chunk, source_chunk in zip(received, chunks))
    assert underlying.stream_closed
    attrs = dict(exporter.get_finished_spans()[0].attributes)
    assert attrs["gen_ai.request.stream"] is True
    assert attrs["gen_ai.response.model"] == "second"
    assert attrs["gen_ai.response.id"] == "id-1"
    assert list(attrs["gen_ai.response.finish_reasons"]) == ["stop"]
    assert attrs["gen_ai.usage.input_tokens"] == 4
    assert attrs["gen_ai.usage.output_tokens"] == 2


@pytest.mark.asyncio
async def test_stream_error_has_no_final_response_telemetry(span_exporter):
    tracer, exporter = span_exporter
    error = RuntimeError("stream failed")
    source = RecordingModel(
        chunks=[LLMResponseChunk(delta_content="partial", model="partial-model")],
        error=error,
    )
    model = InstrumentedLLMModel(source, tracer=tracer)

    with pytest.raises(RuntimeError) as exc_info:
        async for _ in model.stream_async("question"):
            pass

    assert exc_info.value is error
    traceback = exc_info.value.__traceback__
    while traceback and traceback.tb_next:
        traceback = traceback.tb_next
    assert traceback is not None
    assert traceback.tb_frame.f_code is RecordingModel.stream_async.__code__
    attrs = dict(exporter.get_finished_spans()[0].attributes)
    assert attrs["error.type"] == "RuntimeError"
    assert "gen_ai.response.model" not in attrs
    assert "gen_ai.usage.input_tokens" not in attrs


@pytest.mark.asyncio
async def test_stream_consumer_close_has_no_final_response_telemetry(span_exporter):
    tracer, exporter = span_exporter
    source = RecordingModel(
        chunks=[
            LLMResponseChunk(delta_content="partial", model="partial-model"),
            LLMResponseChunk(usage=UsageInfo(input_tokens=1, output_tokens=1)),
        ]
    )
    model = InstrumentedLLMModel(source, tracer=tracer)
    stream = model.stream_async("question")

    assert await anext(stream) is source.chunks[0]
    await stream.aclose()

    assert source.stream_closed
    attrs = dict(exporter.get_finished_spans()[0].attributes)
    assert attrs["error.type"] == "GeneratorExit"
    assert "gen_ai.response.model" not in attrs
    assert "gen_ai.usage.input_tokens" not in attrs


@pytest.mark.asyncio
async def test_stream_task_cancellation_preserves_cancelled_error(span_exporter):
    tracer, exporter = span_exporter
    started = asyncio.Event()

    class BlockingModel(RecordingModel):
        async def stream_async(self, prompt, *, stop=None, **kwargs):
            try:
                started.set()
                await asyncio.Event().wait()
                yield LLMResponseChunk(delta_content="unreachable")
            finally:
                self.stream_closed = True

    source = BlockingModel()
    model = InstrumentedLLMModel(source, tracer=tracer)

    async def consume():
        async for _ in model.stream_async("question"):
            pass

    task = asyncio.create_task(consume())
    await started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert source.stream_closed
    attrs = dict(exporter.get_finished_spans()[0].attributes)
    assert attrs["error.type"] == "CancelledError"
    assert "gen_ai.response.model" not in attrs


@pytest.mark.asyncio
async def test_content_capture_is_disabled_by_default(span_exporter):
    tracer, exporter = span_exporter
    model = InstrumentedLLMModel(RecordingModel(response=LLMResponse(content="secret output")), tracer=tracer)

    await model.generate_async("secret input")

    span = exporter.get_finished_spans()[0]
    assert all("secret" not in str(value) for value in span.attributes.values())
    assert all("secret" not in str(event.attributes) for event in span.events)


@pytest.mark.asyncio
async def test_content_capture_uses_existing_format_switch(span_exporter):
    tracer, exporter = span_exporter
    model = InstrumentedLLMModel(
        RecordingModel(response=LLMResponse(content="captured output")),
        tracer=tracer,
        content_capture_enabled=True,
    )

    with patch.dict("os.environ", {"OTEL_SEMCONV_STABILITY_OPT_IN": "gen_ai_latest_experimental"}):
        await model.generate_async("captured input")

    attrs = dict(exporter.get_finished_spans()[0].attributes)
    assert json.loads(attrs["gen_ai.input.messages"])[0]["parts"][0]["content"] == "captured input"
    assert json.loads(attrs["gen_ai.output.messages"])[0]["parts"][0]["content"] == "captured output"


@pytest.mark.asyncio
async def test_metrics_and_chunk_timing(metric_reader):
    usage = UsageInfo(input_tokens=5, output_tokens=2)
    chunks = [
        LLMResponseChunk(delta_reasoning="think"),
        LLMResponseChunk(delta_content="answer"),
        LLMResponseChunk(usage=usage),
    ]
    model = InstrumentedLLMModel(RecordingModel(chunks=chunks), metrics_enabled=True)

    async for _ in model.stream_async("question"):
        pass

    points = collect_metric_points(metric_reader)
    assert len(points["gen_ai.client.token.usage"]) == 2
    assert points["gen_ai.client.operation.time_to_first_chunk"][0].value == 1
    assert points["gen_ai.client.operation.time_per_output_chunk"][0].value == 1
    assert points["gen_ai.client.operation.duration"][0].value == 1


@pytest.mark.asyncio
async def test_disabled_telemetry_returns_original_model(metric_reader):
    model = RecordingModel()

    result = instrument_llm_model(model)
    response = await result.generate_async("question")

    assert result is model
    assert response is model.response
    assert collect_metric_points(metric_reader) == {}


@pytest.mark.asyncio
async def test_instrumentation_is_idempotent_and_does_not_own_model(span_exporter):
    tracer, exporter = span_exporter
    model = RecordingModel()
    first = InstrumentedLLMModel(model, tracer=tracer)
    second = InstrumentedLLMModel(first, metrics_enabled=True)
    third = instrument_llm_model(second, tracer=tracer)

    await third.generate_async("question")

    assert second is first
    assert third is first
    assert len(exporter.get_finished_spans()) == 1
    assert first.wrapped_model is model
    assert not hasattr(first, "close")
    assert not hasattr(first, "aclose")


@pytest.mark.asyncio
async def test_stream_cleanup_runs_outside_duration_metric(metric_reader):
    close_delay = 0.2

    class SlowClosingModel(RecordingModel):
        async def stream_async(self, prompt, *, stop=None, **kwargs):
            try:
                yield LLMResponseChunk(delta_content="a")
                yield LLMResponseChunk(delta_content="b")
            finally:
                await asyncio.sleep(close_delay)
                self.stream_closed = True

    source = SlowClosingModel()
    model = InstrumentedLLMModel(source, metrics_enabled=True)
    stream = model.stream_async("question")

    assert await anext(stream) is not None
    await stream.aclose()

    assert source.stream_closed
    points = collect_metric_points(metric_reader)
    assert len(points["gen_ai.client.operation.duration"]) == 1
    recorded_duration = collect_histogram_sum(metric_reader, "gen_ai.client.operation.duration")
    assert recorded_duration < close_delay / 2


def test_reinstrument_with_changed_settings_warns_and_keeps_original():
    instrumented = InstrumentedLLMModel(RecordingModel(), metrics_enabled=True)

    with pytest.warns(UserWarning, match="already instrumented"):
        again = InstrumentedLLMModel(instrumented, content_capture_enabled=True)

    assert again is instrumented
    assert again._content_capture_enabled is False


def test_reinstrument_with_identical_settings_does_not_warn():
    instrumented = InstrumentedLLMModel(RecordingModel(), metrics_enabled=True)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        again = InstrumentedLLMModel(instrumented, metrics_enabled=True)

    assert again is instrumented
