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

"""A per-tool rail rewriting a tool result or a tool call, driven end to end through IORails.

The rails are stubs compiled in place of the configured flows, and only the main model's
transport is mocked, so what is asserted is what the model was sent, what the other rails were
handed and what the request span recorded.
"""

import asyncio
import copy
import json
from contextlib import asynccontextmanager, suppress
from typing import AsyncIterator, Optional

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from nemoguardrails.actions.rail_outcome import RailOutcome, TransformTarget
from nemoguardrails.guardrails.iorails import REFUSAL_MESSAGE, IORails
from nemoguardrails.guardrails.model_engine import ModelEngine
from nemoguardrails.manifests import RailDirection as SurfaceDirection
from nemoguardrails.rails.llm.options import RailType
from tests.guardrails.async_helpers import started_iorails
from tests.guardrails.rail_stubs import StubRail, rails_compiled_as, rewriting_stub
from tests.guardrails.test_data import CONTENT_SAFETY_CONFIG
from tests.guardrails.test_tool_rails_iorails import (
    WEATHER_TOOL,
    _inject_json_response,
    _inject_sse_stream,
    _stream_violation_chunks,
    _text_payload,
    _text_sse_lines,
    _tool_call_payload,
    _tool_call_sse_lines,
)
from tests.guardrails.test_transform_rail_pipeline import (
    CAPTURE_CONTENT,
    _captured_input,
    _captured_output,
    _capturing,
    _has_captured_input,
    _has_captured_output,
)
from tests.guardrails.tool_helpers import make_tool_conversation

INPUT_FLOW = "content safety check input $model=content_safety"
OUTPUT_FLOW = "content safety check output $model=content_safety"
MASK_FLOW = "cel check tool result"
JUDGE_FLOW = "regex check tool result"

CALL_MASK_FLOW = "cel check tool call"
CALL_JUDGE_FLOW = "regex check tool call"

RAW_RESULT = "18C, reported by ada@example.com"
MASKED_RESULT = "18C, reported by <EMAIL>"
RAW_ARGUMENTS = {"city": "Paris, for ada@example.com"}
MASKED_ARGUMENTS = {"city": "Paris, for <EMAIL>"}


def tool_transform_config(*, output: bool = True, capture: bool = False) -> dict:
    """Content safety on input (and output), and a masking rail ahead of a judge on get_weather's results."""
    config = copy.deepcopy(CONTENT_SAFETY_CONFIG)
    if not output:
        config["rails"]["output"] = {"flows": []}
    config["rails"]["tool_result"] = {"per_tool": {"get_weather": [JUDGE_FLOW, MASK_FLOW]}}
    if capture:
        config["tracing"] = dict(CAPTURE_CONTENT)
    return config


def tool_call_transform_config(*, output: bool = True, results: bool = False, capture: bool = False) -> dict:
    """get_weather declared on the main model, and a masking rail ahead of a judge on its calls."""
    config = tool_transform_config(output=output, capture=capture)
    if not results:
        del config["rails"]["tool_result"]
    config["models"][0] = {**config["models"][0], "parameters": {"tools": [WEATHER_TOOL]}}
    config["rails"]["tool_call"] = {"per_tool": {"get_weather": [CALL_JUDGE_FLOW, CALL_MASK_FLOW]}}
    return config


class ToolTransformRails:
    """The stubs one IORails instance was compiled with, so a test can say which one decides."""

    def __init__(
        self,
        *,
        mask: Optional[StubRail] = None,
        judge: Optional[StubRail] = None,
        input_rail: Optional[StubRail] = None,
        output_rail: Optional[StubRail] = None,
        call_mask: Optional[StubRail] = None,
        call_judge: Optional[StubRail] = None,
    ) -> None:
        self.mask = mask if mask is not None else rewriting_stub(MASKED_RESULT, SurfaceDirection.TOOL_RESULT)
        self.judge = judge if judge is not None else StubRail()
        self.input = input_rail if input_rail is not None else StubRail()
        self.output = output_rail if output_rail is not None else StubRail()
        if call_mask is None:
            call_mask = rewriting_stub(json.dumps(MASKED_ARGUMENTS), SurfaceDirection.TOOL_CALL)
        self.call_mask = call_mask
        self.call_judge = call_judge if call_judge is not None else StubRail()

    def compiled(self):
        return rails_compiled_as(
            {
                MASK_FLOW: self.mask,
                JUDGE_FLOW: self.judge,
                INPUT_FLOW: self.input,
                OUTPUT_FLOW: self.output,
                CALL_MASK_FLOW: self.call_mask,
                CALL_JUDGE_FLOW: self.call_judge,
            }
        )


@asynccontextmanager
async def tool_transform_iorails(rails: ToolTransformRails, config: dict) -> AsyncIterator[IORails]:
    with rails.compiled():
        async with started_iorails(config) as engine:
            yield engine


@asynccontextmanager
async def capturing_tool_transform_iorails(
    rails: ToolTransformRails, config: dict, exporter: InMemorySpanExporter
) -> AsyncIterator[IORails]:
    with rails.compiled():
        async with _capturing(config, exporter) as engine:
            yield engine


def _sent_messages(engine: IORails) -> list[dict]:
    """The messages the main model's transport was sent."""
    client = engine.engine_registry._get_engine("main", ModelEngine)._client
    return client.post.call_args.kwargs["json"]["messages"]


def with_tool_content(messages: list[dict], content: str) -> list[dict]:
    return [{**message, "content": content} if message.get("role") == "tool" else message for message in messages]


async def _drain(stream) -> list:
    return [chunk async for chunk in stream]


@pytest.fixture
def span_exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.mark.asyncio
class TestGenerateSendsTheRewrittenResult:
    """In generate, the rewritten tool result is what every later rail and the main model read."""

    async def test_the_model_receives_the_rewritten_result(self):
        """The rewrite reaches the wire, and the caller's conversation is left as it sent it."""
        messages = make_tool_conversation(content=RAW_RESULT)
        snapshot = copy.deepcopy(messages)
        async with tool_transform_iorails(ToolTransformRails(), tool_transform_config()) as engine:
            _inject_json_response(engine, _text_payload("It is 18C."))

            result = await engine.generate_async(messages=messages)
            sent = _sent_messages(engine)

        assert result == {"role": "assistant", "content": "It is 18C."}
        assert sent == with_tool_content(snapshot, MASKED_RESULT)
        assert messages == snapshot

    async def test_the_judge_and_the_input_and_output_rails_read_the_rewrite(self):
        """Rewriting runs first per tool, and the input and output rails are handed the rewritten history."""
        rails = ToolTransformRails()
        async with tool_transform_iorails(rails, tool_transform_config()) as engine:
            _inject_json_response(engine, _text_payload("It is 18C."))

            await engine.generate_async(messages=make_tool_conversation(content=RAW_RESULT))

        assert [r.content for r in rails.judge.seen_tool_results] == [MASKED_RESULT]
        assert rails.input.seen_messages[-1]["content"] == MASKED_RESULT
        assert rails.output.seen_messages[-1]["content"] == MASKED_RESULT

    async def test_an_earlier_turns_result_is_rewritten_again(self):
        """The engine keeps nothing between requests, so each request rewrites every result in its history."""
        messages = make_tool_conversation(content=RAW_RESULT) + [
            {"role": "assistant", "content": "It is 18C."},
            {"role": "user", "content": "Thanks. And tomorrow?"},
        ]
        async with tool_transform_iorails(ToolTransformRails(), tool_transform_config()) as engine:
            _inject_json_response(engine, _text_payload("Probably the same."))

            await engine.generate_async(messages=messages)
            sent = _sent_messages(engine)

        assert sent == with_tool_content(messages, MASKED_RESULT)

    async def test_a_config_that_never_rewrites_sends_the_conversation_as_it_came(self):
        """A judging rail alone changes nothing on the wire."""
        messages = make_tool_conversation(content=RAW_RESULT)
        async with tool_transform_iorails(ToolTransformRails(mask=StubRail()), tool_transform_config()) as engine:
            _inject_json_response(engine, _text_payload("It is 18C."))

            await engine.generate_async(messages=messages)
            sent = _sent_messages(engine)

        assert sent == messages


@pytest.mark.asyncio
class TestStreamSendsTheRewrittenResult:
    """In stream, the rewritten tool result is what the input rails and the main model read."""

    async def test_the_model_receives_the_rewritten_result(self):
        messages = make_tool_conversation(content=RAW_RESULT)
        rails = ToolTransformRails()
        async with tool_transform_iorails(rails, tool_transform_config(output=False)) as engine:
            _inject_sse_stream(engine, _text_sse_lines("It is 18C."))

            chunks = await _drain(engine.stream_async(messages=messages))
            sent = _sent_messages(engine)

        assert "".join(chunk for chunk in chunks if isinstance(chunk, str)) == "It is 18C."
        assert sent == with_tool_content(messages, MASKED_RESULT)
        assert rails.input.seen_messages[-1]["content"] == MASKED_RESULT


@pytest.mark.asyncio
class TestTheRequestSpanRecordsTheMask:
    """With content capture on, the request span records the tool result as the mask left it, block or not."""

    async def test_an_allowed_generation_records_the_masked_result(self, span_exporter):
        async with capturing_tool_transform_iorails(
            ToolTransformRails(), tool_transform_config(capture=True), span_exporter
        ) as engine:
            _inject_json_response(engine, _text_payload("It is 18C."))

            await engine.generate_async(messages=make_tool_conversation(content=RAW_RESULT))

        assert _captured_input(span_exporter)[-1]["content"] == MASKED_RESULT

    async def test_a_generation_blocked_by_a_tool_rail_behind_the_mask_records_the_masked_result(self, span_exporter):
        """The judge refuses after the mask ran, and the span still never carries the raw result."""
        rails = ToolTransformRails(judge=StubRail(RailOutcome.block(reason="unsafe")))
        async with capturing_tool_transform_iorails(
            rails, tool_transform_config(capture=True), span_exporter
        ) as engine:
            result = await engine.generate_async(messages=make_tool_conversation(content=RAW_RESULT))

        assert result == {"role": "assistant", "content": REFUSAL_MESSAGE}
        assert _captured_input(span_exporter)[-1]["content"] == MASKED_RESULT

    async def test_a_generation_blocked_by_an_input_rail_records_the_masked_result(self, span_exporter):
        rails = ToolTransformRails(input_rail=StubRail(RailOutcome.block(reason="unsafe")))
        async with capturing_tool_transform_iorails(
            rails, tool_transform_config(capture=True), span_exporter
        ) as engine:
            result = await engine.generate_async(messages=make_tool_conversation(content=RAW_RESULT))

        assert result == {"role": "assistant", "content": REFUSAL_MESSAGE}
        assert _captured_input(span_exporter)[-1]["content"] == MASKED_RESULT

    async def test_a_generation_blocked_by_an_output_rail_records_the_masked_result(self, span_exporter):
        rails = ToolTransformRails(output_rail=StubRail(RailOutcome.block(reason="unsafe")))
        async with capturing_tool_transform_iorails(
            rails, tool_transform_config(capture=True), span_exporter
        ) as engine:
            _inject_json_response(engine, _text_payload("It is 18C."))

            result = await engine.generate_async(messages=make_tool_conversation(content=RAW_RESULT))

        assert result == {"role": "assistant", "content": REFUSAL_MESSAGE}
        assert _captured_input(span_exporter)[-1]["content"] == MASKED_RESULT

    async def test_a_stream_blocked_by_a_tool_rail_behind_the_mask_records_the_masked_result(self, span_exporter):
        rails = ToolTransformRails(judge=StubRail(RailOutcome.block(reason="unsafe")))
        async with capturing_tool_transform_iorails(
            rails, tool_transform_config(output=False, capture=True), span_exporter
        ) as engine:
            chunks = await _drain(engine.stream_async(messages=make_tool_conversation(content=RAW_RESULT)))

        assert [v["error"]["param"] for v in _stream_violation_chunks(chunks)] == ["tool_result_rails"]
        assert _captured_input(span_exporter)[-1]["content"] == MASKED_RESULT

    async def test_a_stream_blocked_by_an_input_rail_records_the_masked_result(self, span_exporter):
        rails = ToolTransformRails(input_rail=StubRail(RailOutcome.block(reason="unsafe")))
        async with capturing_tool_transform_iorails(
            rails, tool_transform_config(output=False, capture=True), span_exporter
        ) as engine:
            chunks = await _drain(engine.stream_async(messages=make_tool_conversation(content=RAW_RESULT)))

        assert [v["error"]["param"] for v in _stream_violation_chunks(chunks)] == ["input_rails"]
        assert _captured_input(span_exporter)[-1]["content"] == MASKED_RESULT

    async def test_an_allowed_stream_records_the_masked_result(self, span_exporter):
        async with capturing_tool_transform_iorails(
            ToolTransformRails(), tool_transform_config(output=False, capture=True), span_exporter
        ) as engine:
            _inject_sse_stream(engine, _text_sse_lines("It is 18C."))

            await _drain(engine.stream_async(messages=make_tool_conversation(content=RAW_RESULT)))

        assert _captured_input(span_exporter)[-1]["content"] == MASKED_RESULT


def failing_mask(target: TransformTarget = TransformTarget.TOOL_RESULT) -> StubRail:
    """A masking rail whose provider is down, which ``CompiledRail`` turns into a fail-closed block."""
    return StubRail(RailOutcome.failure(reason="the masking provider is down"), target)


@pytest.mark.asyncio
class TestTheRequestSpanWithholdsAResultTheMaskNeverReached:
    """A configured mask that never decided leaves the result off the span rather than recording it as sent."""

    async def test_a_failed_mask_withholds_the_result(self, span_exporter):
        """The mask raised, so the conversation still holds the raw result and the span records none of it."""
        async with capturing_tool_transform_iorails(
            ToolTransformRails(mask=failing_mask()), tool_transform_config(capture=True), span_exporter
        ) as engine:
            result = await engine.generate_async(messages=make_tool_conversation(content=RAW_RESULT))

        assert result["content"] != RAW_RESULT
        assert _has_captured_input(span_exporter) is False

    async def test_a_result_the_manager_cannot_link_withholds_the_result(self, span_exporter):
        """The block lands before any per-tool rail runs, so the mask never saw the result the request carries."""
        messages = make_tool_conversation(result_call_id="call_other", content=RAW_RESULT)
        async with capturing_tool_transform_iorails(
            ToolTransformRails(), tool_transform_config(capture=True), span_exporter
        ) as engine:
            result = await engine.generate_async(messages=messages)

        assert result == {"role": "assistant", "content": REFUSAL_MESSAGE}
        assert _has_captured_input(span_exporter) is False

    async def test_tool_rails_turned_off_for_the_request_withhold_the_result(self, span_exporter):
        """The generation options turned the mask off, so the model read the raw result and the span records neither."""
        async with capturing_tool_transform_iorails(
            ToolTransformRails(), tool_transform_config(capture=True), span_exporter
        ) as engine:
            _inject_json_response(engine, _text_payload("It is 18C."))

            await engine.generate_async(
                messages=make_tool_conversation(content=RAW_RESULT), options={"rails": {"tool_result": False}}
            )

        assert _has_captured_input(span_exporter) is False
        assert _has_captured_output(span_exporter) is False

    async def test_a_check_that_leaves_tool_results_out_withholds_the_result(self, span_exporter):
        """``rail_types`` ran the input rails alone, so no mask decided on the result the conversation carries."""
        async with capturing_tool_transform_iorails(
            ToolTransformRails(), tool_transform_config(capture=True), span_exporter
        ) as engine:
            await engine.check_async(make_tool_conversation(content=RAW_RESULT), rail_types=[RailType.INPUT])

        assert _has_captured_input(span_exporter) is False

    async def test_a_stream_with_the_tool_rails_turned_off_withholds_the_result(self, span_exporter):
        """The streamed request reaches the same verdict as the generated one: nothing masked, nothing recorded."""
        async with capturing_tool_transform_iorails(
            ToolTransformRails(), tool_transform_config(output=False, capture=True), span_exporter
        ) as engine:
            _inject_sse_stream(engine, _text_sse_lines("It is 18C."))

            await _drain(
                engine.stream_async(
                    messages=make_tool_conversation(content=RAW_RESULT), options={"rails": {"tool_result": False}}
                )
            )

        assert _has_captured_input(span_exporter) is False
        assert _has_captured_output(span_exporter) is False

    async def test_a_stream_closed_before_its_input_rails_finish_records_the_masked_result(self, span_exporter):
        """The caller left while the input rails ran, and the mask had already reached what the span records."""
        input_rails_reached = asyncio.Event()

        async def input_rails_that_never_finish(messages, **kwargs):
            input_rails_reached.set()
            await asyncio.Event().wait()

        async with capturing_tool_transform_iorails(
            ToolTransformRails(), tool_transform_config(output=False, capture=True), span_exporter
        ) as engine:
            engine.rails_manager.is_input_safe = input_rails_that_never_finish
            stream = engine.stream_async(messages=make_tool_conversation(content=RAW_RESULT))
            first_chunk = asyncio.create_task(anext(stream))
            await input_rails_reached.wait()
            first_chunk.cancel()
            with suppress(asyncio.CancelledError):
                await first_chunk

        assert _captured_input(span_exporter)[-1]["content"] == MASKED_RESULT


WEATHER_QUESTION = [{"role": "user", "content": "What's the weather in Paris?"}]


def wire_weather_call(arguments: dict) -> dict:
    return {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": json.dumps(arguments)}}


def _terminal_tool_calls(chunks: list) -> list[dict]:
    """The calls on a stream's terminal tool-call chunk, or none when it sent none."""
    for chunk in chunks:
        if isinstance(chunk, str) and chunk.startswith('{"tool_calls"'):
            return json.loads(chunk)["tool_calls"]
    return []


@pytest.mark.asyncio
class TestGenerateReturnsTheRewrittenCall:
    """In generate, the rewritten arguments are what the later rails read and what the caller gets back."""

    async def test_the_bare_message_carries_the_rewritten_arguments_as_a_json_string(self):
        rails = ToolTransformRails()
        async with tool_transform_iorails(rails, tool_call_transform_config()) as engine:
            _inject_json_response(engine, _tool_call_payload("get_weather", json.dumps(RAW_ARGUMENTS)))

            result = await engine.generate_async(messages=WEATHER_QUESTION)

        assert result == {"role": "assistant", "content": None, "tool_calls": [wire_weather_call(MASKED_ARGUMENTS)]}
        assert [c.function.arguments for c in rails.call_judge.seen_tool_calls] == [MASKED_ARGUMENTS]

    async def test_the_generation_response_carries_the_rewritten_arguments(self):
        async with tool_transform_iorails(ToolTransformRails(), tool_call_transform_config()) as engine:
            _inject_json_response(engine, _tool_call_payload("get_weather", json.dumps(RAW_ARGUMENTS)))

            result = await engine.generate_async(messages=WEATHER_QUESTION, options={"llm_params": {"temperature": 0}})

        assert result.tool_calls == [
            {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": MASKED_ARGUMENTS}}
        ]

    async def test_a_rewrite_that_breaks_the_tool_schema_is_refused(self):
        rails = ToolTransformRails(call_mask=rewriting_stub('{"city": 5}', SurfaceDirection.TOOL_CALL))
        async with tool_transform_iorails(rails, tool_call_transform_config()) as engine:
            _inject_json_response(engine, _tool_call_payload("get_weather", json.dumps(RAW_ARGUMENTS)))

            result = await engine.generate_async(messages=WEATHER_QUESTION)

        assert result == {"role": "assistant", "content": REFUSAL_MESSAGE}

    async def test_a_config_that_never_rewrites_returns_the_call_as_the_model_made_it(self):
        async with tool_transform_iorails(
            ToolTransformRails(call_mask=StubRail()), tool_call_transform_config()
        ) as engine:
            _inject_json_response(engine, _tool_call_payload("get_weather", json.dumps(RAW_ARGUMENTS)))

            result = await engine.generate_async(messages=WEATHER_QUESTION)

        assert result["tool_calls"] == [wire_weather_call(RAW_ARGUMENTS)]


@pytest.mark.asyncio
class TestStreamReturnsTheRewrittenCall:
    """In stream, the terminal tool-call chunk carries the rewritten arguments."""

    async def test_the_terminal_chunk_carries_the_rewritten_arguments(self):
        async with tool_transform_iorails(ToolTransformRails(), tool_call_transform_config(output=False)) as engine:
            _inject_sse_stream(engine, _tool_call_sse_lines("get_weather", [json.dumps(RAW_ARGUMENTS)]))

            chunks = await _drain(engine.stream_async(messages=WEATHER_QUESTION))

        assert _terminal_tool_calls(chunks) == [wire_weather_call(MASKED_ARGUMENTS)]

    async def test_the_request_span_records_the_rewritten_terminal_chunk(self, span_exporter):
        async with capturing_tool_transform_iorails(
            ToolTransformRails(), tool_call_transform_config(output=False, capture=True), span_exporter
        ) as engine:
            _inject_sse_stream(engine, _tool_call_sse_lines("get_weather", [json.dumps(RAW_ARGUMENTS)]))

            await _drain(engine.stream_async(messages=WEATHER_QUESTION))

        captured = _captured_output(span_exporter)
        assert "<EMAIL>" in captured
        assert "ada@example.com" not in captured

    async def test_a_rewrite_that_breaks_the_tool_schema_sends_a_violation_and_no_call(self):
        rails = ToolTransformRails(call_mask=rewriting_stub('{"city": 5}', SurfaceDirection.TOOL_CALL))
        async with tool_transform_iorails(rails, tool_call_transform_config(output=False)) as engine:
            _inject_sse_stream(engine, _tool_call_sse_lines("get_weather", [json.dumps(RAW_ARGUMENTS)]))

            chunks = await _drain(engine.stream_async(messages=WEATHER_QUESTION))

        assert [v["error"]["param"] for v in _stream_violation_chunks(chunks)] == ["tool_call_rails"]
        assert _terminal_tool_calls(chunks) == []


def assistant_weather_call(arguments: dict) -> list[dict]:
    """A user question answered by an assistant turn that calls ``get_weather`` with *arguments*."""
    return WEATHER_QUESTION + [{"role": "assistant", "content": None, "tool_calls": [wire_weather_call(arguments)]}]


@pytest.mark.asyncio
class TestTheRequestSpanWithholdsACallTheMaskNeverReached:
    """A configured call mask that never decided leaves the arguments off the span rather than recording them."""

    async def test_a_stream_with_the_tool_call_rails_turned_off_withholds_the_output(self, span_exporter):
        """The options turned the mask off, so the terminal chunk carries the raw arguments and the span records none."""
        async with capturing_tool_transform_iorails(
            ToolTransformRails(), tool_call_transform_config(output=False, capture=True), span_exporter
        ) as engine:
            _inject_sse_stream(engine, _tool_call_sse_lines("get_weather", [json.dumps(RAW_ARGUMENTS)]))

            chunks = await _drain(
                engine.stream_async(messages=WEATHER_QUESTION, options={"rails": {"tool_call": False}})
            )

        assert _terminal_tool_calls(chunks) == [wire_weather_call(RAW_ARGUMENTS)]
        assert _has_captured_output(span_exporter) is False

    async def test_a_check_that_leaves_tool_calls_out_withholds_the_conversation(self, span_exporter):
        """``rail_types`` ran the input rails alone, so no mask decided on the call the conversation carries."""
        async with capturing_tool_transform_iorails(
            ToolTransformRails(), tool_call_transform_config(capture=True), span_exporter
        ) as engine:
            await engine.check_async(assistant_weather_call(RAW_ARGUMENTS), rail_types=[RailType.INPUT])

        assert _has_captured_input(span_exporter) is False

    async def test_a_failed_call_mask_withholds_the_checked_conversation(self, span_exporter):
        """The mask raised, so the conversation still holds the raw arguments and the span records none of them."""
        rails = ToolTransformRails(call_mask=failing_mask(TransformTarget.TOOL_CALL))
        async with capturing_tool_transform_iorails(
            rails, tool_call_transform_config(capture=True), span_exporter
        ) as engine:
            result = await engine.check_async(assistant_weather_call(RAW_ARGUMENTS), rail_types=[RailType.TOOL_CALL])

        assert result.status.value == "blocked"
        assert _has_captured_input(span_exporter) is False

    async def test_a_call_the_extractor_cannot_parse_withholds_the_checked_conversation(self, span_exporter):
        """No mask can decide on a call nobody could parse, and the conversation still holds every call as sent."""
        messages = WEATHER_QUESTION + [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{not json"}}
                ],
            }
        ]
        async with capturing_tool_transform_iorails(
            ToolTransformRails(), tool_call_transform_config(capture=True), span_exporter
        ) as engine:
            result = await engine.check_async(messages, rail_types=[RailType.TOOL_CALL])

        assert result.status.value == "blocked"
        assert _has_captured_input(span_exporter) is False

    async def test_a_turn_the_extractor_cannot_read_withholds_the_checked_conversation(self, span_exporter):
        """A legacy ``function_call`` fails extraction for the whole message, so no mask decided on any call on it."""
        messages = assistant_weather_call(RAW_ARGUMENTS)
        messages[-1] = {**messages[-1], "function_call": {"name": "get_weather", "arguments": "{}"}}
        async with capturing_tool_transform_iorails(
            ToolTransformRails(), tool_call_transform_config(capture=True), span_exporter
        ) as engine:
            result = await engine.check_async(messages, rail_types=[RailType.TOOL_CALL])

        assert result.status.value == "blocked"
        assert _has_captured_input(span_exporter) is False
