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

"""A per-tool rail rewriting a tool result, driven end to end through IORails.

The rails are stubs compiled in place of the configured flows, and only the main model's
transport is mocked, so what is asserted is what the model was sent, what the other rails were
handed and what the request span recorded.
"""

import copy
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from nemoguardrails.actions.rail_outcome import RailOutcome
from nemoguardrails.guardrails.iorails import REFUSAL_MESSAGE, IORails
from nemoguardrails.guardrails.model_engine import ModelEngine
from nemoguardrails.manifests import RailDirection as SurfaceDirection
from tests.guardrails.async_helpers import started_iorails
from tests.guardrails.rail_stubs import StubRail, rails_compiled_as, rewriting_stub
from tests.guardrails.test_data import CONTENT_SAFETY_CONFIG
from tests.guardrails.test_tool_rails_iorails import (
    _inject_json_response,
    _inject_sse_stream,
    _stream_violation_chunks,
    _text_payload,
    _text_sse_lines,
)
from tests.guardrails.test_transform_rail_pipeline import CAPTURE_CONTENT, _captured_input, _capturing
from tests.guardrails.tool_helpers import make_tool_conversation

INPUT_FLOW = "content safety check input $model=content_safety"
OUTPUT_FLOW = "content safety check output $model=content_safety"
MASK_FLOW = "cel check tool result"
JUDGE_FLOW = "regex check tool result"

RAW_RESULT = "18C, reported by ada@example.com"
MASKED_RESULT = "18C, reported by <EMAIL>"


def _config(*, output: bool = True, capture: bool = False) -> dict:
    """Content safety on input (and output), and a masking rail ahead of a judge on get_weather's results."""
    config = copy.deepcopy(CONTENT_SAFETY_CONFIG)
    if not output:
        config["rails"]["output"] = {"flows": []}
    config["rails"]["tool_result"] = {"per_tool": {"get_weather": [JUDGE_FLOW, MASK_FLOW]}}
    if capture:
        config["tracing"] = dict(CAPTURE_CONTENT)
    return config


class Rails:
    """The stubs one IORails instance was compiled with, so a test can say which one decides."""

    def __init__(
        self,
        *,
        mask: Optional[StubRail] = None,
        judge: Optional[StubRail] = None,
        input_rail: Optional[StubRail] = None,
        output_rail: Optional[StubRail] = None,
    ) -> None:
        self.mask = mask if mask is not None else rewriting_stub(MASKED_RESULT, SurfaceDirection.TOOL_RESULT)
        self.judge = judge if judge is not None else StubRail()
        self.input = input_rail if input_rail is not None else StubRail()
        self.output = output_rail if output_rail is not None else StubRail()

    def compiled(self):
        return rails_compiled_as(
            {MASK_FLOW: self.mask, JUDGE_FLOW: self.judge, INPUT_FLOW: self.input, OUTPUT_FLOW: self.output}
        )


@asynccontextmanager
async def _iorails(rails: Rails, config: dict) -> AsyncIterator[IORails]:
    with rails.compiled():
        async with started_iorails(config) as engine:
            yield engine


@asynccontextmanager
async def _capturing_iorails(rails: Rails, config: dict, exporter: InMemorySpanExporter) -> AsyncIterator[IORails]:
    with rails.compiled():
        async with _capturing(config, exporter) as engine:
            yield engine


def _sent_messages(engine: IORails) -> list[dict]:
    """The messages the main model's transport was sent."""
    client = engine.engine_registry._get_engine("main", ModelEngine)._client
    return client.post.call_args.kwargs["json"]["messages"]


def _with_tool_content(messages: list[dict], content: str) -> list[dict]:
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
        async with _iorails(Rails(), _config()) as engine:
            _inject_json_response(engine, _text_payload("It is 18C."))

            result = await engine.generate_async(messages=messages)
            sent = _sent_messages(engine)

        assert result == {"role": "assistant", "content": "It is 18C."}
        assert sent == _with_tool_content(snapshot, MASKED_RESULT)
        assert messages == snapshot

    async def test_the_judge_and_the_input_and_output_rails_read_the_rewrite(self):
        """Rewriting runs first per tool, and the input and output rails are handed the rewritten history."""
        rails = Rails()
        async with _iorails(rails, _config()) as engine:
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
        async with _iorails(Rails(), _config()) as engine:
            _inject_json_response(engine, _text_payload("Probably the same."))

            await engine.generate_async(messages=messages)
            sent = _sent_messages(engine)

        assert sent == _with_tool_content(messages, MASKED_RESULT)

    async def test_a_config_that_never_rewrites_sends_the_conversation_as_it_came(self):
        """A judging rail alone changes nothing on the wire."""
        messages = make_tool_conversation(content=RAW_RESULT)
        async with _iorails(Rails(mask=StubRail()), _config()) as engine:
            _inject_json_response(engine, _text_payload("It is 18C."))

            await engine.generate_async(messages=messages)
            sent = _sent_messages(engine)

        assert sent == messages


@pytest.mark.asyncio
class TestStreamSendsTheRewrittenResult:
    """In stream, the rewritten tool result is what the input rails and the main model read."""

    async def test_the_model_receives_the_rewritten_result(self):
        messages = make_tool_conversation(content=RAW_RESULT)
        rails = Rails()
        async with _iorails(rails, _config(output=False)) as engine:
            _inject_sse_stream(engine, _text_sse_lines("It is 18C."))

            chunks = await _drain(engine.stream_async(messages=messages))
            sent = _sent_messages(engine)

        assert "".join(chunk for chunk in chunks if isinstance(chunk, str)) == "It is 18C."
        assert sent == _with_tool_content(messages, MASKED_RESULT)
        assert rails.input.seen_messages[-1]["content"] == MASKED_RESULT


@pytest.mark.asyncio
class TestTheRequestSpanRecordsTheMask:
    """With content capture on, the request span records the tool result as the mask left it, block or not."""

    async def test_an_allowed_generation_records_the_masked_result(self, span_exporter):
        async with _capturing_iorails(Rails(), _config(capture=True), span_exporter) as engine:
            _inject_json_response(engine, _text_payload("It is 18C."))

            await engine.generate_async(messages=make_tool_conversation(content=RAW_RESULT))

        assert _captured_input(span_exporter)[-1]["content"] == MASKED_RESULT

    async def test_a_generation_blocked_by_a_tool_rail_behind_the_mask_records_the_masked_result(self, span_exporter):
        """The judge refuses after the mask ran, and the span still never carries the raw result."""
        rails = Rails(judge=StubRail(RailOutcome.block(reason="unsafe")))
        async with _capturing_iorails(rails, _config(capture=True), span_exporter) as engine:
            result = await engine.generate_async(messages=make_tool_conversation(content=RAW_RESULT))

        assert result == {"role": "assistant", "content": REFUSAL_MESSAGE}
        assert _captured_input(span_exporter)[-1]["content"] == MASKED_RESULT

    async def test_a_generation_blocked_by_an_input_rail_records_the_masked_result(self, span_exporter):
        rails = Rails(input_rail=StubRail(RailOutcome.block(reason="unsafe")))
        async with _capturing_iorails(rails, _config(capture=True), span_exporter) as engine:
            result = await engine.generate_async(messages=make_tool_conversation(content=RAW_RESULT))

        assert result == {"role": "assistant", "content": REFUSAL_MESSAGE}
        assert _captured_input(span_exporter)[-1]["content"] == MASKED_RESULT

    async def test_a_generation_blocked_by_an_output_rail_records_the_masked_result(self, span_exporter):
        rails = Rails(output_rail=StubRail(RailOutcome.block(reason="unsafe")))
        async with _capturing_iorails(rails, _config(capture=True), span_exporter) as engine:
            _inject_json_response(engine, _text_payload("It is 18C."))

            result = await engine.generate_async(messages=make_tool_conversation(content=RAW_RESULT))

        assert result == {"role": "assistant", "content": REFUSAL_MESSAGE}
        assert _captured_input(span_exporter)[-1]["content"] == MASKED_RESULT

    async def test_a_stream_blocked_by_a_tool_rail_behind_the_mask_records_the_masked_result(self, span_exporter):
        rails = Rails(judge=StubRail(RailOutcome.block(reason="unsafe")))
        async with _capturing_iorails(rails, _config(output=False, capture=True), span_exporter) as engine:
            chunks = await _drain(engine.stream_async(messages=make_tool_conversation(content=RAW_RESULT)))

        assert [v["error"]["param"] for v in _stream_violation_chunks(chunks)] == ["tool_result_rails"]
        assert _captured_input(span_exporter)[-1]["content"] == MASKED_RESULT

    async def test_a_stream_blocked_by_an_input_rail_records_the_masked_result(self, span_exporter):
        rails = Rails(input_rail=StubRail(RailOutcome.block(reason="unsafe")))
        async with _capturing_iorails(rails, _config(output=False, capture=True), span_exporter) as engine:
            chunks = await _drain(engine.stream_async(messages=make_tool_conversation(content=RAW_RESULT)))

        assert [v["error"]["param"] for v in _stream_violation_chunks(chunks)] == ["input_rails"]
        assert _captured_input(span_exporter)[-1]["content"] == MASKED_RESULT

    async def test_an_allowed_stream_records_the_masked_result(self, span_exporter):
        async with _capturing_iorails(Rails(), _config(output=False, capture=True), span_exporter) as engine:
            _inject_sse_stream(engine, _text_sse_lines("It is 18C."))

            await _drain(engine.stream_async(messages=make_tool_conversation(content=RAW_RESULT)))

        assert _captured_input(span_exporter)[-1]["content"] == MASKED_RESULT
