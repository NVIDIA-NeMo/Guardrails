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

import json

import pytest

from nemoguardrails.server.experimental._buffered_kernel import (
    OperationBlocked,
    OperationCheckFailed,
    OperationModificationUnsupported,
    OperationProjectionFailed,
)
from nemoguardrails.server.experimental._content_checker import (
    ContentAllowed,
    ContentBlocked,
    ContentCheckFailed,
    StreamBufferingPolicy,
)
from nemoguardrails.server.experimental._guarded_stream import (
    StreamProcessingFailed,
    StreamUpstreamFailed,
    UnsupportedStreamInspection,
    guard_provider_stream,
    validate_streaming_policy,
)
from nemoguardrails.server.experimental._http_kernel import BufferedHttpResponse
from nemoguardrails.server.experimental._json_payload import parse_json_object
from nemoguardrails.server.experimental.provider.stream import (
    ClassifiedStreamAdapter,
    GuardedStreamEvent,
    StreamCapabilityProfile,
    StreamEventRole,
    StreamProjectionContract,
    StreamShapeCoverage,
    UnsupportedProviderStream,
)
from nemoguardrails.server.experimental.provider.types import GuardedMessage

_CONTRACT = StreamProjectionContract(
    projection_id="test.stream",
    profile=StreamCapabilityProfile.SINGLE_TEXT_DELTA_V1,
    shapes=StreamShapeCoverage(
        guarded_shapes=frozenset({"text"}),
        snapshot_shapes=frozenset({"snapshot"}),
        opaque_shapes=frozenset({"metadata", "end"}),
        provider_error_shapes=frozenset({"error"}),
    ),
)


class StaticChecker:
    def __init__(self, decisions=()):
        self.decisions = list(decisions)
        self.calls = []

    async def check_output(self, check):
        self.calls.append(check)
        decision = self.decisions.pop(0) if self.decisions else ContentAllowed()
        if isinstance(decision, Exception):
            raise decision
        return decision


class Classifier:
    contract = _CONTRACT

    def classify_event(self, event):
        if event.data == b"[END]":
            return GuardedStreamEvent("end", StreamEventRole.OPAQUE_METADATA)
        try:
            payload = parse_json_object(event.data or b"")
        except ValueError as error:
            raise UnsupportedProviderStream("unsupported fake stream event") from error
        if "error" in payload:
            return GuardedStreamEvent("error", StreamEventRole.PROVIDER_ERROR)
        if "snapshot" in payload:
            return GuardedStreamEvent("snapshot", StreamEventRole.TEXT_SNAPSHOT, payload["snapshot"])
        if "text" in payload:
            return GuardedStreamEvent("text", StreamEventRole.GUARDED_TEXT, payload["text"])
        return GuardedStreamEvent("metadata", StreamEventRole.OPAQUE_METADATA)


class Hooks:
    def __init__(self):
        self.observed = []
        self.finished = False
        self.encoded = []

    def observe_event(self, event, classified):
        self.observed.append((event, classified))

    def validate_end_of_stream(self):
        self.finished = True

    def encode_error(self, rendered_body):
        self.encoded.append(rendered_body)
        return (b"data: " + rendered_body + b"\n\n", b"data: [END]\n\n")


class Source:
    def __init__(self, chunks, failure=None):
        self.chunks = iter(chunks)
        self.failure = failure
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self.chunks)
        except StopIteration:
            if self.failure is not None:
                failure, self.failure = self.failure, None
                raise failure
            raise StopAsyncIteration

    async def aclose(self):
        self.closed = True


def render_outcome(outcome):
    if isinstance(outcome, OperationBlocked):
        code = "blocked"
    elif isinstance(outcome, OperationCheckFailed):
        code = "check_failed"
    elif isinstance(outcome, OperationModificationUnsupported):
        code = "modification_unsupported"
    elif isinstance(outcome, OperationProjectionFailed):
        code = "unsupported_stream"
    elif isinstance(outcome, StreamUpstreamFailed):
        code = "upstream_failed"
    elif isinstance(outcome, StreamProcessingFailed):
        code = "processing_failed"
    else:
        raise AssertionError("unknown outcome")
    return BufferedHttpResponse(200, (), json.dumps({"code": code}).encode())


def stream_event(payload):
    return b"data: " + json.dumps(payload, separators=(",", ":")).encode() + b"\n\n"


async def collect(source, checker, hooks, policy=StreamBufferingPolicy(1, 0), **limits):
    return b"".join(
        [
            chunk
            async for chunk in guard_provider_stream(
                source,
                checker=checker,
                streaming_policy=policy,
                input_message=GuardedMessage("user", "question"),
                adapter=ClassifiedStreamAdapter(Classifier(), hooks),
                render_outcome=render_outcome,
                **limits,
            )
        ]
    )


@pytest.mark.asyncio
async def test_guarded_stream_checks_windows_before_releasing_original_events():
    events = [
        stream_event({"meta": 1}),
        stream_event({"text": "one "}),
        stream_event({"text": "two "}),
        stream_event({"text": "three"}),
        b"data: [END]\n\n",
    ]
    source = Source([b"".join(events)[:23], b"".join(events)[23:]])
    checker = StaticChecker()
    hooks = Hooks()

    result = await collect(source, checker, hooks, StreamBufferingPolicy(chunk_size=2, context_size=1))

    assert result == b"".join(events)
    assert [call.output_content for call in checker.calls] == ["one two ", "two three"]
    assert hooks.finished is True
    assert source.closed is True


@pytest.mark.asyncio
async def test_zero_context_does_not_retain_the_previous_window():
    events = [stream_event({"text": text}) for text in ("one ", "two ", "three ", "four")]
    checker = StaticChecker()

    result = await collect(
        Source([b"".join(events)]),
        checker,
        Hooks(),
        StreamBufferingPolicy(chunk_size=2, context_size=0),
    )

    assert result == b"".join(events)
    assert [call.output_content for call in checker.calls] == ["one two ", "three four"]


@pytest.mark.asyncio
async def test_stream_block_suppresses_the_complete_pending_window():
    first = stream_event({"text": "unsafe "})
    second = stream_event({"text": "content"})
    checker = StaticChecker([ContentBlocked("blocked", "rule")])

    result = await collect(
        Source([first + second + b"data: [END]\n\n"]),
        checker,
        Hooks(),
        StreamBufferingPolicy(2, 0),
    )

    assert first not in result
    assert second not in result
    assert b'"code": "blocked"' in result
    assert result.endswith(b"data: [END]\n\n")


@pytest.mark.asyncio
@pytest.mark.parametrize("decision", [ContentCheckFailed("failed"), RuntimeError("checker failed")])
async def test_stream_check_failure_is_distinct_from_provider_drift(decision):
    result = await collect(Source([stream_event({"text": "answer"})]), StaticChecker([decision]), Hooks())

    assert b'"code": "check_failed"' in result
    assert b"unsupported_stream" not in result


@pytest.mark.asyncio
async def test_stream_replacement_is_rejected_without_releasing_pending_text():
    event = stream_event({"text": "answer"})

    result = await collect(Source([event]), StaticChecker([ContentAllowed(replacement="changed")]), Hooks())

    assert event not in result
    assert b'"code": "modification_unsupported"' in result


@pytest.mark.asyncio
async def test_provider_error_events_are_preserved_without_being_checked_as_text():
    error = stream_event({"error": {"message": "upstream failed"}})
    terminal = b"data: [END]\n\n"
    checker = StaticChecker()

    result = await collect(Source([error + terminal]), checker, Hooks(), StreamBufferingPolicy(2, 0))

    assert result == error + terminal
    assert checker.calls == []


@pytest.mark.asyncio
async def test_snapshot_must_equal_the_guarded_delta_history():
    source = Source([stream_event({"text": "answer"}) + stream_event({"snapshot": "different"})])

    result = await collect(source, StaticChecker(), Hooks())

    assert b'"code": "unsupported_stream"' in result


@pytest.mark.asyncio
async def test_unsupported_event_and_pending_limit_fail_closed():
    unsupported = await collect(Source([b"data: not-json\n\n"]), StaticChecker(), Hooks())
    oversized = await collect(
        Source([stream_event({"meta": "long"})]),
        StaticChecker(),
        Hooks(),
        max_pending_bytes=5,
    )

    assert b'"code": "unsupported_stream"' in unsupported
    assert b'"code": "unsupported_stream"' in oversized


@pytest.mark.asyncio
async def test_manual_adapter_contract_mismatch_fails_closed():
    class MismatchedClassifier(Classifier):
        def classify_event(self, _event):
            return GuardedStreamEvent("text", StreamEventRole.OPAQUE_METADATA)

    hooks = Hooks()
    result = b"".join(
        [
            chunk
            async for chunk in guard_provider_stream(
                Source([stream_event({"text": "answer"})]),
                checker=StaticChecker(),
                streaming_policy=StreamBufferingPolicy(1, 0),
                input_message=GuardedMessage("user", "question"),
                adapter=ClassifiedStreamAdapter(MismatchedClassifier(), hooks),
                render_outcome=render_outcome,
            )
        ]
    )

    assert b'"code": "unsupported_stream"' in result
    assert hooks.finished is False


@pytest.mark.asyncio
async def test_uninspected_stream_is_relayed_by_chunk_without_parsing():
    chunks = [b"not sse", b" and still opaque"]
    source = Source(chunks)

    result = await collect(source, StaticChecker(), Hooks(), policy=None)

    assert result == b"".join(chunks)
    assert source.closed is True


@pytest.mark.asyncio
async def test_cancelling_downstream_iteration_closes_the_upstream_source():
    source = Source([b"first", b"second"])
    stream = guard_provider_stream(
        source,
        checker=StaticChecker(),
        streaming_policy=None,
        input_message=GuardedMessage("user", "question"),
        adapter=ClassifiedStreamAdapter(Classifier(), Hooks()),
        render_outcome=render_outcome,
    )

    assert await anext(stream) == b"first"
    await stream.aclose()

    assert source.closed is True


@pytest.mark.asyncio
async def test_upstream_failure_is_distinct_from_processing_failure():
    source = Source([], RuntimeError("upstream disconnected"))

    result = await collect(source, StaticChecker(), Hooks())

    assert b'"code": "upstream_failed"' in result
    assert source.closed is True


def test_stream_policy_must_buffer_before_release():
    with pytest.raises(UnsupportedStreamInspection, match="requires a buffering policy"):
        validate_streaming_policy(None)

    with pytest.raises(UnsupportedStreamInspection, match="cannot release"):
        validate_streaming_policy(StreamBufferingPolicy(1, 0, True))
