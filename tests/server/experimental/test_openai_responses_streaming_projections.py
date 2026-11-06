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

"""Exercise Responses event policy and per-stream protocol state offline."""

import json

import pytest

from nemoguardrails.server.experimental.provider.sse import ServerSentEvent
from nemoguardrails.server.experimental.provider.stream import StreamEventRole, UnsupportedProviderStream
from nemoguardrails.server.experimental.providers.openai.responses.endpoint import RESPONSES_ENDPOINT
from nemoguardrails.server.experimental.providers.openai.responses.stream_classifier import STREAM_CLASSIFIER
from tests.server.experimental.test_openai_responses_projections import response_payload


def event_bytes(event_type, **fields):
    """Serialize synthetic API-shaped events; these are not provider recordings."""
    return (
        b"event: "
        + event_type.encode()
        + b"\ndata: "
        + json.dumps({"type": event_type, **fields}, separators=(",", ":")).encode()
        + b"\n\n"
    )


def event(event_type, **fields):
    return ServerSentEvent.from_bytes(event_bytes(event_type, **fields))


def adapter():
    return RESPONSES_ENDPOINT.stream.create_adapter()


@pytest.mark.parametrize(
    "native,fields,role,text",
    [
        ("response.output_text.delta", {"delta": "answer"}, StreamEventRole.GUARDED_TEXT, "answer"),
        ("response.output_text.done", {"text": "answer"}, StreamEventRole.TEXT_SNAPSHOT, "answer"),
        ("response.created", {"response": {"output": []}}, StreamEventRole.TEXT_SNAPSHOT, ""),
        ("response.completed", {"response": response_payload()}, StreamEventRole.TEXT_SNAPSHOT, "answer"),
        (
            "response.output_item.added",
            {"item": {"type": "message", "role": "assistant", "content": []}},
            StreamEventRole.TEXT_SNAPSHOT,
            "",
        ),
        (
            "response.output_item.done",
            {"item": {"type": "reasoning", "summary": []}},
            StreamEventRole.OPAQUE_METADATA,
            None,
        ),
        ("error", {"message": "failed", "code": "provider_error"}, StreamEventRole.PROVIDER_ERROR, None),
    ],
)
def test_classifier_assigns_text_and_metadata_roles(native, fields, role, text):
    classified = STREAM_CLASSIFIER.classify_event(event(native, **fields))
    assert classified.role is role
    assert classified.text == text


@pytest.mark.parametrize(
    "native,fields",
    [
        ("response.refusal.delta", {"delta": "hidden"}),
        ("response.function_call_arguments.delta", {"delta": "hidden"}),
        ("response.reasoning_summary_text.delta", {"delta": "hidden"}),
        ("response.output_text.annotation.added", {"annotation": {"title": "hidden"}}),
        ("response.output_text.delta", {"delta": "answer", "content_index": 1}),
        ("response.output_text.delta", {"delta": "answer", "content_index": False}),
        ("response.output_text.delta", {"delta": "answer", "output_index": -1}),
        ("response.output_text.delta", {"delta": "answer", "sequence_number": "1"}),
        ("response.output_text.delta", {"delta": "answer", "logprobs": [{"token": "hidden"}]}),
        ("response.output_text.done", {"text": "answer", "logprobs": [{"token": "hidden"}]}),
        (
            "response.content_part.done",
            {"part": {"type": "output_text", "text": "answer", "annotations": [{"title": "hidden"}]}},
        ),
        ("response.output_item.done", {"item": {"type": "reasoning", "summary": [{"text": "hidden"}]}}),
        ("response.output_text.delta", {"delta": "answer", "future_text": "hidden"}),
    ],
)
def test_unreviewed_content_and_invalid_index_fields_fail_closed(native, fields):
    with pytest.raises(UnsupportedProviderStream):
        STREAM_CLASSIFIER.classify_event(event(native, **fields))


@pytest.mark.parametrize("terminal", ["response.completed", "response.failed", "response.incomplete", "error"])
@pytest.mark.parametrize("sentinel", [False, True])
def test_native_terminal_events_finish_streams_without_requiring_done(terminal, sentinel):
    stream = adapter()
    fields = {"message": "failed"} if terminal == "error" else {"response": {"output": []}}
    stream.classify_event(event(terminal, **fields))
    if sentinel:
        stream.classify_event(ServerSentEvent.from_bytes(b"data: [DONE]\n\n"))
    stream.classify_event(ServerSentEvent.from_bytes(b": keepalive\n\n"))
    stream.validate_end_of_stream()
    with pytest.raises(UnsupportedProviderStream):
        stream.classify_event(event("response.output_text.delta", delta="late"))


def test_done_cannot_replace_native_completion_or_repeat():
    stream = adapter()
    done = ServerSentEvent.from_bytes(b"data: [DONE]\n\n")
    with pytest.raises(UnsupportedProviderStream):
        stream.classify_event(done)
    with pytest.raises(UnsupportedProviderStream):
        stream.validate_end_of_stream()
    stream.classify_event(event("error", message="failed"))
    stream.classify_event(done)
    with pytest.raises(UnsupportedProviderStream):
        stream.classify_event(done)


@pytest.mark.parametrize("change", [{"item_id": "other"}, {"output_index": 2}, {"sequence_number": 1}])
def test_supplied_identity_and_sequence_cannot_change_or_regress(change):
    stream = adapter()
    stream.classify_event(
        event("response.output_text.delta", delta="a", item_id="msg_test", output_index=1, sequence_number=1)
    )
    fields = {"delta": "b", "item_id": "msg_test", "output_index": 1, "sequence_number": 2, **change}
    with pytest.raises(UnsupportedProviderStream):
        stream.classify_event(event("response.output_text.delta", **fields))


def test_lifecycle_snapshot_identity_matches_the_delta_target():
    stream = adapter()
    stream.classify_event(event("response.output_text.delta", delta="answer", item_id="different", output_index=1))
    with pytest.raises(UnsupportedProviderStream, match="item ID"):
        stream.classify_event(event("response.completed", response=response_payload()))


def test_hook_state_is_request_local_and_errors_use_responses_framing():
    first, second = adapter(), adapter()
    first.classify_event(event("error", message="failed", sequence_number=4))
    second.classify_event(event("response.output_text.delta", delta="fresh", sequence_number=0))
    with pytest.raises(UnsupportedProviderStream):
        second.validate_end_of_stream()
    encoded = first.encode_error(b'{"error":{"code":"content_blocked","message":"blocked","param":null}}')
    assert len(encoded) == 1
    decoded = ServerSentEvent.from_bytes(encoded[0])
    assert decoded.event_type == b"error"
    assert json.loads(decoded.data) == {
        "type": "error",
        "code": "content_blocked",
        "message": "blocked",
        "param": None,
        "sequence_number": 5,
    }
    assert b"[DONE]" not in encoded[0]


def test_sse_event_and_json_discriminator_must_agree():
    wire = b'event: response.completed\ndata: {"type":"response.output_text.delta","delta":"hidden"}\n\n'
    with pytest.raises(UnsupportedProviderStream):
        STREAM_CLASSIFIER.classify_event(ServerSentEvent.from_bytes(wire))
