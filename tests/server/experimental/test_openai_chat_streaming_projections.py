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
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from nemoguardrails.server.experimental._content_checker import ContentAllowed, StreamBufferingPolicy
from nemoguardrails.server.experimental._guarded_stream import guard_provider_stream
from nemoguardrails.server.experimental.provider.projection_policy import (
    EXTENSION,
    export_payload_schema,
    field_coverage,
    field_policy,
    model_graph,
)
from nemoguardrails.server.experimental.provider.sse import ServerSentEvent
from nemoguardrails.server.experimental.provider.stream import (
    ClassifiedStreamAdapter,
    StreamEventRole,
    UnsupportedProviderStream,
    create_classified_stream_adapter_factory,
)
from nemoguardrails.server.experimental.provider.types import GuardedMessage, UnknownContentFieldPolicy
from nemoguardrails.server.experimental.providers.openai.chat_completions.request_binding import (
    PAYLOAD_CONTRACT,
    REQUEST_CONSTRAINED_FIELDS,
    REQUEST_GUARDED_FIELDS,
    REQUEST_OPAQUE_FIELDS,
    STREAM_SELECTOR_FIELD,
)
from nemoguardrails.server.experimental.providers.openai.chat_completions.stream_classifier import (
    CAPABILITY_PROFILE,
    GUARDED_SHAPES,
    OPAQUE_SHAPES,
    PROVIDER_ERROR_SHAPES,
    SNAPSHOT_SHAPES,
    STREAM_CLASSIFIER,
    STREAM_CONTRACT,
    STREAM_FIELDS,
    STREAM_SOURCE_SCHEMA,
)
from nemoguardrails.server.experimental.providers.openai.chat_completions.stream_hooks import (
    ChatCompletionsStreamHooks,
)
from nemoguardrails.server.experimental.providers.openai.chat_completions.stream_projection import (
    ChatCompletionsStreamDeltaProjection,
    ChatCompletionsStreamErrorProjection,
    ChatCompletionsStreamPayloadProjection,
)
from nemoguardrails.server.experimental.providers.openai.errors import render_openai_error


def event(payload):
    data = payload if isinstance(payload, bytes) else json.dumps(payload, separators=(",", ":")).encode()
    return ServerSentEvent.from_bytes(b"data: " + data + b"\n\n")


def chunk(delta, **fields):
    return {
        "id": "chatcmpl-stream",
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": delta, **fields}],
        "system_fingerprint": "fp-stream",
    }


@pytest.mark.parametrize(
    ("native", "shape", "role", "text"),
    [
        (chunk({"role": "assistant"}), "chat.completion.chunk:metadata", StreamEventRole.OPAQUE_METADATA, None),
        (chunk({"content": "answer"}), "chat.completion.chunk:content", StreamEventRole.GUARDED_TEXT, "answer"),
        (
            {"object": "chat.completion.chunk", "choices": [], "usage": {"output_tokens": 1}},
            "chat.completion.chunk:metadata",
            StreamEventRole.OPAQUE_METADATA,
            None,
        ),
        (
            {"error": {"message": "upstream failed", "type": "server_error"}},
            "error",
            StreamEventRole.PROVIDER_ERROR,
            None,
        ),
        (b"[DONE]", "[DONE]", StreamEventRole.OPAQUE_METADATA, None),
    ],
)
def test_stream_classifier_covers_the_declared_event_inventory(native, shape, role, text):
    classified = STREAM_CLASSIFIER.classify_event(event(native))

    assert (classified.shape, classified.role, classified.text) == (shape, role, text)
    assert shape in GUARDED_SHAPES | OPAQUE_SHAPES | PROVIDER_ERROR_SHAPES


def test_stream_classifier_treats_non_data_events_as_declared_opaque_shape():
    classified = STREAM_CLASSIFIER.classify_event(ServerSentEvent.from_bytes(b": keepalive\n\n"))

    assert classified.shape == "[DONE]"
    assert classified.role is StreamEventRole.OPAQUE_METADATA


@pytest.mark.parametrize(
    "native",
    [
        chunk({"reasoning_content": "unguarded"}),
        chunk({"content": "answer", "tool_calls": []}),
        chunk({"content": "answer", "future_content": "unguarded"}),
        {
            "object": "chat.completion.chunk",
            "choices": [
                {"index": 0, "delta": {"content": "first"}},
                {"index": 1, "delta": {"content": "second"}},
            ],
        },
        {"error": {"message": "failed"}, "choices": [{"index": 0, "delta": {"content": "hidden"}}]},
    ],
)
def test_stream_classifier_rejects_content_outside_the_closed_profile(native):
    with pytest.raises(UnsupportedProviderStream):
        STREAM_CLASSIFIER.classify_event(event(native))


def test_stream_classifier_accepts_reviewed_error_extensions_without_guarding_them():
    classified = STREAM_CLASSIFIER.classify_event(
        event({"error": {"message": "failed", "misalignment": {"provider": "opaque"}}})
    )

    assert classified.role is StreamEventRole.PROVIDER_ERROR


def test_stream_contract_preserves_declared_identity_and_field_inventory():
    assert CAPABILITY_PROFILE == "single_text_delta.v1"
    assert STREAM_SOURCE_SCHEMA == "CreateChatCompletionStreamResponse"
    assert STREAM_CONTRACT.projection_id == "openai.chat_completions.stream.text.v1"
    assert STREAM_CONTRACT.fields is not None
    assert STREAM_FIELDS == (
        STREAM_CONTRACT.fields.guarded_fields,
        STREAM_CONTRACT.fields.constrained_fields,
        STREAM_CONTRACT.fields.opaque_fields,
    )


def test_shared_request_binding_owns_the_stream_selector_and_field_inventory():
    assert STREAM_SELECTOR_FIELD == "stream"
    assert REQUEST_GUARDED_FIELDS == PAYLOAD_CONTRACT.root.guarded_fields
    assert REQUEST_CONSTRAINED_FIELDS == PAYLOAD_CONTRACT.root.constrained_fields
    assert REQUEST_OPAQUE_FIELDS == PAYLOAD_CONTRACT.root.opaque_fields


def test_stream_models_classify_every_declared_field_and_derive_root_coverage():
    for root in (ChatCompletionsStreamPayloadProjection, ChatCompletionsStreamErrorProjection):
        for model in model_graph(root).values():
            for field in model.model_fields.values():
                assert field_policy(field)["classification"] in {"guarded", "constrained", "opaque"}
    coverage = field_coverage(ChatCompletionsStreamPayloadProjection)
    assert STREAM_CONTRACT.fields == coverage
    assert coverage.guarded_fields == frozenset({"choices"})
    assert coverage.constrained_fields == frozenset({"object"})
    assert coverage.opaque_fields == frozenset(
        {"created", "id", "model", "moderation", "obfuscation", "service_tier", "system_fingerprint", "usage"}
    )
    assert field_coverage(ChatCompletionsStreamDeltaProjection).local_extension_fields == frozenset(
        {"audio", "reasoning_content"}
    )


def test_stream_schema_preserves_closure_defaults_and_text_policy():
    schema = export_payload_schema(ChatCompletionsStreamPayloadProjection, projection_id=STREAM_CONTRACT.projection_id)
    choices = schema["properties"]["choices"]
    choice = choices["items"]
    delta = choice["properties"]["delta"]
    assert choices["maxItems"] == 1
    assert "minItems" not in choices
    assert choice["additionalProperties"] is delta["additionalProperties"] is False
    assert choice["required"] == ["delta", "index"]
    assert choice["properties"]["index"]["const"] == 0
    content = delta["properties"]["content"]
    assert content["default"] is None
    assert content["oneOf"] == [{"type": "string"}, {"type": "null"}]
    assert content[EXTENSION]["subject"] == {"kind": "text", "role": "assistant"}
    assert delta[EXTENSION]["unknown_fields"] == "forbid"
    assert choice["properties"]["logprobs"][EXTENSION]["gate"] == "disabled"
    for native in (chunk({}), chunk({"content": None}), chunk({"content": ""}), chunk({"content": "safe"})):
        Draft202012Validator(schema).validate(native)
    error = export_payload_schema(ChatCompletionsStreamErrorProjection, projection_id="error")
    assert error["additionalProperties"] is False
    assert error["properties"]["error"]["additionalProperties"] is False
    assert error["properties"]["error"][EXTENSION]["unknown_fields"] == "configurable"
    assert error["properties"]["error"]["properties"]["message"][EXTENSION]["classification"] == "constrained"
    Draft202012Validator(error).validate({"error": {"message": "failed", "type": {"opaque": True}}})


def test_stream_hooks_frame_provider_native_error_and_terminal_events():
    body = b'{"error":{"code":"failed"}}'

    assert ChatCompletionsStreamHooks().encode_error(body) == (
        b'data: {"error":{"code":"failed"}}\n\n',
        b"data: [DONE]\n\n",
    )


@pytest.mark.parametrize(
    "module",
    [
        "nemoguardrails.server.experimental.providers.openai.chat_completions.request_binding",
        "nemoguardrails.server.experimental.providers.openai.chat_completions.stream_projection",
        "nemoguardrails.server.experimental.providers.openai.chat_completions.stream_classifier",
        "nemoguardrails.server.experimental.providers.openai.chat_completions.stream_hooks",
    ],
)
def test_stream_modules_import_in_a_fresh_interpreter(module):
    result = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_stream_declarations_preserve_transport_shapes_and_provenance():
    """Check the staged stream declarations without a second authored policy."""
    contracts = Path(__file__).parents[3] / "nemoguardrails/server/experimental/contracts/openai"
    source = yaml.safe_load((contracts / "source.yaml").read_text(encoding="utf-8"))
    assert STREAM_CLASSIFIER.require_event_type is False
    assert STREAM_CLASSIFIER.non_data_shape == "[DONE]"
    assert STREAM_CLASSIFIER.sentinels == ((b"[DONE]", "[DONE]"),)
    assert GUARDED_SHAPES == frozenset({"chat.completion.chunk:content"})
    assert OPAQUE_SHAPES == frozenset({"[DONE]", "chat.completion.chunk:metadata"})
    assert PROVIDER_ERROR_SHAPES == frozenset({"error"})
    assert SNAPSHOT_SHAPES == frozenset()
    content, error = STREAM_CLASSIFIER.rules
    assert content.match == (("object", "chat.completion.chunk"),)
    assert content.required_fields == frozenset()
    assert content.text_path == ("choices", 0, "delta", "content")
    assert content.missing_text_shape == "chat.completion.chunk:metadata"
    assert content.missing_text_role is StreamEventRole.OPAQUE_METADATA
    assert error.match == ()
    assert error.required_fields == frozenset({"error"})
    assert error.role is StreamEventRole.PROVIDER_ERROR


@pytest.mark.parametrize(
    ("delta", "text"),
    [
        ({}, None),
        ({"content": None}, None),
        ({"role": None}, None),
        ({"role": "assistant", "content": ""}, ""),
        ({"content": "answer"}, "answer"),
    ],
)
def test_stream_text_preserves_missing_null_and_empty_string_semantics(delta, text):
    """Null or absent content is metadata; an empty string is still guarded text."""
    classified = STREAM_CLASSIFIER.classify_event(event(chunk(delta)))
    assert classified.text == text
    expected = StreamEventRole.OPAQUE_METADATA if text is None else StreamEventRole.GUARDED_TEXT
    assert classified.role is expected


@pytest.mark.parametrize("index", [False, True, "0", 0.0, 1])
def test_stream_choice_index_requires_the_declared_integer_zero(index):
    native = chunk({"content": "hidden"})
    native["choices"][0]["index"] = index

    with pytest.raises(UnsupportedProviderStream):
        STREAM_CLASSIFIER.classify_event(event(native))


@pytest.mark.parametrize("delta", [{"content": "safe"}, {"role": "assistant"}, {}])
@pytest.mark.parametrize(
    "fields",
    [
        {"message": {"role": "assistant", "content": "unchecked"}},
        {"text": "unchecked"},
        {"future_content": "unchecked"},
    ],
)
def test_stream_classifier_rejects_unreviewed_choice_fields(delta, fields):
    with pytest.raises(UnsupportedProviderStream):
        STREAM_CLASSIFIER.classify_event(event(chunk(delta, **fields)))


def test_stream_classifier_preserves_reviewed_choice_metadata():
    native = chunk({"content": "safe"}, finish_reason="stop", logprobs=None)

    classified = STREAM_CLASSIFIER.classify_event(event(native))

    assert classified.text == "safe"


@pytest.mark.parametrize("logprobs", [{"content": [{"token": "uninspected"}]}, {"content": []}])
def test_stream_classifier_rejects_log_probabilities(logprobs):
    """Stream log probabilities carry token text, as in buffered responses."""
    with pytest.raises(UnsupportedProviderStream):
        STREAM_CLASSIFIER.classify_event(event(chunk({"content": "safe"}, logprobs=logprobs)))


@pytest.mark.parametrize("member", [{"future": 1}, {"Choices": []}])
def test_stream_classifier_rejects_unreviewed_root_members(member):
    """The stream root is closed like the buffered response root."""
    with pytest.raises(UnsupportedProviderStream):
        STREAM_CLASSIFIER.classify_event(event({**chunk({"content": "safe"}), **member}))


def observe(hooks, value):
    parsed = value if isinstance(value, ServerSentEvent) else event(value)
    hooks.observe_event(parsed, STREAM_CLASSIFIER.classify_event(parsed))


@pytest.mark.parametrize("keepalive", [b": keepalive\n\n", b"\n", b"id: keepalive\n\n"])
def test_stream_hooks_require_actual_done_after_keepalives(keepalive):
    hooks = ChatCompletionsStreamHooks()
    observe(hooks, ServerSentEvent.from_bytes(keepalive))

    with pytest.raises(UnsupportedProviderStream, match="before.*DONE"):
        hooks.validate_end_of_stream()

    observe(hooks, chunk({"content": "safe"}))
    observe(hooks, b"[DONE]")
    observe(hooks, ServerSentEvent.from_bytes(keepalive))
    hooks.validate_end_of_stream()


@pytest.mark.parametrize("tail", [b"[DONE]", chunk({"content": "later"}), chunk({}), {"error": {"message": "late"}}])
def test_stream_hooks_reject_data_after_done(tail):
    hooks = ChatCompletionsStreamHooks()
    observe(hooks, b"[DONE]")

    with pytest.raises(UnsupportedProviderStream, match="after.*DONE"):
        observe(hooks, tail)


@pytest.mark.parametrize("done", [False, True])
def test_stream_hooks_preserve_provider_error_endings(done):
    hooks = ChatCompletionsStreamHooks()
    observe(hooks, {"error": {"message": "upstream failed"}})
    observe(hooks, ServerSentEvent.from_bytes(b": keepalive\n\n"))
    if done:
        observe(hooks, b"[DONE]")

    hooks.validate_end_of_stream()


@pytest.mark.parametrize("tail", [chunk({"content": "later"}), chunk({}), {"error": {"message": "again"}}])
def test_stream_hooks_reject_payloads_after_provider_error(tail):
    hooks = ChatCompletionsStreamHooks()
    observe(hooks, {"error": {"message": "upstream failed"}})

    with pytest.raises(UnsupportedProviderStream, match="after.*error"):
        observe(hooks, tail)


def test_stream_adapter_factory_creates_independent_completion_state():
    factory = create_classified_stream_adapter_factory(STREAM_CLASSIFIER, ChatCompletionsStreamHooks)
    first, second = factory(), factory()
    first.classify_event(event(b"[DONE]"))
    second.classify_event(event(chunk({"content": "safe"})))

    first.validate_end_of_stream()
    with pytest.raises(UnsupportedProviderStream):
        second.validate_end_of_stream()
    second.classify_event(event(b"[DONE]"))
    second.validate_end_of_stream()


class StreamChecker:
    def __init__(self):
        self.checked = []

    async def check_output(self, check):
        self.checked.append(check.output_content)
        return ContentAllowed()


async def guarded_events(events, chunk_size=1):
    closed = []

    async def source():
        try:
            for item in events:
                yield item.raw
        finally:
            closed.append(True)

    checker = StreamChecker()
    result = b"".join(
        [
            raw
            async for raw in guard_provider_stream(
                source(),
                checker=checker,
                streaming_policy=StreamBufferingPolicy(chunk_size, 0),
                input_message=GuardedMessage("user", "question"),
                adapter=ClassifiedStreamAdapter(STREAM_CLASSIFIER, ChatCompletionsStreamHooks()),
                render_outcome=render_openai_error,
            )
        ]
    )
    assert closed == [True]
    return result, checker.checked


@pytest.mark.asyncio
@pytest.mark.parametrize("delta", [{"content": "safe"}, {"role": "assistant"}])
async def test_guarded_stream_hides_unreviewed_choice_content(delta):
    invalid = event(chunk(delta, message={"content": "unchecked"}))

    result, checked = await guarded_events([invalid, event(b"[DONE]")])

    assert invalid.raw not in result
    assert b"unchecked" not in result
    assert checked == []
    assert b"unsupported_chat_completions_response_shape" in result
    assert result.endswith(event(b"[DONE]").raw)


@pytest.mark.asyncio
@pytest.mark.parametrize("chunk_size", [1, 2])
async def test_guarded_stream_reports_missing_done_without_releasing_pending_text(chunk_size):
    text = event(chunk({"content": "safe"}))

    result, checked = await guarded_events([text], chunk_size)

    assert (text.raw in result) is (chunk_size == 1)
    assert checked == (["safe"] if chunk_size == 1 else [])
    assert b"unsupported_chat_completions_response_shape" in result
    assert result.endswith(event(b"[DONE]").raw)


@pytest.mark.asyncio
@pytest.mark.parametrize("tail", [event(b"[DONE]"), event(chunk({"content": "later"}))])
async def test_guarded_stream_hides_invalid_events_after_done(tail):
    text = event(chunk({"content": "safe"}))

    result, checked = await guarded_events([text, event(b"[DONE]"), tail])

    assert result.startswith(text.raw)
    assert checked == ["safe"]
    assert b"later" not in result
    assert b"unsupported_chat_completions_response_shape" in result
    assert result.count(event(b"[DONE]").raw) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("done", [False, True])
async def test_guarded_stream_preserves_provider_errors_without_synthesizing_another(done):
    events = [event({"error": {"message": "upstream failed"}})]
    if done:
        events.append(event(b"[DONE]"))

    result, checked = await guarded_events(events)

    assert result == b"".join(item.raw for item in events)
    assert checked == []


@pytest.mark.asyncio
async def test_guarded_stream_preserves_valid_sequence_and_reviewed_metadata():
    events = [
        ServerSentEvent.from_bytes(b": keepalive\r\n\r\n"),
        event(chunk({"role": "assistant"})),
        event(chunk({"content": "safe"}, finish_reason=None, logprobs=None)),
        event(chunk({}, finish_reason="stop")),
        event({"object": "chat.completion.chunk", "choices": [], "usage": {"total_tokens": 1}}),
        event(b"[DONE]"),
        ServerSentEvent.from_bytes(b": trailing keepalive\n\n"),
    ]

    result, checked = await guarded_events(events)

    assert result == b"".join(item.raw for item in events)
    assert checked == ["safe"]


_CHUNK = chunk({"content": "safe"}, finish_reason="stop")
_ERROR = {"error": {"message": "failed", "type": "server_error"}}


@pytest.mark.parametrize("policy", list(UnknownContentFieldPolicy))
@pytest.mark.parametrize(
    ("model", "document"),
    [
        (ChatCompletionsStreamPayloadProjection, _CHUNK),
        (ChatCompletionsStreamPayloadProjection, {**_CHUNK, "usage": {"total_tokens": 1}}),
        (ChatCompletionsStreamPayloadProjection, {**_CHUNK, "future": 1}),
        (ChatCompletionsStreamPayloadProjection, {**_CHUNK, "Usage": {}}),
        (ChatCompletionsStreamPayloadProjection, chunk({"content": "safe"}, future=1)),
        (ChatCompletionsStreamPayloadProjection, chunk({"content": "safe", "future": 1})),
        (ChatCompletionsStreamPayloadProjection, chunk({"content": "safe", "Content": "x"})),
        (ChatCompletionsStreamErrorProjection, _ERROR),
        (ChatCompletionsStreamErrorProjection, {"error": {**_ERROR["error"], "future": 1}}),
        (ChatCompletionsStreamErrorProjection, {"error": {**_ERROR["error"], "Message": "x"}}),
        (ChatCompletionsStreamErrorProjection, {**_ERROR, "future": 1}),
    ],
)
def test_stream_export_member_policy_matches_projections(model, document, policy, export_policy_validator):
    """Export and typed projections agree on which object members are accepted.

    This covers object-member policy only, not full value equivalence (see the
    integer index test below). The policy is applied by direct projection
    validation; the stream classifier itself always validates with the closed
    default, so trusted ALLOW is not exercised through the classifier here.
    """
    exported = export_payload_schema(model, projection_id=model.__name__)
    derived = export_policy_validator(exported, policy).is_valid(document)
    try:
        model.validate_payload(document, unknown_content_fields=policy)
        runtime = True
    except ValidationError:
        runtime = False

    assert derived is runtime


def test_strict_index_is_a_known_export_value_gap():
    """JSON Schema accepts 0.0 as integer 0; the runtime keeps the stricter integer check."""
    exported = export_payload_schema(ChatCompletionsStreamPayloadProjection, projection_id="stream")
    native = chunk({"content": "safe"})
    native["choices"][0]["index"] = 0.0

    assert Draft202012Validator(exported).is_valid(native)
    with pytest.raises(ValidationError):
        ChatCompletionsStreamPayloadProjection.validate_payload(native)
