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
from typing import Annotated, Literal

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

import nemoguardrails.server.experimental._json_payload as json_payload
from nemoguardrails.server.experimental._json_payload import (
    InvalidJson,
    UnsupportedJsonShape,
    encode_json_object,
    parse_json_object,
)
from nemoguardrails.server.experimental.provider.payload import (
    GuardedMessageTarget,
    validate_payload_projection_contract,
)
from nemoguardrails.server.experimental.provider.projection_policy import (
    EXTENSION,
    export_payload_schema,
    field_coverage,
    text_location,
)
from nemoguardrails.server.experimental.provider.types import GuardedMessage, UnknownContentFieldPolicy
from nemoguardrails.server.experimental.providers.openai.chat_completions.request_binding import (
    CAPABILITY_PROFILE as REQUEST_PROFILE,
)
from nemoguardrails.server.experimental.providers.openai.chat_completions.request_binding import (
    PAYLOAD_CONTRACT as REQUEST_CONTRACT,
)
from nemoguardrails.server.experimental.providers.openai.chat_completions.request_binding import (
    REQUEST_SOURCE_SCHEMA,
    ChatCompletionsGuardedRequest,
)
from nemoguardrails.server.experimental.providers.openai.chat_completions.request_projection import (
    ChatCompletionsGuardedRequestProjection,
    ChatCompletionsUserMessageProjection,
)
from nemoguardrails.server.experimental.providers.openai.chat_completions.response_binding import (
    CAPABILITY_PROFILE as RESPONSE_PROFILE,
)
from nemoguardrails.server.experimental.providers.openai.chat_completions.response_binding import (
    PAYLOAD_CONTRACT as RESPONSE_CONTRACT,
)
from nemoguardrails.server.experimental.providers.openai.chat_completions.response_binding import (
    RESPONSE_SOURCE_SCHEMA,
    ChatCompletionsGuardedResponse,
)
from nemoguardrails.server.experimental.providers.openai.chat_completions.response_projection import (
    ChatCompletionsAssistantMessageProjection,
    ChatCompletionsGuardedResponseProjection,
)


def _json_bytes(payload):
    """Encode a JSON-compatible payload without insignificant whitespace."""
    return json.dumps(payload, separators=(",", ":")).encode()


def _request(**updates):
    """Build a representative OpenAI Chat request with optional changes."""
    payload = {
        "messages": [{"role": "user", "content": "question", "name": None}],
        "model": "gpt-example",
        "temperature": 0.2,
    }
    payload.update(updates)
    return payload


def _response(**updates):
    """Build a representative OpenAI Chat response with optional changes."""
    payload = {
        "id": "chatcmpl-example",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "logprobs": None,
                "message": {
                    "role": "assistant",
                    "content": "answer",
                    "annotations": [],
                },
            }
        ],
        "usage": {"total_tokens": 2},
    }
    payload.update(updates)
    return payload


def _guarded_request(body: bytes):
    """Parse, project, and locate guarded text in a Chat request."""
    payload = parse_json_object(body)
    projection = ChatCompletionsGuardedRequest.validate_payload(payload)
    return payload, projection, projection.locate_guarded_message(payload)


def _guarded_response(body: bytes):
    """Parse, project, and locate guarded text in a Chat response."""
    payload = parse_json_object(body)
    projection = ChatCompletionsGuardedResponse.validate_payload(payload)
    return payload, projection, projection.locate_guarded_message(payload)


def test_request_binding_targets_original_provider_object_without_rewriting_bytes():
    """The request binding targets the decoded provider object without rewriting bytes."""
    body = b'{ "messages" : [ { "role" : "user", "content" : "question" } ], "model" : "gpt-example", "temperature" : 0.2 }'
    original = bytes(body)

    payload, projection, target = _guarded_request(body)

    assert isinstance(target, GuardedMessageTarget)
    assert target.message == GuardedMessage("user", "question")
    assert target._object is payload["messages"][0]
    assert target.allows_replacement is True
    assert projection.streams_response is False
    assert body == original


@pytest.mark.parametrize(
    "payload",
    [
        _request(messages=[]),
        _request(messages=[{"role": "user", "content": "one"}, {"role": "user", "content": "two"}]),
        _request(messages=[{"role": "assistant", "content": "question"}]),
        _request(messages=[{"role": "user", "content": ""}]),
        _request(stream=0),
        _request(n=2),
        _request(tools=[{"type": "function"}]),
        _request(response_format={"type": "json_object"}),
        _request(logprobs=True),
        _request(logprobs=0),
        _request(top_logprobs=3),
        _request(logprobs=False, top_logprobs=0),
        _request(reasoning_effort="high. uninspected instructions"),
        _request(reasoning_effort="HIGH"),
    ],
)
def test_request_projection_rejects_shapes_outside_buffered_text_profile(payload):
    """The request projection rejects shapes outside its supported text profile."""
    with pytest.raises(ValidationError):
        _guarded_request(_json_bytes(payload))


@pytest.mark.parametrize("logprobs", [None, False])
def test_request_projection_accepts_requests_without_logprobs(logprobs):
    """Clients may state that they do not want log probabilities."""
    _, projection, _ = _guarded_request(_json_bytes(_request(logprobs=logprobs, top_logprobs=None)))

    assert projection.logprobs is logprobs


@pytest.mark.parametrize("name", ["caller", "uninspected instructions", {"value": "uninspected"}])
def test_request_projection_rejects_participant_names(name):
    """The participant name reaches the model, but input rails do not inspect it."""
    with pytest.raises(ValidationError):
        _guarded_request(_json_bytes(_request(messages=[{"role": "user", "content": "question", "name": name}])))


@pytest.mark.parametrize("effort", [None, "none", "minimal", "low", "medium", "high", "xhigh", "max"])
def test_request_projection_accepts_openai_reasoning_efforts(effort):
    """Every reasoning effort in OpenAI's schema remains accepted."""
    _, projection, _ = _guarded_request(_json_bytes(_request(reasoning_effort=effort)))

    assert projection.reasoning_effort == effort


def test_request_projection_reports_streaming_response_mode():
    """The request projection preserves the operation's boolean stream selector."""
    _, projection, _ = _guarded_request(_json_bytes(_request(stream=True)))

    assert projection.streams_response is True


def test_response_binding_targets_original_provider_object_without_rewriting_bytes():
    """The response binding targets the decoded provider object without rewriting bytes."""
    body = _json_bytes(_response())
    original = bytes(body)

    payload, _, target = _guarded_response(body)

    assert isinstance(target, GuardedMessageTarget)
    assert target.message == GuardedMessage("assistant", "answer")
    assert target._object is payload["choices"][0]["message"]
    assert target.allows_replacement is True
    assert body == original


def test_declared_annotation_blocking_still_prevents_replacement():
    """Non-empty annotations are rejected, but the declared blocker still applies."""
    payload = _response()
    payload["choices"][0]["message"]["annotations"] = [{"type": "url_citation"}]

    target = ChatCompletionsGuardedResponse.guarded_text_location.locate(payload)

    assert target.allows_replacement is False


@pytest.mark.parametrize("tool_calls", [None, []])
def test_response_projection_accepts_absent_tool_calls(tool_calls):
    """Null and empty tool calls both mean that no tool was called."""
    response = _response()
    response["choices"][0]["message"]["tool_calls"] = tool_calls

    _, _, target = _guarded_response(_json_bytes(response))

    assert target.message == GuardedMessage("assistant", "answer")


@pytest.mark.parametrize(
    ("content", "finish_reason"),
    [("", "length"), (None, "content_filter"), (None, "stop")],
)
def test_response_without_text_has_nothing_to_inspect(content, finish_reason):
    """Empty or null content in an otherwise content-free response yields no checked text."""
    response = _response()
    response["choices"][0]["message"]["content"] = content
    response["choices"][0]["finish_reason"] = finish_reason

    _, _, target = _guarded_response(_json_bytes(response))

    assert target.has_text is False


@pytest.mark.parametrize(
    "message",
    [
        {"role": "assistant", "content": None, "refusal": "untrusted model text"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call", "type": "function"}]},
        {"role": "assistant", "content": None, "annotations": [{"type": "url_citation"}]},
        {"role": "assistant", "content": None, "reasoning_content": "untrusted model text"},
        {"role": "assistant"},
    ],
)
def test_response_without_text_is_rejected_when_other_fields_carry_content(message):
    """Null content does not excuse uninspected content elsewhere in the message."""
    with pytest.raises(ValidationError):
        _guarded_response(_json_bytes(_response(choices=[{"index": 0, "message": message}])))


def test_response_with_text_has_text_to_inspect():
    _, _, target = _guarded_response(_json_bytes(_response()))

    assert target.has_text is True


def test_response_binding_allows_unannotated_text_replacement():
    """Unannotated assistant text remains eligible for replacement."""
    response = _response()
    del response["choices"][0]["message"]["annotations"]

    _, _, target = _guarded_response(_json_bytes(response))

    assert target.allows_replacement is True


@pytest.mark.parametrize(
    "payload",
    [
        _response(choices=[]),
        _response(choices=[_response()["choices"][0], _response()["choices"][0]]),
        _response(choices=[{"message": {"role": "user", "content": "answer"}}]),
        _response(
            choices=[
                {
                    "message": {
                        "role": "assistant",
                        "content": "answer",
                        "tool_calls": [{"id": "call", "type": "function"}],
                    }
                }
            ]
        ),
        _response(choices=[{"message": {"role": "assistant", "content": "answer", "reasoning_content": "hidden"}}]),
        _response(choices=[{"logprobs": {"content": []}, "message": {"role": "assistant", "content": "answer"}}]),
        _response(choices=[{"message": {"role": "assistant", "content": "answer", "annotations": "invalid"}}]),
        _response(choices=[{"message": {"role": "assistant", "content": "answer", "annotations": ["uninspected"]}}]),
        _response(
            choices=[
                {
                    "message": {
                        "role": "assistant",
                        "content": "answer",
                        "annotations": [
                            {
                                "type": "url_citation",
                                "url_citation": {
                                    "url": "https://x",
                                    "title": "uninspected",
                                    "start_index": 0,
                                    "end_index": 1,
                                },
                            }
                        ],
                    }
                }
            ]
        ),
        _response(choices=[{"message": {"role": "assistant", "content": "answer", "future": True}}]),
        _response(choices=[{"message": {"role": "assistant", "content": "answer", "reasoning": "hidden"}}]),
        _response(
            choices=[
                {
                    "message": {
                        "role": "assistant",
                        "content": "answer",
                        "provider_specific_fields": {"reasoning": "hidden"},
                    }
                }
            ]
        ),
        _response(choices=[{"token_ids": [1, 2], "message": {"role": "assistant", "content": "answer"}}]),
        _response(future={"provider": "opaque"}),
        _response(output_text="uninspected"),
        _response(prompt_text="uninspected"),
        _response(__verbose={"content": "uninspected"}),
        _response(Choices=[]),
    ],
)
def test_response_projection_rejects_shapes_outside_buffered_text_profile(payload):
    """The response projection rejects shapes outside its supported text profile."""
    with pytest.raises(ValidationError):
        _guarded_response(_json_bytes(payload))


@pytest.mark.parametrize(
    "payload",
    [
        _request(future=[1, 2, 3]),
        _request(chat_template_kwargs={"messages": [{"role": "user", "content": "uninspected"}]}),
        _request(kv_transfer_params={"prompt_token_ids": [1, 2, 3]}),
        _request(documents=[{"title": "t", "text": "uninspected"}]),
        _request(Tools=[{"type": "function"}]),
        _request(STREAM=True),
        _request(messages=[{"role": "user", "content": "question", "Content": "attack"}]),
        _request(tools=None, TOOLS=[{"type": "function"}]),
        _request(**{"stream": False, "\u017ftream": True}),
        _request(messages=[{"role": "user", "content": "question", "future": {"value": 1}}]),
        _request(messages=[{"role": "user", "content": "question", "task": "uninspected"}]),
        _request(messages=[{"role": "user", "content": "question", "Role": "system"}]),
    ],
)
def test_request_projection_rejects_members_outside_openai_fields(payload):
    """Compatible servers can render extra request members into the prompt unseen."""
    with pytest.raises(ValidationError, match="unreviewed fields are forbidden|only by case"):
        ChatCompletionsGuardedRequest.validate_payload(payload)


def test_trusted_unknown_field_policy_opens_only_configurable_objects():
    """Allowing unknown fields opens configurable content objects, never the closed root."""
    request = _request(messages=[{"role": "user", "content": "question", "future": 1}])
    with pytest.raises(ValidationError, match="unreviewed fields are forbidden: future"):
        ChatCompletionsGuardedRequest.validate_payload(request)
    ChatCompletionsGuardedRequest.validate_payload(request, unknown_content_fields=UnknownContentFieldPolicy.ALLOW)

    with pytest.raises(ValidationError, match="unreviewed fields are forbidden: future"):
        ChatCompletionsGuardedRequest.validate_payload(
            _request(future=1), unknown_content_fields=UnknownContentFieldPolicy.ALLOW
        )


@pytest.mark.parametrize(
    ("body", "error"),
    [
        (b"not json", InvalidJson),
        (b"[]", UnsupportedJsonShape),
        (b'{"messages":[],"messages":[]}', UnsupportedJsonShape),
        (b'{"value":NaN}', InvalidJson),
        (b"\xff", InvalidJson),
    ],
)
def test_strict_json_parser_rejects_ambiguous_or_non_object_payloads(body, error):
    """Strict JSON parsing rejects ambiguous, invalid, and non-object bodies."""
    with pytest.raises(error):
        parse_json_object(body)


def test_strict_json_parser_rejects_non_finite_numbers_and_non_bytes():
    """Overflowing floats are invalid, and only raw bytes are parsed."""
    with pytest.raises(InvalidJson):
        parse_json_object(b'{"value":1e400}')
    with pytest.raises(TypeError, match="must be bytes"):
        parse_json_object('{"value":1}')


def test_modified_objects_encode_as_compact_utf8_json():
    """A modified provider object is serialized without escapes or extra whitespace."""
    assert encode_json_object({"text": "caf\u00e9", "n": [1, 2]}) == '{"text":"caf\u00e9","n":[1,2]}'.encode()


def test_strict_json_parser_keeps_case_variant_names_in_opaque_data():
    """Only exact duplicates are ambiguous to every parser; closed models handle case variants."""
    assert parse_json_object(b'{"metadata":{"Env":"a","env":"b"}}') == {"metadata": {"Env": "a", "env": "b"}}


def test_strict_json_parser_normalizes_integer_conversion_failures(monkeypatch):
    """Integer conversion failures produce the parser's stable invalid-JSON outcome."""

    def reject_integer(_value):
        """Simulate an interpreter integer-size rejection."""
        raise ValueError("integer exceeds the configured digit limit")

    monkeypatch.setattr(json_payload, "int", reject_integer, raising=False)

    with pytest.raises(InvalidJson):
        parse_json_object(b'{"value":1}')


def test_bindings_match_contract_identity_and_replacement_policy():
    """Staged bindings preserve the generated contract identity and replacement policy."""
    assert REQUEST_PROFILE == RESPONSE_PROFILE == "single_text.v1"
    assert REQUEST_SOURCE_SCHEMA == "CreateChatCompletionRequest"
    assert RESPONSE_SOURCE_SCHEMA == "CreateChatCompletionResponse"
    assert REQUEST_CONTRACT.direction == "request"
    assert RESPONSE_CONTRACT.direction == "response"
    assert validate_payload_projection_contract(ChatCompletionsGuardedRequest, "request") is REQUEST_CONTRACT
    assert validate_payload_projection_contract(ChatCompletionsGuardedResponse, "response") is RESPONSE_CONTRACT
    assert ChatCompletionsGuardedRequest.guarded_text_location.allows_replacement is True
    assert ChatCompletionsGuardedResponse.guarded_text_location.allows_replacement is True


@pytest.mark.parametrize(
    "module",
    [
        "nemoguardrails.server.experimental.provider.payload",
        "nemoguardrails.server.experimental.providers.openai.chat_completions.request_binding",
        "nemoguardrails.server.experimental.providers.openai.chat_completions.response_binding",
    ],
)
def test_staged_projection_modules_import_in_fresh_interpreter(module):
    """Each staged projection module imports in a fresh interpreter."""
    result = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_explicit_defaults_are_visible_to_python_and_pydantic():
    message = ChatCompletionsUserMessageProjection(content="question", role="user")
    request = ChatCompletionsGuardedRequestProjection(messages=[message])
    assert request.n == 1
    assert request.stream is False
    assert request.audio is None
    for name, expected in (("n", 1), ("stream", False), ("audio", None)):
        field = ChatCompletionsGuardedRequestProjection.model_fields[name]
        assert not field.is_required()
        assert field.default is expected


@pytest.mark.parametrize("value", [None, []])
def test_annotations_accept_only_null_or_empty(value):
    message = ChatCompletionsAssistantMessageProjection(role="assistant", content="answer", annotations=value)
    assert message.annotations == value
    with pytest.raises(ValidationError):
        ChatCompletionsAssistantMessageProjection(role="assistant", content="answer", annotations=[{"type": "x"}])


def test_n_accepts_the_integer_one():
    request = ChatCompletionsGuardedRequest.model_validate({"messages": [{"role": "user", "content": "q"}], "n": 1})
    assert request.n == 1


@pytest.mark.parametrize("value", [True, False, "1", 1.0, 0, 2, None])
def test_n_rejects_coercible_and_other_values(value):
    with pytest.raises(ValidationError):
        ChatCompletionsGuardedRequest.model_validate({"messages": [{"role": "user", "content": "q"}], "n": value})


@pytest.mark.parametrize("value", [0, 1, "true", None])
def test_stream_stays_strict(value):
    with pytest.raises(ValidationError):
        ChatCompletionsGuardedRequest.model_validate({"messages": [{"role": "user", "content": "q"}], "stream": value})


@pytest.mark.parametrize("value", [[], {}, False, "audio"])
def test_disabled_fields_reject_every_non_null_value(value):
    with pytest.raises(ValidationError):
        ChatCompletionsGuardedRequest.model_validate({"messages": [{"role": "user", "content": "q"}], "audio": value})


def test_bindings_derive_coverage_and_targets_from_the_typed_models():
    assert REQUEST_CONTRACT.root == field_coverage(ChatCompletionsGuardedRequestProjection)
    assert RESPONSE_CONTRACT.root == field_coverage(ChatCompletionsGuardedResponseProjection)
    assert ChatCompletionsGuardedRequest.guarded_text_location == text_location(ChatCompletionsGuardedRequestProjection)
    assert ChatCompletionsGuardedResponse.guarded_text_location == text_location(
        ChatCompletionsGuardedResponseProjection
    )
    response_content = RESPONSE_CONTRACT.content_models[-1]
    assert response_content.model is ChatCompletionsAssistantMessageProjection
    assert response_content.coverage.local_extension_fields == frozenset({"reasoning_content"})
    assert ChatCompletionsGuardedResponse.guarded_text_location.allows_empty is True
    assert ChatCompletionsGuardedRequest.guarded_text_location.allows_empty is False


def test_export_preserves_nullable_annotation_schema():
    exported = export_payload_schema(
        ChatCompletionsGuardedResponseProjection, projection_id=RESPONSE_CONTRACT.projection_id
    )
    annotations = exported["properties"]["choices"]["items"]["properties"]["message"]["properties"]["annotations"]
    assert annotations["default"] is None
    assert {"type": "null"} in annotations["oneOf"]
    assert {"type": "array", "items": {}, "maxItems": 0} in annotations["oneOf"]


def test_export_lists_reviewed_opaque_names_as_properties():
    request = export_payload_schema(ChatCompletionsGuardedRequestProjection, projection_id="test.request")
    assert request["properties"]["model"] == {EXTENSION: {"classification": "opaque"}}
    assert "opaque_fields" not in request[EXTENSION]
    validator = Draft202012Validator(request)
    message = {"role": "user", "content": "q"}
    assert validator.is_valid({"messages": [message], "model": "m", "temperature": 0.2})
    assert not validator.is_valid({"messages": [message], "future": 1})


def test_export_follows_each_object_unknown_field_policy():
    request = export_payload_schema(ChatCompletionsGuardedRequestProjection, projection_id="test.request")
    response = export_payload_schema(ChatCompletionsGuardedResponseProjection, projection_id="test.response")
    choice = response["properties"]["choices"]["items"]
    configurable = (request["properties"]["messages"]["items"], choice, choice["properties"]["message"])
    for content in configurable:
        assert content["additionalProperties"] is True
        assert content[EXTENSION]["unknown_fields"] == "configurable"
    for root in (request, response):
        assert root["additionalProperties"] is False
        assert "unknown_fields" not in root[EXTENSION]


def test_field_validation_matches_original_unannotated_declarations():
    from itertools import product
    from typing import Any

    from pydantic import BaseModel, Field, StrictBool

    class OriginalRequestFields(BaseModel):
        n: Literal[1] = 1
        stream: StrictBool = False
        audio: None = None

    class OriginalResponseFields(BaseModel):
        annotations: Annotated[list[Any] | None, Field(max_length=0)] = None
        content: str | None
        logprobs: None = None

    def validated(model, payload):
        try:
            return model.model_validate(payload).model_dump()
        except ValidationError:
            return "rejected"

    values = [None, True, False, 0, 1, 1.0, 2, "1", "true", "", [], {}, ["x"]]
    # n is intentionally stricter than Literal[1]; test_n_rejects_coercible_and_other_values covers it.
    for stream, audio in product(values, repeat=2):
        fields = {"n": 1, "stream": stream, "audio": audio}
        old = validated(OriginalRequestFields, fields)
        new = validated(
            ChatCompletionsGuardedRequestProjection,
            {
                "messages": [{"role": "user", "content": "q"}],
                **fields,
            },
        )
        if isinstance(new, dict):
            new = {name: new[name] for name in fields}
        assert old == new, fields

    for annotations, content, logprobs in product(values + ["answer"], repeat=3):
        fields = {"annotations": annotations, "content": content, "logprobs": logprobs}
        old = validated(OriginalResponseFields, fields)
        new = validated(
            ChatCompletionsGuardedResponseProjection,
            {
                "choices": [
                    {
                        "message": {"role": "assistant", "content": content, "annotations": annotations},
                        "logprobs": logprobs,
                    }
                ],
            },
        )
        if isinstance(new, dict):
            choice = new["choices"][0]
            new = {name: choice["message"][name] for name in ("annotations", "content")}
            new["logprobs"] = choice["logprobs"]
        assert old == new, fields


_MESSAGE = {"role": "user", "content": "q"}
_REQUEST = {"messages": [_MESSAGE], "model": "m"}
_CHOICE = {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "a"}}
_RESPONSE = {"id": "r", "choices": [_CHOICE], "usage": {"total_tokens": 1}}


@pytest.mark.parametrize("policy", list(UnknownContentFieldPolicy))
@pytest.mark.parametrize(
    ("model", "document"),
    [
        (ChatCompletionsGuardedRequest, _REQUEST),
        (ChatCompletionsGuardedRequest, {**_REQUEST, "temperature": 0.2, "metadata": {"Env": "a", "env": "b"}}),
        (ChatCompletionsGuardedRequest, {**_REQUEST, "future": 1}),
        (ChatCompletionsGuardedRequest, {**_REQUEST, "Tools": []}),
        (ChatCompletionsGuardedRequest, {**_REQUEST, "MODEL": "other"}),
        (ChatCompletionsGuardedRequest, {**_REQUEST, "messages": [{**_MESSAGE, "future": 1}]}),
        (ChatCompletionsGuardedRequest, {**_REQUEST, "messages": [{**_MESSAGE, "Content": "x"}]}),
        (ChatCompletionsGuardedRequest, {**_REQUEST, "messages": [{**_MESSAGE, "namK": "x"}]}),
        (ChatCompletionsGuardedResponse, _RESPONSE),
        (ChatCompletionsGuardedResponse, {**_RESPONSE, "future": 1}),
        (ChatCompletionsGuardedResponse, {**_RESPONSE, "Usage": {}}),
        (ChatCompletionsGuardedResponse, {**_RESPONSE, "choices": [{**_CHOICE, "future": 1}]}),
        (ChatCompletionsGuardedResponse, {**_RESPONSE, "choices": [{**_CHOICE, "INDEX": 1}]}),
        (ChatCompletionsGuardedResponse, {**_RESPONSE, "choices": [{**_CHOICE, "meßage": {"content": "x"}}]}),
        (ChatCompletionsGuardedResponse, {**_RESPONSE, "choices": [{**_CHOICE, "Message\n": {"content": "x"}}]}),
        (
            ChatCompletionsGuardedResponse,
            {**_RESPONSE, "choices": [{**_CHOICE, "message": {**_CHOICE["message"], "future": 1}}]},
        ),
        (
            ChatCompletionsGuardedResponse,
            {**_RESPONSE, "choices": [{**_CHOICE, "message": {**_CHOICE["message"], "Refusal": "x"}}]},
        ),
        (
            ChatCompletionsGuardedResponse,
            {**_RESPONSE, "choices": [{**_CHOICE, "message": {**_CHOICE["message"], "refuſal": "x"}}]},
        ),
    ],
)
def test_export_derived_acceptance_matches_handwritten_runtime(model, document, policy, export_policy_validator):
    """Acceptance derived only from the export agrees with the handwritten runtime.

    This shows that the export carries the member policy the runtime enforces.
    It is not compiler equivalence, which needs tests against generated models.
    """
    exported = export_payload_schema(model, projection_id=model.projection_contract.projection_id)
    derived = export_policy_validator(exported, policy).is_valid(document)
    try:
        model.validate_payload(document, unknown_content_fields=policy)
        handwritten = True
    except ValidationError:
        handwritten = False

    assert derived is handwritten
