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

import pytest
from pydantic import ValidationError

from nemoguardrails.server.experimental._json_payload import InvalidJson, UnsupportedJsonShape, parse_json_object
from nemoguardrails.server.experimental.provider.payload import (
    GuardedMessageTarget,
    validate_payload_projection_contract,
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


def _json_bytes(payload):
    """Encode a JSON-compatible payload without insignificant whitespace."""
    return json.dumps(payload, separators=(",", ":")).encode()


def _request(**updates):
    """Build a representative OpenAI Chat request with optional changes."""
    payload = {
        "messages": [{"role": "user", "content": "question", "name": "caller", "future": {"value": 1}}],
        "model": "gpt-example",
        "temperature": 0.2,
        "future": [1, 2, 3],
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
                    "annotations": [{"provider": "opaque"}],
                    "future": True,
                },
            }
        ],
        "usage": {"total_tokens": 2},
        "future": {"provider": "opaque"},
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
    body = b'{ "messages" : [ { "role" : "user", "content" : "question", "future" : 7 } ], "model" : "gpt-example", "future" : true }'
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
    ],
)
def test_request_projection_rejects_shapes_outside_buffered_text_profile(payload):
    """The request projection rejects shapes outside its supported text profile."""
    with pytest.raises(ValidationError):
        _guarded_request(_json_bytes(payload))


def test_request_projection_reports_streaming_response_mode():
    """The request projection preserves the operation's boolean stream selector."""
    _, projection, _ = _guarded_request(_json_bytes(_request(stream=True)))

    assert projection.streams_response is True


def test_response_binding_targets_original_provider_object_without_rewriting_bytes():
    """Provider annotations block replacement without rewriting response bytes."""
    body = _json_bytes(_response())
    original = bytes(body)

    payload, _, target = _guarded_response(body)

    assert isinstance(target, GuardedMessageTarget)
    assert target.message == GuardedMessage("assistant", "answer")
    assert target._object is payload["choices"][0]["message"]
    assert target.allows_replacement is False
    assert body == original


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
        _response(choices=[{"message": {"role": "assistant", "content": ""}}]),
        _response(choices=[{"message": {"role": "assistant", "content": "answer", "tool_calls": []}}]),
        _response(choices=[{"message": {"role": "assistant", "content": "answer", "reasoning_content": "hidden"}}]),
        _response(choices=[{"logprobs": {"content": []}, "message": {"role": "assistant", "content": "answer"}}]),
        _response(choices=[{"message": {"role": "assistant", "content": "answer", "annotations": "invalid"}}]),
    ],
)
def test_response_projection_rejects_shapes_outside_buffered_text_profile(payload):
    """The response projection rejects shapes outside its supported text profile."""
    with pytest.raises(ValidationError):
        _guarded_response(_json_bytes(payload))


def test_closed_content_policy_accepts_reviewed_opaque_fields_and_rejects_unreviewed_fields():
    """Closed content policy distinguishes reviewed opaque and unknown fields."""
    request = _request(messages=[{"role": "user", "content": "question", "name": "caller"}])
    ChatCompletionsGuardedRequest.validate_payload(
        request,
        unknown_content_fields=UnknownContentFieldPolicy.FORBID,
    )
    request["messages"][0]["unknown"] = "content-bearing"

    with pytest.raises(ValidationError, match="unreviewed content fields"):
        ChatCompletionsGuardedRequest.validate_payload(
            request,
            unknown_content_fields=UnknownContentFieldPolicy.FORBID,
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
