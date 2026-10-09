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

import nemoguardrails.server.experimental._json_payload as json_payload
from nemoguardrails.server.experimental._json_payload import InvalidJson, UnsupportedJsonShape, parse_json_object
from nemoguardrails.server.experimental.provider.payload import (
    GuardedMessageTarget,
    validate_payload_projection_contract,
)
from nemoguardrails.server.experimental.provider.types import GuardedMessage
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
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ChatCompletionsGuardedRequest.validate_payload(payload)


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
