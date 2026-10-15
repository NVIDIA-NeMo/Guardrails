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

"""Characterize the typed foreground Responses boundary without provider calls."""

from copy import deepcopy

import pytest
from pydantic import ValidationError

from nemoguardrails.server.experimental.provider.types import UnknownContentFieldPolicy
from nemoguardrails.server.experimental.providers.openai.responses.request_binding import ResponsesGuardedRequest
from nemoguardrails.server.experimental.providers.openai.responses.response_binding import ResponsesGuardedResponse


def output_message(text="answer", **updates):
    """Build a synthetic single-text message, not a recorded provider response."""
    return {
        "type": "message",
        "role": "assistant",
        "id": "msg_test",
        "content": [{"type": "output_text", "text": text, "annotations": [], "logprobs": []}],
        **updates,
    }


def response_payload(text="answer"):
    """Include reasoning metadata before the assistant item to exercise selection."""
    return {
        "id": "resp_test",
        "object": "response",
        "output": [
            {"type": "reasoning", "id": "rs_test", "summary": [], "content": [], "encrypted_content": "opaque"},
            output_message(text),
        ],
        "usage": {"output_tokens": 1},
    }


def test_request_extracts_original_input_without_modifying_provider_fields():
    payload = {
        "model": "test-model",
        "input": "hello",
        "reasoning": {"effort": "medium"},
        "text": {"format": {"type": "text"}, "verbosity": "medium"},
    }
    original = deepcopy(payload)
    request = ResponsesGuardedRequest.validate_payload(payload)
    target = request.locate_guarded_message(payload)
    assert target.message.content == "hello"
    assert request.streams_response is False
    assert not target.allows_replacement
    assert payload == original


@pytest.mark.parametrize(
    "updates",
    [
        {"input": ""},
        {"input": [{"role": "user", "content": "hidden"}]},
        {"instructions": "hidden"},
        {"previous_response_id": "prior"},
        {"conversation": "prior"},
        {"tools": []},
        {"tool_choice": "none"},
        {"include": []},
        {"background": True},
        {"background": 0},
        {"prompt": {}},
        {"context_management": []},
        {"top_logprobs": 1},
        {"top_logprobs": True},
        {"stream": "true"},
        {"stream": 1},
        {"stream": None},
        {"future_prompt": "hidden"},
        {"Input": "hidden"},
        {"reasoning": {"summary": "auto"}},
        {"text": {"format": {"type": "json_schema", "schema": {}}}},
    ],
)
def test_unsupported_requests_fail_closed(updates):
    with pytest.raises(ValidationError):
        ResponsesGuardedRequest.validate_payload({"input": "hello", **updates})


@pytest.mark.parametrize("stream", [False, True])
def test_request_selects_response_mode_from_a_strict_boolean(stream):
    request = ResponsesGuardedRequest.validate_payload({"input": "hello", "stream": stream, "top_logprobs": 0})
    assert request.streams_response is stream


def test_buffered_response_finds_the_only_message_and_preserves_metadata():
    payload = response_payload()
    original = deepcopy(payload)
    projection = ResponsesGuardedResponse.validate_payload(payload)
    target = projection.locate_guarded_message(payload)
    assert target.message.content == "answer"
    assert not target.allows_replacement
    with pytest.raises(ValueError, match="replacement"):
        target.replace_content("changed")
    assert payload == original
    payload["output"].reverse()
    assert (
        ResponsesGuardedResponse.validate_payload(payload).locate_guarded_message(payload).message.content == "answer"
    )


@pytest.mark.parametrize(
    "output",
    [
        [],
        [{"type": "reasoning", "summary": []}],
        [output_message(), output_message()],
        [output_message(role="user")],
        [output_message(content=[])],
        [output_message(content=[{"type": "refusal", "refusal": "hidden"}])],
        [output_message(), {"type": "function_call", "arguments": "hidden"}],
        [output_message(), {"type": "reasoning", "summary": [{"type": "summary_text", "text": "hidden"}]}],
    ],
)
def test_response_rejects_missing_ambiguous_or_unsupported_content(output):
    with pytest.raises(ValidationError):
        ResponsesGuardedResponse.validate_payload({"output": output})


@pytest.mark.parametrize(
    "field,value",
    [
        ("annotations", [{"title": "hidden"}]),
        ("logprobs", [{"token": "hidden"}]),
        ("text", ""),
        ("future_text", "hidden"),
        ("Text", "hidden"),
    ],
)
def test_nested_response_text_is_closed_and_disallows_side_channels(field, value):
    payload = response_payload()
    payload["output"][1]["content"][0][field] = value
    with pytest.raises(ValidationError):
        ResponsesGuardedResponse.validate_payload(payload)


def test_only_declared_configurable_objects_allow_trusted_unknown_fields():
    payload = response_payload()
    payload["output"][1]["future_metadata"] = 1
    ResponsesGuardedResponse.validate_payload(payload, unknown_content_fields=UnknownContentFieldPolicy.ALLOW)
    payload["future_text"] = "hidden"
    with pytest.raises(ValidationError):
        ResponsesGuardedResponse.validate_payload(payload, unknown_content_fields=UnknownContentFieldPolicy.ALLOW)


def test_declared_payload_coverage_is_derived_not_a_second_inventory():
    from nemoguardrails.server.experimental.provider.projection_policy import payload_contract

    for model, direction in ((ResponsesGuardedRequest, "request"), (ResponsesGuardedResponse, "response")):
        assert model.projection_contract == payload_contract(
            model, projection_id=model.projection_contract.projection_id, direction=direction
        )
