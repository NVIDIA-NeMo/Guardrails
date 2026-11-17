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

"""Validate reviewed reasoning-input metadata independently of captured empty lists."""

import json
from copy import deepcopy

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from nemoguardrails.server.experimental.provider.contract_export import export_guard_contract
from nemoguardrails.server.experimental.provider.sse import ServerSentEvent
from nemoguardrails.server.experimental.provider.stream import UnsupportedProviderStream
from nemoguardrails.server.experimental.provider.types import UnknownContentFieldPolicy
from nemoguardrails.server.experimental.providers.openai.chat_completions.endpoint import CHAT_COMPLETIONS_ENDPOINT
from nemoguardrails.server.experimental.providers.openai.dialects.litellm.chat_completions import (
    LITELLM_CHAT_COMPLETIONS_ENDPOINT,
)
from nemoguardrails.server.experimental.providers.openai.dialects.litellm.responses import LITELLM_RESPONSES_ENDPOINT
from nemoguardrails.server.experimental.providers.openai.responses.endpoint import RESPONSES_ENDPOINT
from tests.server.experimental.test_openai_responses_projections import response_payload

REASONS = ["organization_mismatch", "end_user_mismatch", "model_mismatch", "prefix_mismatch", "turn_boundary"]
POPULATED = [{"id": "rs_test", "action": "removed", "reason": reason, "type": "reasoning"} for reason in REASONS]
LIFECYCLE = [
    "response.created",
    "response.in_progress",
    "response.queued",
    "response.completed",
    "response.failed",
    "response.incomplete",
]
CASES = [
    (ep, shape) for ep in [RESPONSES_ENDPOINT, LITELLM_RESPONSES_ENDPOINT] for shape in ["buffered", *LIFECYCLE]
] + [(LITELLM_CHAT_COMPLETIONS_ENDPOINT, shape) for shape in ["buffered", "chunk", "extension"]]


def sse(payload):
    prefix = b"event: " + payload["type"].encode() + b"\n" if "type" in payload else b""
    return ServerSentEvent.from_bytes(prefix + b"data: " + json.dumps(payload).encode() + b"\n\n")


@pytest.fixture(params=CASES, ids=[f"{ep.contract_name}-{shape}" for ep, shape in CASES])
def response_case(request):
    endpoint, shape = request.param
    if endpoint is LITELLM_CHAT_COMPLETIONS_ENDPOINT:
        if shape == "buffered":
            payload = {"choices": [{"message": {"role": "assistant", "content": "answer"}}]}
        else:
            payload = {"object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": "answer"}}]}
        target = payload.setdefault("provider_specific_fields", {}) if shape == "extension" else payload
    elif shape == "buffered":
        payload = response_payload()
        target = payload
    else:
        payload = {"type": shape, "response": response_payload() if shape in LIFECYCLE[3:] else {"output": []}}
        target = payload["response"]
    return endpoint, shape, payload, target


def validate(endpoint, shape, payload):
    if shape == "buffered":
        return endpoint.guarded_response_model.validate_payload(
            payload, unknown_content_fields=UnknownContentFieldPolicy.ALLOW
        )
    return endpoint.stream.classifier.classify_event(sse(payload))


def schema(endpoint, shape):
    return Draft202012Validator(export_guard_contract(endpoint)["response" if shape == "buffered" else "stream"])


@pytest.mark.parametrize("value", ["absent", None, [], POPULATED])
def test_valid_transformations_preserve_payload_and_export(response_case, value):
    endpoint, shape, payload, target = response_case
    if value != "absent":
        target["input_transformations"] = deepcopy(value)
    original = deepcopy(payload)
    validate(endpoint, shape, payload)
    schema(endpoint, shape).validate(payload)
    assert payload == original


INVALID = [False, 1, "reasoning", {}, [None], [[]], [{}]]
INVALID += [[{key: value for key, value in POPULATED[0].items() if key != missing}] for missing in POPULATED[0]]
INVALID += [
    [{**POPULATED[0], key: value}]
    for key, value in [
        ("id", 1),
        ("id", None),
        ("action", "retained"),
        ("type", "message"),
        ("reason", "unknown"),
        ("reason", None),
        ("text", "hidden"),
        ("ID", "rs_other"),
    ]
]


@pytest.mark.parametrize("value", INVALID)
def test_malformed_transformations_reject_in_runtime_and_export(response_case, value):
    endpoint, shape, payload, target = response_case
    target["input_transformations"] = value
    with pytest.raises((ValidationError, UnsupportedProviderStream)):
        validate(endpoint, shape, payload)
    assert not schema(endpoint, shape).is_valid(payload)


@pytest.mark.parametrize("unknown", ["unreviewed_metadata", "Input_Transformations"])
def test_reviewed_transformations_do_not_open_other_fields(response_case, unknown):
    endpoint, shape, payload, target = response_case
    target["input_transformations"] = POPULATED
    target[unknown] = []
    with pytest.raises((ValidationError, UnsupportedProviderStream)):
        validate(endpoint, shape, payload)
    assert not schema(endpoint, shape).is_valid(payload)


@pytest.mark.parametrize("endpoint", [RESPONSES_ENDPOINT, LITELLM_RESPONSES_ENDPOINT])
@pytest.mark.parametrize("failure", ["sequence", "truncated", "after_terminal"])
def test_populated_transformations_preserve_stream_integrity(endpoint, failure):
    adapter = endpoint.stream.create_adapter()
    adapter.classify_event(
        sse(
            {
                "type": "response.created",
                "sequence_number": 0,
                "response": {"output": [], "input_transformations": POPULATED},
            }
        )
    )
    adapter.classify_event(sse({"type": "response.output_text.delta", "sequence_number": 1, "delta": "answer"}))
    if failure == "truncated":
        with pytest.raises(UnsupportedProviderStream):
            adapter.validate_end_of_stream()
        return
    payload = response_payload()
    payload["input_transformations"] = POPULATED
    terminal = sse(
        {"type": "response.completed", "sequence_number": 1 if failure == "sequence" else 2, "response": payload}
    )
    if failure == "after_terminal":
        adapter.classify_event(terminal)
        adapter.validate_end_of_stream()
    with pytest.raises(UnsupportedProviderStream):
        adapter.classify_event(terminal)


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", [RESPONSES_ENDPOINT, LITELLM_RESPONSES_ENDPOINT])
async def test_populated_transformations_cannot_hide_changed_snapshot(endpoint, httpx_mock):
    from nemoguardrails.server.experimental._content_checker import ContentInspectionPolicy, StreamBufferingPolicy
    from tests.server.experimental.checkers import StaticChecker
    from tests.server.experimental.responses_proxy import responses_proxy

    payload = response_payload()
    payload["input_transformations"] = POPULATED
    next(item for item in payload["output"] if item["type"] == "message")["content"][0]["text"] = "uninspected"
    wire = b"".join(
        sse(p).raw
        for p in [
            {"type": "response.created", "response": {"output": [], "input_transformations": POPULATED}},
            {"type": "response.output_text.delta", "delta": "answer"},
            {"type": "response.completed", "response": payload},
        ]
    )
    httpx_mock.add_response(
        url="https://api.openai.com/v1/responses", headers={"content-type": "text/event-stream"}, content=wire
    )
    checker = StaticChecker(ContentInspectionPolicy(True, True, StreamBufferingPolicy(100, 0)))
    async with responses_proxy("test-key", checker, [], endpoint=endpoint) as client:
        response = await client.post("/v1/responses", json={"input": "question", "store": False, "stream": True})
    assert "unsupported_" in response.text
    assert "uninspected" not in response.text


@pytest.mark.parametrize("endpoint", [CHAT_COMPLETIONS_ENDPOINT, LITELLM_CHAT_COMPLETIONS_ENDPOINT])
def test_transformations_do_not_enable_native_chat_or_message_fields(endpoint):
    payload = {"choices": [{"message": {"role": "assistant", "content": "answer", "input_transformations": POPULATED}}]}
    with pytest.raises(ValidationError):
        endpoint.guarded_response_model.validate_payload(payload)
    if endpoint is CHAT_COMPLETIONS_ENDPOINT:
        del payload["choices"][0]["message"]["input_transformations"]
        payload["input_transformations"] = POPULATED
        with pytest.raises(ValidationError):
            endpoint.guarded_response_model.validate_payload(payload)
