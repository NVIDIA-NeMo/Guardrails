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

from nemoguardrails.server.experimental.provider.projection_policy import field_coverage
from nemoguardrails.server.experimental.providers.openai.dialects.litellm.responses import LITELLM_RESPONSES_ENDPOINT
from nemoguardrails.server.experimental.providers.openai.responses.endpoint import RESPONSES_ENDPOINT
from tests.server.experimental.conformance.openai.responses.support import (
    ENDPOINTS,
    INSPECTION_MODES,
    UNINSPECTED,
    assert_rejected,
    events,
    injected_response,
    recording,
    sse_bytes,
)

pytestmark = [pytest.mark.asyncio, pytest.mark.block_network(allowed_hosts=[r"^127\.0\.0\.1$", r"^::1$"])]


def test_dialect_adds_only_event_model_metadata_to_the_bound_models():
    assert LITELLM_RESPONSES_ENDPOINT.guarded_request_model is RESPONSES_ENDPOINT.guarded_request_model
    assert LITELLM_RESPONSES_ENDPOINT.guarded_response_model is RESPONSES_ENDPOINT.guarded_response_model
    base_rules = RESPONSES_ENDPOINT.stream.classifier.rules
    dialect_rules = LITELLM_RESPONSES_ENDPOINT.stream.classifier.rules
    assert len(base_rules) == len(dialect_rules) == 15
    for base, dialect in zip(base_rules, dialect_rules):
        native = field_coverage(base.model)
        gateway = field_coverage(dialect.model)
        assert gateway.opaque_fields == native.opaque_fields | {"model"}
        assert gateway.guarded_fields == native.guarded_fields
        assert gateway.constrained_fields == native.constrained_fields
        assert gateway.local_extension_fields == native.local_extension_fields
        if "response" in base.model.model_fields:
            assert dialect.model.model_fields["response"].annotation is base.model.model_fields["response"].annotation


@pytest.mark.parametrize("endpoint", ENDPOINTS)
@pytest.mark.parametrize("stream", [False, True], ids=["buffered", "streaming"])
@pytest.mark.parametrize("inspect_input,inspect_output", INSPECTION_MODES)
async def test_gateway_recording_requires_dialect_only_for_annotated_events(
    endpoint, stream, inspect_input, inspect_output
):
    request, body = recording(stream, gateway=True)
    response, checker, sent = await injected_response(
        request, body, endpoint=endpoint, inspect_input=inspect_input, inspect_output=inspect_output
    )
    assert len(sent) == 1
    assert json.loads(sent[0].content) == request
    if stream and inspect_output and endpoint is RESPONSES_ENDPOINT:
        await assert_rejected(response, checker, request, inspect_input=inspect_input)
        return
    assert response.status_code == 200
    assert response.content == body
    if stream:
        payloads = [json.loads(event.data) for event in await events(body) if event.data not in (None, b"[DONE]")]
        assert all(payload["model"] == request["model"] for payload in payloads)
        text = "".join(payload["delta"] for payload in payloads if payload["type"] == "response.output_text.delta")
    else:
        message = next(item for item in json.loads(body)["output"] if item["type"] == "message")
        text = message["content"][0]["text"]
    expected = [("input", request["input"])] if inspect_input else []
    if inspect_output:
        expected.append(("output", text))
    assert checker.text_calls == expected


@pytest.mark.parametrize("inspect_input,inspect_output", INSPECTION_MODES)
@pytest.mark.parametrize(
    "field,value",
    [
        ("model", None),
        ("model", {"opaque": UNINSPECTED}),
        ("Model", "alias"),
        ("provider_specific_fields", {}),
        ("future_content", UNINSPECTED),
    ],
)
async def test_event_annotation_does_not_authorize_other_containers(field, value, inspect_input, inspect_output):
    request, body = recording(True, gateway=True)
    payloads = [json.loads(event.data) for event in await events(body) if event.data not in (None, b"[DONE]")]
    for payload in payloads:
        payload[field] = value
    mutated = sse_bytes(payloads)
    response, checker, sent = await injected_response(
        request,
        mutated,
        endpoint=LITELLM_RESPONSES_ENDPOINT,
        inspect_input=inspect_input,
        inspect_output=inspect_output,
    )
    assert len(sent) == 1
    if not inspect_output or field == "model":
        assert response.status_code == 200
        assert response.content == mutated
        expected = "".join(p["delta"] for p in payloads if p["type"] == "response.output_text.delta")
        expected_calls = [("input", request["input"])] if inspect_input else []
        if inspect_output:
            expected_calls.append(("output", expected))
        assert checker.text_calls == expected_calls
    else:
        await assert_rejected(response, checker, request, inspect_input=inspect_input)


@pytest.mark.parametrize("inspect_input,inspect_output", INSPECTION_MODES)
@pytest.mark.parametrize("probe", ["reasoning-summary", "completed-text", "nested-error", "nested-error-after-text"])
async def test_gateway_metadata_cannot_hide_new_text_or_change_error_shape(probe, inspect_input, inspect_output):
    request, body = recording(True, gateway=True)
    payloads = [json.loads(event.data) for event in await events(body) if event.data not in (None, b"[DONE]")]
    if probe == "reasoning-summary":
        target = next(
            p for p in payloads if p["type"] == "response.output_item.done" and p["item"]["type"] == "reasoning"
        )
        target["item"]["summary"] = [{"type": "summary_text", "text": UNINSPECTED}]
    elif probe == "completed-text":
        target = next(p for p in payloads if p["type"] == "response.completed")
        message = next(item for item in target["response"]["output"] if item["type"] == "message")
        message["content"][0]["text"] = UNINSPECTED
    else:
        if probe == "nested-error-after-text":
            delta_index = next(
                i for i, payload in enumerate(payloads) if payload["type"] == "response.output_text.delta"
            )
            payloads = payloads[: delta_index + 1]
            assert payloads[-1]["delta"]
        else:
            payloads = []
        payloads.append(
            {"type": "error", "model": request["model"], "error": {"message": UNINSPECTED, "code": "provider_error"}}
        )
    mutated = sse_bytes(payloads)
    response, checker, sent = await injected_response(
        request,
        mutated,
        endpoint=LITELLM_RESPONSES_ENDPOINT,
        inspect_input=inspect_input,
        inspect_output=inspect_output,
    )
    assert len(sent) == 1
    if inspect_output:
        await assert_rejected(response, checker, request, inspect_input=inspect_input)
        if probe == "nested-error-after-text":
            assert sse_bytes([payloads[-2]]) not in response.content
    else:
        assert response.status_code == 200
        assert response.content == mutated
        assert checker.text_calls == ([("input", request["input"])] if inspect_input else [])
