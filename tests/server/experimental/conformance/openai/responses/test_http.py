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

from tests.server.experimental.conformance.openai.responses.support import (
    ENDPOINTS,
    INSPECTION_MODES,
    INVENTORY,
    UNINSPECTED,
    events,
    injected_response,
    recording,
)

pytestmark = [pytest.mark.asyncio, pytest.mark.block_network(allowed_hosts=[r"^127\.0\.0\.1$", r"^::1$"])]


@pytest.mark.parametrize("stream", [False, True], ids=["buffered", "streaming"])
@pytest.mark.parametrize("endpoint", ENDPOINTS)
@pytest.mark.parametrize("inspect_input,inspect_output", INSPECTION_MODES)
async def test_unmodified_native_recording_passes_the_same_probe_path(stream, endpoint, inspect_input, inspect_output):
    request, body = recording(stream)
    response, checker, sent = await injected_response(
        request, body, endpoint=endpoint, inspect_input=inspect_input, inspect_output=inspect_output
    )
    assert len(sent) == 1
    assert json.loads(sent[0].content) == request
    if stream:
        payloads = [json.loads(event.data) for event in await events(body)]
        text = "".join(p["delta"] for p in payloads if p["type"] == "response.output_text.delta")
    else:
        message = next(item for item in json.loads(body)["output"] if item["type"] == "message")
        text = message["content"][0]["text"]
    assert response.status_code == 200
    assert response.content == body
    expected = [("input", request["input"])] if inspect_input else []
    if inspect_output:
        expected.append(("output", text))
    assert checker.text_calls == expected
    for kind, check in checker.calls:
        if kind == "output":
            assert check.input_message.role == "user"
            assert check.input_message.content == request["input"]


async def test_native_observations_distinguish_metadata_from_gateway_annotations():
    declared = set(INVENTORY["schema_fields"]["Response"])
    _, buffered = recording(False)
    payload = json.loads(buffered)
    assert payload.keys() - declared == {
        "access_programs",
        "billing",
        "frequency_penalty",
        "presence_penalty",
        "store",
        "tool_usage",
    }
    assert payload["store"] is False
    assert payload["access_programs"] is None
    assert payload["billing"] == {"payer": "developer"}
    assert payload["frequency_penalty"] == payload["presence_penalty"] == 0
    assert set(payload["tool_usage"]) == {"image_gen", "web_search"}
    _, streaming = recording(True)
    lifecycle = []
    for event in await events(streaming):
        payload = json.loads(event.data)
        assert "model" not in payload
        assert "provider_specific_fields" not in payload
        if payload["type"] == "response.output_text.delta":
            assert payload.keys() - set(INVENTORY["schema_fields"]["ResponseTextDeltaEvent"]) == {"obfuscation"}
        if "response" in payload:
            lifecycle.append(payload["type"])
            assert payload["response"].keys() - declared == {
                "access_programs",
                "frequency_penalty",
                "presence_penalty",
                "store",
                "tool_usage",
            }
    assert lifecycle == ["response.created", "response.in_progress", "response.completed"]


@pytest.mark.parametrize(
    "path,value",
    [
        (("output", 0, "summary"), [{"type": "summary_text", "text": UNINSPECTED}]),
        (("output", 0, "content"), [{"type": "reasoning_text", "text": UNINSPECTED}]),
        (("output", 1, "content", 0, "annotations"), [{"type": "url_citation", "title": UNINSPECTED}]),
        (("output", 1, "content", 0, "logprobs"), [{"token": UNINSPECTED}]),
        (("output", 1, "content", 0, "future_text"), UNINSPECTED),
        (("output", 1, "content"), [{"type": "refusal", "refusal": UNINSPECTED}]),
        (("output_text",), UNINSPECTED),
        (("future_output",), UNINSPECTED),
        (("Output",), UNINSPECTED),
    ],
)
@pytest.mark.parametrize("endpoint", ENDPOINTS)
@pytest.mark.parametrize("inspect_input,inspect_output", INSPECTION_MODES)
async def test_synthetic_buffered_content_mutations_of_native_recording_fail_closed(
    path, value, endpoint, inspect_input, inspect_output
):
    request, body = recording(False)
    payload = json.loads(body)
    assert [item["type"] for item in payload["output"]] == ["reasoning", "message"]
    node = payload
    for part in path[:-1]:
        node = node[part]
    node[path[-1]] = value
    mutated = json.dumps(payload).encode()
    response, checker, sent = await injected_response(
        request, mutated, endpoint=endpoint, inspect_input=inspect_input, inspect_output=inspect_output
    )
    assert len(sent) == 1
    if not inspect_output:
        assert response.status_code == 200
        assert response.content == mutated
        assert checker.text_calls == ([("input", request["input"])] if inspect_input else [])
        return
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "unsupported_responses_response_shape"
    assert UNINSPECTED not in response.text
    assert checker.text_calls == ([("input", request["input"])] if inspect_input else [])


@pytest.mark.parametrize("probe", ["reasoning-summary", "done-text", "completed-text", "unreviewed-delta"])
@pytest.mark.parametrize("endpoint", ENDPOINTS)
@pytest.mark.parametrize("inspect_input,inspect_output", INSPECTION_MODES)
async def test_synthetic_stream_mutations_of_native_recording_cannot_escape_as_metadata(
    probe, endpoint, inspect_input, inspect_output
):
    request, body = recording(True)
    payloads = [json.loads(event.data) for event in await events(body)]
    if probe == "reasoning-summary":
        target = next(
            p for p in payloads if p["type"] == "response.output_item.done" and p["item"]["type"] == "reasoning"
        )
        target["item"]["summary"] = [{"type": "summary_text", "text": UNINSPECTED}]
    elif probe == "done-text":
        target = next(p for p in payloads if p["type"] == "response.output_text.done")
        target["text"] = UNINSPECTED
    elif probe == "completed-text":
        target = payloads[-1]
        assert target["type"] == "response.completed"
        message = next(item for item in target["response"]["output"] if item["type"] == "message")
        message["content"][0]["text"] = UNINSPECTED
    else:
        target = next(p for p in payloads if p["type"] == "response.output_text.delta")
        target["type"] = "response.reasoning_summary_text.delta"
        target["delta"] = UNINSPECTED
    mutated = b"".join(
        b"event: " + payload["type"].encode() + b"\ndata: " + json.dumps(payload).encode() + b"\n\n"
        for payload in payloads
    )
    response, checker, sent = await injected_response(
        request, mutated, endpoint=endpoint, inspect_input=inspect_input, inspect_output=inspect_output
    )
    assert len(sent) == 1
    if not inspect_output:
        assert response.status_code == 200
        assert response.content == mutated
        assert checker.text_calls == ([("input", request["input"])] if inspect_input else [])
        return
    assert response.status_code == 200
    (error,) = await events(response.content)
    assert error.event_type == b"error"
    assert json.loads(error.data)["code"] == "unsupported_responses_response_shape"
    assert UNINSPECTED not in response.text
    assert checker.text_calls == ([("input", request["input"])] if inspect_input else [])


@pytest.mark.parametrize(
    "updates",
    [
        {"instructions": "uninspected instructions"},
        {"previous_response_id": "resp_history"},
        {"conversation": "conv_history"},
        {"prompt": {"id": "pmpt_hidden"}},
        {"input": [{"role": "user", "content": "uninspected array"}]},
        {"tools": [{"type": "function", "name": "hidden", "parameters": {}}]},
        {"reasoning": {"summary": "auto"}},
        {"text": {"format": {"type": "json_schema", "schema": {}}}},
        {"include": ["reasoning.encrypted_content"]},
        {"future_input": "unreviewed text"},
    ],
)
@pytest.mark.parametrize("endpoint", ENDPOINTS)
@pytest.mark.parametrize("stream", [False, True], ids=["buffered", "streaming"])
@pytest.mark.parametrize("inspect_input,inspect_output", INSPECTION_MODES)
async def test_request_content_rejected_before_dispatch(updates, endpoint, stream, inspect_input, inspect_output):
    request, body = recording(stream)
    request.update(updates)
    response, checker, sent = await injected_response(
        request, body, endpoint=endpoint, inspect_input=inspect_input, inspect_output=inspect_output
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "unsupported_responses_shape"
    assert sent == []
    assert checker.calls == []
