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

"""Source-backed synthetic conformance through the actual Chat router and transport.

These are not provider recordings or real Rails verdicts. Expected payloads and
policy restrictions are explicit, independent of generated guard schemas.
"""

import pytest
from jsonschema import Draft202012Validator

from nemoguardrails.server.experimental._content_checker import ContentInspectionPolicy
from nemoguardrails.server.experimental.provider.contract_export import export_guard_contract
from nemoguardrails.server.experimental.provider.projection_policy import field_coverage, field_policy
from nemoguardrails.server.experimental.providers.openai.chat_completions.endpoint import CHAT_COMPLETIONS_ENDPOINT
from tests.server.experimental.checkers import StaticChecker
from tests.server.experimental.conformance.openai.chat_completions.http_support import (
    assert_native_error,
    assert_output_error,
    checker_for,
    proxy,
)
from tests.server.experimental.conformance.openai.chat_completions.support import (
    chat_chunk,
    chat_request,
    chat_response,
    json_bytes,
    source_document,
    source_validator,
    sse_bytes,
)

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.block_network(allowed_hosts=[r"^127\.0\.0\.1$", r"^::1$"]),
]
TEXT = "Cafe\u0301 — “quoted” ☃"
REQUEST_METADATA = {
    "safety_identifier": "test-user",
    "logit_bias": {"42": 1},
    "presence_penalty": 0.5,
    "store": True,
    "stop": ["END"],
    "top_p": 0.5,
    "verbosity": "low",
    "seed": 42,
    "stream_options": {"include_usage": True},
    "moderation": {"model": "test-moderator"},
    "service_tier": "default",
    "user": "test-user",
    "prompt_cache_options": {},
    "temperature": 0.5,
    "max_tokens": 12,
    "max_completion_tokens": 16,
    "frequency_penalty": 0.5,
    "prompt_cache_key": "test-cache",
    "model": "test-model",
    "metadata": {"description": TEXT},
    "prompt_cache_retention": "in_memory",
}
RESPONSE_METADATA = {
    "service_tier": "default",
    "created": 123,
    "object": "chat.completion",
    "system_fingerprint": "test-fingerprint",
    "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
    "id": "chatcmpl-test",
    "model": "test-model",
    "metadata": {"title": TEXT},
    "moderation": None,
}
STREAM_METADATA = {name: value for name, value in RESPONSE_METADATA.items() if name not in {"object", "metadata"}} | {
    "obfuscation": TEXT
}
DISABLED_REQUEST_VALUES = {
    "audio": {"voice": "alloy", "format": "wav"},
    "function_call": "auto",
    "functions": [{"name": "lookup"}],
    "modalities": ["text", "audio"],
    "parallel_tool_calls": True,
    "prediction": {"type": "content", "content": "answer"},
    "response_format": {"type": "json_object"},
    "tool_choice": "auto",
    "tools": [{"type": "function", "function": {"name": "lookup"}}],
    "web_search_options": {"search_context_size": "low"},
    "top_logprobs": 1,
}
RESTRICTED_REQUEST_VALUES = DISABLED_REQUEST_VALUES | {"n": 2, "logprobs": True}
UNSUPPORTED_MESSAGES = [
    pytest.param({"role": "user", "content": "question", "name": "caller"}, id="participant-name"),
    pytest.param({"role": "developer", "content": "question"}, id="developer"),
    pytest.param({"role": "system", "content": "question"}, id="system"),
    pytest.param({"role": "assistant", "content": "question"}, id="assistant"),
    pytest.param({"role": "tool", "content": "question", "tool_call_id": "call_1"}, id="tool"),
    pytest.param({"role": "function", "content": "question", "name": "lookup"}, id="function"),
    pytest.param({"role": "user", "content": [{"type": "text", "text": "question"}]}, id="text-parts"),
    pytest.param(
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://example.test/image"}}]},
        id="image",
    ),
]


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("inspect", [False, True], ids=["uninspected", "inspected"])
@pytest.mark.parametrize("field,value", REQUEST_METADATA.items())
async def test_each_reviewed_request_metadata_field_survives_transport(field, value, inspect, streaming):
    request = chat_request(messages=[{"role": "user", "content": TEXT}], stream=streaming, **{field: value})
    source_validator("CreateChatCompletionRequest").validate(request)
    request_body = json_bytes(request)
    payload = chat_chunk() if streaming else chat_response()
    source_validator("CreateChatCompletionStreamResponse" if streaming else "CreateChatCompletionResponse").validate(
        payload
    )
    provider_body = sse_bytes(payload) if streaming else json_bytes(payload)
    checker = checker_for(inspect)
    async with proxy(checker, provider_body, streaming=streaming) as (client, requests, source):
        result = await client.post(
            "/v1/chat/completions", content=request_body, headers={"content-type": "application/json"}
        )

    assert result.status_code == 200
    assert result.content == provider_body
    assert len(requests) == 1
    assert requests[0].content == request_body
    assert str(requests[0].url) == "https://provider.test/v1/chat/completions"
    assert checker.text_calls == ([("input", TEXT), ("output", "answer")] if inspect else [])
    assert source.closed


@pytest.mark.parametrize(
    "streaming,field,value",
    [(False, field, value) for field, value in RESPONSE_METADATA.items()]
    + [(True, field, value) for field, value in STREAM_METADATA.items()],
)
@pytest.mark.parametrize("inspect", [False, True], ids=["uninspected", "inspected"])
async def test_each_reviewed_response_metadata_field_survives_transport(streaming, field, value, inspect):
    payload = chat_chunk(TEXT, **{field: value}) if streaming else chat_response(**{field: value})
    if not streaming:
        payload["choices"][0]["message"]["content"] = TEXT
    source_validator("CreateChatCompletionStreamResponse" if streaming else "CreateChatCompletionResponse").validate(
        payload
    )
    body = sse_bytes(payload) if streaming else json_bytes(payload)
    checker = checker_for(inspect)
    async with proxy(checker, body, streaming=streaming) as (client, requests, source):
        result = await client.post("/v1/chat/completions", json=chat_request(stream=streaming))

    assert result.status_code == 200
    assert result.content == body
    assert len(requests) == 1
    assert checker.text_calls == ([("input", "question"), ("output", TEXT)] if inspect else [])
    assert source.closed


async def test_explicit_examples_cover_the_reviewed_root_metadata_and_disabled_requests():
    endpoint = CHAT_COMPLETIONS_ENDPOINT
    assert set(REQUEST_METADATA) == field_coverage(endpoint.guarded_request_model).opaque_fields
    assert set(RESPONSE_METADATA) == field_coverage(endpoint.guarded_response_model).opaque_fields
    assert set(STREAM_METADATA) == endpoint.stream.classifier.contract.fields.opaque_fields
    assert set(DISABLED_REQUEST_VALUES) == {
        name
        for name, field in endpoint.guarded_request_model.model_fields.items()
        if field_policy(field).get("gate") == "disabled"
    }


@pytest.mark.parametrize("inspect", [False, True], ids=["uninspected", "inspected"])
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("field,value", RESTRICTED_REQUEST_VALUES.items())
async def test_provider_valid_restricted_request_never_dispatches(field, value, inspect, streaming):
    payload = chat_request(stream=streaming, **{field: value})
    source_validator("CreateChatCompletionRequest").validate(payload)
    assert not Draft202012Validator(export_guard_contract(CHAT_COMPLETIONS_ENDPOINT)["request"]).is_valid(payload)
    checker = checker_for(inspect)
    async with proxy(checker, b"must not be requested", streaming=streaming) as (client, requests, source):
        response = await client.post("/v1/chat/completions", json=payload)

    assert_native_error(response, 422, "unsupported_chat_completions_shape")
    assert requests == []
    assert checker.calls == []
    assert not source.closed


@pytest.mark.parametrize("message", UNSUPPORTED_MESSAGES)
async def test_provider_valid_message_outside_text_profile_is_rejected(message):
    payload = chat_request(messages=[message])
    source_validator("CreateChatCompletionRequest").validate(payload)
    checker = checker_for(True)
    async with proxy(checker, b"must not be requested") as (client, requests, _):
        response = await client.post("/v1/chat/completions", json=payload)
    assert_native_error(response, 422, "unsupported_chat_completions_shape")
    assert requests == []
    assert checker.calls == []


@pytest.mark.parametrize("inspect_input,inspect_output", [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize("location", ["root", "message"])
async def test_unknown_request_content_is_not_a_way_to_disable_the_guard_boundary(
    location, inspect_input, inspect_output
):
    payload = chat_request()
    target = payload if location == "root" else payload["messages"][0]
    target["future_content"] = "unreviewed text"
    checker = StaticChecker(ContentInspectionPolicy(inspect_input, inspect_output))
    async with proxy(checker, b"must not be requested") as (client, requests, _):
        response = await client.post("/v1/chat/completions", json=payload)
    assert_native_error(response, 422, "unsupported_chat_completions_shape")
    assert requests == []
    assert checker.calls == []


@pytest.mark.parametrize("inspect", [False, True], ids=["uninspected", "inspected"])
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("location", ["root", "choice", "content"])
async def test_unknown_response_content_fails_closed_only_when_output_is_inspected(inspect, streaming, location):
    payload = chat_chunk() if streaming else chat_response()
    target = payload
    if location != "root":
        target = payload["choices"][0]
    if location == "content":
        target = target["delta" if streaming else "message"]
    target["future_content"] = "unreviewed text"
    body = sse_bytes(payload) if streaming else json_bytes(payload)
    checker = checker_for(inspect)
    async with proxy(checker, body, streaming=streaming) as (client, requests, source):
        response = await client.post("/v1/chat/completions", json=chat_request(stream=streaming))
    assert len(requests) == 1
    assert source.closed
    if not inspect:
        assert response.status_code == 200
        assert response.content == body
        assert checker.calls == []
    else:
        assert b"unreviewed text" not in response.content
        assert checker.text_calls == [("input", "question")]
        assert_output_error(response, streaming)


@pytest.mark.parametrize(
    "streaming,path,value",
    [
        pytest.param(False, ("choices", 0, "message", "refusal"), "refused", id="response-refusal"),
        pytest.param(
            False,
            ("choices", 0, "message", "function_call"),
            {"name": "lookup", "arguments": "{}"},
            id="response-function",
        ),
        pytest.param(
            False,
            ("choices", 0, "message", "tool_calls"),
            [{"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}],
            id="response-tool",
        ),
        pytest.param(
            False,
            ("choices", 0, "message", "audio"),
            {"id": "audio_1", "expires_at": 0, "data": "AAAA", "transcript": "hidden"},
            id="response-audio",
        ),
        pytest.param(
            False,
            ("choices", 0, "message", "annotations"),
            [
                {
                    "type": "url_citation",
                    "url_citation": {
                        "start_index": 0,
                        "end_index": 1,
                        "url": "https://example.test",
                        "title": "hidden",
                    },
                }
            ],
            id="response-citation",
        ),
        pytest.param(False, ("choices", 0, "logprobs"), {"content": [], "refusal": []}, id="response-logprobs"),
        pytest.param(True, ("choices", 0, "delta", "refusal"), "refused", id="stream-refusal"),
        pytest.param(
            True, ("choices", 0, "delta", "function_call"), {"name": "lookup", "arguments": "{}"}, id="stream-function"
        ),
        pytest.param(
            True,
            ("choices", 0, "delta", "tool_calls"),
            [{"index": 0, "id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}],
            id="stream-tool",
        ),
        pytest.param(True, ("choices", 0, "logprobs"), {"content": [], "refusal": []}, id="stream-logprobs"),
        pytest.param(True, ("choices", 0, "index"), 1, id="stream-choice-index"),
    ],
)
@pytest.mark.parametrize("inspect", [False, True], ids=["uninspected", "inspected"])
async def test_provider_valid_content_outside_profile_is_withheld(streaming, path, value, inspect):
    payload = chat_chunk() if streaming else chat_response()
    target = payload
    for segment in path[:-1]:
        target = target[segment]
    target[path[-1]] = value
    source_validator("CreateChatCompletionStreamResponse" if streaming else "CreateChatCompletionResponse").validate(
        payload
    )
    exported = export_guard_contract(CHAT_COMPLETIONS_ENDPOINT)["stream" if streaming else "response"]
    assert not Draft202012Validator(exported).is_valid(payload)
    body = sse_bytes(payload) if streaming else json_bytes(payload)
    checker = checker_for(inspect)
    async with proxy(checker, body, streaming=streaming) as (client, requests, source):
        response = await client.post("/v1/chat/completions", json=chat_request(stream=streaming))
    assert len(requests) == 1
    assert source.closed
    if inspect:
        assert_output_error(response, streaming)
        assert checker.text_calls == [("input", "question")]
    else:
        assert response.status_code == 200
        assert response.content == body
        assert checker.calls == []


@pytest.mark.parametrize(
    "method,path",
    [
        ("GET", "/v1/chat/completions"),
        ("GET", "/v1/chat/completions/chatcmpl-test"),
        ("POST", "/v1/chat/completions/chatcmpl-test"),
        ("DELETE", "/v1/chat/completions/chatcmpl-test"),
        ("GET", "/v1/chat/completions/chatcmpl-test/messages"),
    ],
)
async def test_other_documented_chat_operations_relay_without_guarded_checks(method, path):
    source_path = path.removeprefix("/v1").replace("chatcmpl-test", "{completion_id}")
    assert method.lower() in source_document()["paths"][source_path]
    body = b'{ "provider_owned": "unchanged" }'
    checker = checker_for(True)
    async with proxy(checker, body) as (client, requests, source):
        result = await client.request(method, path + "?opaque=keep", content=b'{"metadata":{"purpose":"test"}}')
    assert result.status_code == 200
    assert result.content == body
    assert len(requests) == 1
    assert requests[0].method == method
    assert requests[0].url.path == path
    assert requests[0].url.query == b"opaque=keep"
    assert requests[0].content == b'{"metadata":{"purpose":"test"}}'
    assert checker.calls == []
    assert source.closed
