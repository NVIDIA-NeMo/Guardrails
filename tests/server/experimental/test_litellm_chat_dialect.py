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

"""Source-backed shape cases, not recordings, for the LiteLLM single-text policy."""

import json
from copy import deepcopy
from pathlib import Path

import httpx
import pytest
import yaml
from fastapi import FastAPI
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from nemoguardrails.server.experimental._content_checker import (
    ContentAllowed,
    ContentInspectionPolicy,
    StreamBufferingPolicy,
)
from nemoguardrails.server.experimental._http_kernel import BufferedHttpResponse
from nemoguardrails.server.experimental._streaming_http import StreamingHttpResponse
from nemoguardrails.server.experimental.provider.contract_export import export_guard_contract
from nemoguardrails.server.experimental.provider.projection_policy import (
    EXTENSION,
    export_payload_schema,
    field_coverage,
    payload_contract,
    text_location,
)
from nemoguardrails.server.experimental.provider.sse import ServerSentEvent
from nemoguardrails.server.experimental.provider.stream import StreamEventRole, UnsupportedProviderStream
from nemoguardrails.server.experimental.provider.types import UnknownContentFieldPolicy
from nemoguardrails.server.experimental.providers.openai.chat_completions.endpoint import CHAT_COMPLETIONS_ENDPOINT
from nemoguardrails.server.experimental.providers.openai.dialects.litellm.chat_completions import (
    LITELLM_CHAT_COMPLETIONS_ENDPOINT,
)
from nemoguardrails.server.experimental.providers.openai.dialects.litellm.stream import (
    STREAM_CLASSIFIER,
    LiteLLMStreamPayloadProjection,
)
from nemoguardrails.server.experimental.providers.openai.integration import create_openai_chat_router


def response_payload():
    """Represent the reviewed containers without copying recorded provider text."""
    return {
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "answer",
                    "provider_specific_fields": {"refusal": None},
                },
                "provider_specific_fields": {},
            }
        ],
        "usage": {"completion_tokens_details": {"reasoning_tokens": 3}},
    }


def response_model():
    return LITELLM_CHAT_COMPLETIONS_ENDPOINT.guarded_response_model


@pytest.mark.parametrize("location", ["message", "choice"])
@pytest.mark.parametrize("value", ["absent", None, {}])
def test_optional_extension_containers(location, value):
    payload = response_payload()
    target = payload["choices"][0]
    if location == "message":
        target = target["message"]
    if value == "absent":
        target.pop("provider_specific_fields")
    else:
        target["provider_specific_fields"] = value
    response_model().validate_payload(payload)
    Draft202012Validator(export_guard_contract(LITELLM_CHAT_COMPLETIONS_ENDPOINT)["response"]).validate(payload)


@pytest.mark.parametrize("location", ["message", "choice"])
@pytest.mark.parametrize(
    "value",
    [
        {"reasoning": "hidden"},
        {"reasoning_content": "hidden"},
        {"refusal": "hidden"},
        {"token_ids": [123]},
        {"tools": []},
        {"unknown": None},
        {"Refusal": None},
        [],
        "hidden",
        False,
    ],
)
@pytest.mark.parametrize("override", list(UnknownContentFieldPolicy))
def test_extension_containers_are_never_opaque(location, value, override):
    payload = response_payload()
    target = payload["choices"][0]
    if location == "message":
        target = target["message"]
    target["provider_specific_fields"] = value
    with pytest.raises(ValidationError):
        response_model().validate_payload(payload, unknown_content_fields=override)
    assert not Draft202012Validator(export_guard_contract(LITELLM_CHAT_COMPLETIONS_ENDPOINT)["response"]).is_valid(
        payload
    )


@pytest.mark.parametrize("location", ["root", "choice", "message"])
@pytest.mark.parametrize("field", ["reasoning", "token_ids", "Provider_Specific_Fields"])
def test_unreviewed_response_fields_remain_closed(location, field):
    payload = response_payload()
    target = payload
    if location != "root":
        target = payload["choices"][0]
    if location == "message":
        target = target["message"]
    target[field] = "hidden"
    with pytest.raises(ValidationError):
        response_model().validate_payload(payload, unknown_content_fields=UnknownContentFieldPolicy.ALLOW)


def test_dialect_rebinds_nested_models_without_mutating_openai():
    endpoint = LITELLM_CHAT_COMPLETIONS_ENDPOINT
    model = response_model()
    payload = response_payload()
    original = deepcopy(payload)
    model.validate_payload(payload)
    assert payload == original
    with pytest.raises(ValidationError):
        CHAT_COMPLETIONS_ENDPOINT.guarded_response_model.validate_payload(payload)
    assert endpoint.guarded_request_model is CHAT_COMPLETIONS_ENDPOINT.guarded_request_model
    assert endpoint.stream.hooks_factory is CHAT_COMPLETIONS_ENDPOINT.stream.hooks_factory
    assert endpoint.stream.classifier is STREAM_CLASSIFIER
    assert STREAM_CLASSIFIER.contract.shapes == CHAT_COMPLETIONS_ENDPOINT.stream.classifier.contract.shapes
    assert model.guarded_text_location == text_location(model)
    assert model.guarded_text_location == CHAT_COMPLETIONS_ENDPOINT.guarded_response_model.guarded_text_location
    assert model.projection_contract == payload_contract(
        model, projection_id=model.projection_contract.projection_id, direction="response"
    )


def test_export_resolves_the_complete_policy():
    document = export_guard_contract(LITELLM_CHAT_COMPLETIONS_ENDPOINT)
    base = export_guard_contract(CHAT_COMPLETIONS_ENDPOINT)
    assert document["request"] == base["request"]
    assert document["stream"] != base["stream"]
    assert document["profile"] == "single_text.v1"
    assert document["integration"]["name"] == "litellm_chat_completions"
    choice = document["response"]["properties"]["choices"]["items"]
    for container in (choice, choice["properties"]["message"]):
        extension = container["properties"]["provider_specific_fields"]
        assert extension[EXTENSION]["classification"] == "constrained"
        object_schema = next(branch for branch in extension["oneOf"] if branch["type"] == "object")
        assert object_schema["additionalProperties"] is False
    Draft202012Validator(document["response"]).validate(response_payload())


def test_reviewed_choice_metadata_is_opaque_in_runtime_and_export():
    payload = response_payload()
    payload["choices"][0]["provider_specific_fields"] = {"native_finish_reason": "end_turn"}
    response_model().validate_payload(payload)
    schema = export_guard_contract(LITELLM_CHAT_COMPLETIONS_ENDPOINT)["response"]
    extension = schema["properties"]["choices"]["items"]["properties"]["provider_specific_fields"]
    fields = next(branch for branch in extension["oneOf"] if branch["type"] == "object")
    assert fields["properties"]["native_finish_reason"][EXTENSION]["classification"] == "opaque"
    Draft202012Validator(schema).validate(payload)


@pytest.mark.parametrize("field", ["images", "thinking_blocks", "reasoning_items"])
@pytest.mark.parametrize("value", [None, [], ["uninspected"], "uninspected", {}])
def test_buffered_source_content_fields_only_allow_empty_values(field, value):
    payload = response_payload()
    payload["choices"][0]["message"][field] = value
    valid = value is None or value == []
    if valid:
        response_model().validate_payload(payload)
    else:
        with pytest.raises(ValidationError):
            response_model().validate_payload(payload)
    assert (
        Draft202012Validator(export_guard_contract(LITELLM_CHAT_COMPLETIONS_ENDPOINT)["response"]).is_valid(payload)
        == valid
    )


@pytest.mark.parametrize("value", [None, [], ["uninspected"]])
def test_copied_thinking_blocks_are_checked_inside_message_extensions(value):
    payload = response_payload()
    payload["choices"][0]["message"]["provider_specific_fields"]["thinking_blocks"] = value
    if value:
        with pytest.raises(ValidationError):
            response_model().validate_payload(payload)
    else:
        response_model().validate_payload(payload)


def stream_payload():
    return {
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": {"content": "answer"}, "finish_reason": None}],
    }


def classify(payload):
    return STREAM_CLASSIFIER.classify_event(
        ServerSentEvent.from_bytes(b"data: " + json.dumps(payload).encode() + b"\n\n")
    )


@pytest.mark.parametrize("location", ["root", "delta"])
@pytest.mark.parametrize(
    "value", [None, {}, {"reasoning": "uninspected"}, {"native_finish_reason": "stop"}, [], "hidden"]
)
def test_stream_extension_containers_are_closed(location, value):
    payload = stream_payload()
    target = payload if location == "root" else payload["choices"][0]["delta"]
    target["provider_specific_fields"] = value
    valid = value is None or value == {}
    if valid:
        assert classify(payload).text == "answer"
    else:
        with pytest.raises(UnsupportedProviderStream):
            classify(payload)
    schema = export_payload_schema(
        LiteLLMStreamPayloadProjection, projection_id=STREAM_CLASSIFIER.contract.projection_id
    )
    assert Draft202012Validator(schema).is_valid(payload) == valid


@pytest.mark.parametrize("field", ["images", "thinking_blocks", "reasoning_items", "annotations"])
@pytest.mark.parametrize("value", [None, [], ["uninspected"], "uninspected"])
def test_stream_source_content_fields_only_allow_empty_values(field, value):
    payload = stream_payload()
    payload["choices"][0]["delta"][field] = value
    if value is None or value == []:
        assert classify(payload).text == "answer"
    else:
        with pytest.raises(UnsupportedProviderStream):
            classify(payload)


def test_stream_extensions_do_not_change_roles_or_openai_declarations():
    payload = stream_payload()
    payload["provider_specific_fields"] = None
    assert classify(payload).role is StreamEventRole.GUARDED_TEXT
    event = ServerSentEvent.from_bytes(b"data: " + json.dumps(payload).encode() + b"\n\n")
    with pytest.raises(UnsupportedProviderStream):
        CHAT_COMPLETIONS_ENDPOINT.stream.classifier.classify_event(event)
    payload["choices"] = []
    payload["usage"] = {"completion_tokens_details": {"reasoning_tokens": 4}, "cost": 0.01}
    assert classify(payload).role is StreamEventRole.OPAQUE_METADATA
    assert STREAM_CLASSIFIER.contract.fields == field_coverage(LiteLLMStreamPayloadProjection)
    assert classify({"error": {"message": "failed"}}).role is StreamEventRole.PROVIDER_ERROR


@pytest.mark.parametrize("field", ["guardrails", "caching", "num_retries", "context_window_fallback_dict", "fallbacks"])
def test_litellm_request_controls_are_not_implicitly_enabled(field):
    payload = {"messages": [{"role": "user", "content": "question"}], field: None}
    with pytest.raises(ValidationError):
        LITELLM_CHAT_COMPLETIONS_ENDPOINT.guarded_request_model.validate_payload(payload)


def test_export_matches_format_and_generated_artifact():
    contracts = Path(__file__).parents[3] / "nemoguardrails/server/experimental/contracts"
    document = export_guard_contract(LITELLM_CHAT_COMPLETIONS_ENDPOINT)
    schema = json.loads((contracts / "guard-contract.schema.json").read_text())
    Draft202012Validator(schema).validate(document)
    assert yaml.safe_load((contracts / "openai/_generated/litellm-chat-completions.guard.yaml").read_text()) == document


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("inspect", [False, True])
@pytest.mark.asyncio
async def test_dialect_does_not_enable_extra_request_content(stream, inspect):
    async def dispatch(_request):
        raise AssertionError("Unreviewed prompt content must not reach the provider")

    app = FastAPI()
    checker = Checker(inspect)
    app.include_router(
        create_openai_chat_router(
            endpoint=LITELLM_CHAT_COMPLETIONS_ENDPOINT,
            checker=checker,
            dispatch=dispatch,
            stream_dispatch=dispatch,
        )
    )
    payload = {
        "messages": [{"role": "user", "content": "question"}],
        "stream": stream,
        "chat_template_kwargs": {"additional_prompt": "hidden"},
    }
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://proxy.test") as client:
        response = await client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "unsupported_chat_completions_shape"
    assert checker.calls == []


@pytest.mark.asyncio
async def test_default_router_does_not_select_dialect_from_url_or_payload():
    async def dispatch(_request):
        return BufferedHttpResponse(
            200, ((b"content-type", b"application/json"),), json.dumps(response_payload()).encode()
        )

    app = FastAPI()
    app.include_router(create_openai_chat_router(checker=Checker(), dispatch=dispatch))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://litellm.test") as client:
        response = await client.post(
            "/v1/chat/completions", json={"messages": [{"role": "user", "content": "question"}]}
        )
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "unsupported_chat_completions_response_shape"


class Checker:
    """Observe the selected subjects without calling guardrail services."""

    def __init__(self, inspect=True):
        self.inspect = inspect
        self.calls = []

    def inspection_policy(self):
        return ContentInspectionPolicy(
            self.inspect, self.inspect, StreamBufferingPolicy(1, 0) if self.inspect else None
        )

    async def check_input(self, check):
        self.calls.append(("input", check.message.content))
        return ContentAllowed()

    async def check_output(self, check):
        self.calls.append(("output", check.output_content))
        return ContentAllowed()


class Source:
    """Own a finite offline byte source used by the real streaming pipeline."""

    def __init__(self, body):
        self.chunks = iter(body[index : index + 7] for index in range(0, len(body), 7))
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self.chunks)
        except StopIteration:
            raise StopAsyncIteration from None

    async def aclose(self):
        self.closed = True


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("inspect", [False, True])
@pytest.mark.parametrize("hidden", [False, True])
async def test_explicit_dialect_selection_preserves_or_rejects_provider_bytes(stream, inspect, hidden):
    checker = Checker(inspect)
    if stream:
        delta = {"content": "answer", "provider_specific_fields": None, "thinking_blocks": []}
        if hidden:
            delta["provider_specific_fields"] = {"reasoning": "hidden"}
        event = {"object": "chat.completion.chunk", "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}
        event["provider_specific_fields"] = {}
        finish = {"object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
        body = b"".join(b"data: " + json.dumps(value).encode() + b"\n\n" for value in (event, finish))
        body += b"data: [DONE]\n\n"
    else:
        payload = response_payload()
        payload["choices"][0]["provider_specific_fields"]["native_finish_reason"] = "end_turn"
        if hidden:
            payload["choices"][0]["message"]["provider_specific_fields"]["reasoning"] = "hidden"
        body = json.dumps(payload).encode()
    source = Source(body)
    dispatched = []

    async def dispatch(request):
        assert not stream
        dispatched.append(request)
        return BufferedHttpResponse(200, ((b"content-type", b"application/json"),), body)

    async def stream_dispatch(request):
        assert stream
        dispatched.append(request)
        return StreamingHttpResponse(200, ((b"content-type", b"text/event-stream"),), source)

    app = FastAPI()
    app.include_router(
        create_openai_chat_router(
            endpoint=LITELLM_CHAT_COMPLETIONS_ENDPOINT,
            checker=checker,
            dispatch=dispatch,
            stream_dispatch=stream_dispatch,
        )
    )
    request_body = json.dumps({"messages": [{"role": "user", "content": "question"}], "stream": stream}).encode()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://proxy.test") as client:
        response = await client.post(
            "/v1/chat/completions", content=request_body, headers={"content-type": "application/json"}
        )
    assert dispatched[0].body == request_body
    if hidden and inspect:
        assert b"hidden" not in response.content
        assert b"unsupported_chat_completions_response_shape" in response.content
        assert checker.calls == [("input", "question")]
    else:
        assert response.status_code == 200
        assert response.content == body
        assert checker.calls == ([("input", "question"), ("output", "answer")] if inspect else [])
    if stream:
        assert source.closed
