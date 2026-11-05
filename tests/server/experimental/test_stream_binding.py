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

"""Characterize provider stream policy binding and custom classifier compatibility."""

import json
from dataclasses import replace

import httpx
import pytest
from fastapi import FastAPI

from nemoguardrails.server.experimental._content_checker import ContentInspectionPolicy, StreamBufferingPolicy
from nemoguardrails.server.experimental._guarded_proxy import create_guarded_http_operation
from nemoguardrails.server.experimental._streaming_http import StreamingHttpResponse
from nemoguardrails.server.experimental.provider.contract_export import export_guard_contract
from nemoguardrails.server.experimental.provider.sse import ServerSentEvent
from nemoguardrails.server.experimental.provider.stream import StreamBinding, UnsupportedProviderStream
from nemoguardrails.server.experimental.provider.types import UnknownContentFieldPolicy
from nemoguardrails.server.experimental.providers.openai.chat_completions.endpoint import CHAT_COMPLETIONS_ENDPOINT
from nemoguardrails.server.experimental.providers.openai.chat_completions.stream_classifier import STREAM_CLASSIFIER
from nemoguardrails.server.experimental.providers.openai.chat_completions.stream_hooks import ChatCompletionsStreamHooks
from nemoguardrails.server.experimental.providers.openai.errors import OPENAI_ERROR_MAPPING
from nemoguardrails.server.experimental.providers.openai.integration import create_openai_chat_router
from tests.server.experimental.chat_fixtures import request_body
from tests.server.experimental.checkers import StaticChecker
from tests.server.experimental.test_openai_chat_streaming_integration import Source


class ManualClassifier:
    """Keep a handwritten classifier's strict field policy independent of deployment selection."""

    contract = STREAM_CLASSIFIER.contract

    def classify_event(self, event):
        return STREAM_CLASSIFIER.classify_event(event)


class ConfigurableManualClassifier(ManualClassifier):
    """Opt a handwritten classifier into the policy through its structural capability."""

    def __init__(self, delegate=STREAM_CLASSIFIER):
        self.delegate = delegate

    def with_unknown_fields(self, policy):
        return ConfigurableManualClassifier(self.delegate.with_unknown_fields(policy))

    def classify_event(self, event):
        return self.delegate.classify_event(event)


def metadata_event():
    payload = {
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": {"content": "answer"}}],
        "future_metadata": {"counter": 1},
    }
    return b"data: " + json.dumps(payload).encode() + b"\n\n"


async def exchange(classifier, policy):
    source = Source([metadata_event(), b"data: [DONE]\n\n"])
    checker = StaticChecker(ContentInspectionPolicy(True, True, StreamBufferingPolicy(1, 0)))

    async def buffered(_request):
        raise AssertionError("A streaming request must use streaming dispatch")

    async def streaming(_request):
        return StreamingHttpResponse(200, ((b"content-type", b"text/event-stream"),), source)

    endpoint = replace(CHAT_COMPLETIONS_ENDPOINT, stream=StreamBinding(classifier, ChatCompletionsStreamHooks))
    app = FastAPI()
    app.include_router(
        create_openai_chat_router(
            checker=checker,
            dispatch=buffered,
            stream_dispatch=streaming,
            endpoint=endpoint,
            unknown_fields=policy,
        )
    )
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy.test") as client:
        response = await client.post(
            "/v1/chat/completions", content=request_body(), headers={"content-type": "application/json"}
        )
    assert source.closed
    return response, checker


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", list(UnknownContentFieldPolicy))
@pytest.mark.parametrize("manual", [False, True], ids=["typed", "legacy-handwritten"])
async def test_http_policy_preserves_existing_classifier_behavior(policy, manual):
    response, checker = await exchange(ManualClassifier() if manual else STREAM_CLASSIFIER, policy)
    assert response.status_code == 200
    if not manual and policy is UnknownContentFieldPolicy.ALLOW:
        assert response.content == metadata_event() + b"data: [DONE]\n\n"
        assert checker.text_calls[-1] == ("output", "answer")
    else:
        assert "unsupported_chat_completions_response_shape" in response.text
        assert b"answer" not in response.content
        assert not any(kind == "output" for kind, _ in checker.calls)
    assert STREAM_CLASSIFIER.unknown_fields is UnknownContentFieldPolicy.FORBID


@pytest.mark.parametrize("policy", list(UnknownContentFieldPolicy))
def test_operation_binding_does_not_instantiate_protocol_hooks(policy):
    def forbidden():
        raise AssertionError("Binding must not create request-local protocol state")

    endpoint = replace(CHAT_COMPLETIONS_ENDPOINT, stream=StreamBinding(STREAM_CLASSIFIER, forbidden))
    create_guarded_http_operation(
        endpoint,
        OPENAI_ERROR_MAPPING,
        unknown_fields=policy,
        max_stream_event_bytes=1024,
        max_pending_stream_bytes=1024,
    )


def test_direct_source_binding_keeps_strict_validation_and_fresh_hooks():
    binding = StreamBinding(STREAM_CLASSIFIER, ChatCompletionsStreamHooks)
    first, second = binding.create_adapter(), binding.create_adapter()
    assert first.classifier is second.classifier is STREAM_CLASSIFIER
    assert first.hooks is not second.hooks
    with pytest.raises(UnsupportedProviderStream, match="unreviewed fields"):
        first.classify_event(ServerSentEvent.from_bytes(metadata_event()))


@pytest.mark.parametrize("policy", list(UnknownContentFieldPolicy))
def test_native_policy_binding_preserves_source_and_contract(policy):
    source = StreamBinding(STREAM_CLASSIFIER, ChatCompletionsStreamHooks)
    original_contract = export_guard_contract(CHAT_COMPLETIONS_ENDPOINT)
    configured = source.with_unknown_fields(policy)
    assert configured is not source
    assert configured.classifier is not source.classifier
    assert configured.classifier.contract is source.classifier.contract
    assert STREAM_CLASSIFIER.unknown_fields is UnknownContentFieldPolicy.FORBID
    assert export_guard_contract(replace(CHAT_COMPLETIONS_ENDPOINT, stream=configured)) == original_contract

    event = ServerSentEvent.from_bytes(metadata_event())
    if policy is UnknownContentFieldPolicy.ALLOW:
        assert configured.create_adapter().classify_event(event).text == "answer"
    else:
        with pytest.raises(UnsupportedProviderStream, match="unreviewed fields"):
            configured.create_adapter().classify_event(event)
    with pytest.raises(UnsupportedProviderStream, match="unreviewed fields"):
        source.create_adapter().classify_event(event)


@pytest.mark.parametrize("policy", list(UnknownContentFieldPolicy))
def test_legacy_classifier_binding_keeps_its_own_policy(policy):
    binding = StreamBinding(ManualClassifier(), ChatCompletionsStreamHooks)
    assert binding.with_unknown_fields(policy) is binding
    with pytest.raises(UnsupportedProviderStream, match="unreviewed fields"):
        binding.create_adapter().classify_event(ServerSentEvent.from_bytes(metadata_event()))


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", list(UnknownContentFieldPolicy))
async def test_http_binding_configures_structural_custom_classifier(policy):
    classifier = ConfigurableManualClassifier()
    response, checker = await exchange(classifier, policy)
    if policy is UnknownContentFieldPolicy.ALLOW:
        assert response.content == metadata_event() + b"data: [DONE]\n\n"
        assert checker.text_calls[-1] == ("output", "answer")
    else:
        assert "unsupported_chat_completions_response_shape" in response.text
        assert b"answer" not in response.content
    assert classifier.delegate is STREAM_CLASSIFIER
    assert classifier.delegate.unknown_fields is UnknownContentFieldPolicy.FORBID


def test_policy_configuration_errors_propagate_before_dispatch():
    failure = ValueError("This classifier does not support the requested policy")

    class RejectingClassifier(ManualClassifier):
        def with_unknown_fields(self, policy):
            raise failure

    endpoint = replace(
        CHAT_COMPLETIONS_ENDPOINT, stream=StreamBinding(RejectingClassifier(), ChatCompletionsStreamHooks)
    )
    with pytest.raises(ValueError) as raised:
        create_guarded_http_operation(
            endpoint,
            OPENAI_ERROR_MAPPING,
            unknown_fields=UnknownContentFieldPolicy.ALLOW,
            max_stream_event_bytes=1024,
            max_pending_stream_bytes=1024,
        )
    assert raised.value is failure


@pytest.mark.parametrize("result", [None, object()])
def test_policy_binding_rejects_malformed_classifier_results(result):
    class InvalidClassifier(ManualClassifier):
        def with_unknown_fields(self, policy):
            return result

    binding = StreamBinding(InvalidClassifier(), ChatCompletionsStreamHooks)
    with pytest.raises(TypeError, match="stream binding requires a classifier"):
        binding.with_unknown_fields(UnknownContentFieldPolicy.ALLOW)


@pytest.mark.parametrize("classifier", [STREAM_CLASSIFIER, ManualClassifier()])
@pytest.mark.parametrize("invalid", ["strict"])
def test_policy_binding_normalizes_before_configuring_a_classifier(classifier, invalid):
    binding = StreamBinding(classifier, ChatCompletionsStreamHooks)
    with pytest.raises(ValueError, match="not a valid UnknownContentFieldPolicy"):
        binding.with_unknown_fields(invalid)
