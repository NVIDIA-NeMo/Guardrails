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

"""Keep image policy metadata distinct from uninspected image content."""

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


@pytest.fixture(
    params=[
        (RESPONSES_ENDPOINT, "buffered"),
        (LITELLM_RESPONSES_ENDPOINT, "buffered"),
        (LITELLM_CHAT_COMPLETIONS_ENDPOINT, "buffered"),
        (RESPONSES_ENDPOINT, "response.created"),
        (RESPONSES_ENDPOINT, "response.completed"),
        (LITELLM_RESPONSES_ENDPOINT, "response.created"),
        (LITELLM_RESPONSES_ENDPOINT, "response.completed"),
        (LITELLM_CHAT_COMPLETIONS_ENDPOINT, "stream-root"),
        (LITELLM_CHAT_COMPLETIONS_ENDPOINT, "stream-extension"),
    ],
    ids=[
        "native-buffered",
        "litellm-buffered",
        "chat-buffered",
        "native-created",
        "native-completed",
        "litellm-created",
        "litellm-completed",
        "chat-stream-root",
        "chat-stream-extension",
    ],
)
def response_case(request):
    endpoint, shape = request.param
    if endpoint is LITELLM_CHAT_COMPLETIONS_ENDPOINT:
        if shape == "buffered":
            payload = {"choices": [{"message": {"role": "assistant", "content": "answer"}}]}
        else:
            payload = {"object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": "answer"}}]}
        target = payload.setdefault("provider_specific_fields", {}) if shape == "stream-extension" else payload
    elif shape == "buffered":
        payload = response_payload()
        target = payload
    else:
        payload = {"type": shape, "response": {"output": []} if shape == "response.created" else response_payload()}
        target = payload["response"]
    return endpoint, shape, payload, target


def validate_case(endpoint, shape, payload):
    if shape == "buffered":
        return endpoint.guarded_response_model.validate_payload(
            payload, unknown_content_fields=UnknownContentFieldPolicy.FORBID
        )
    prefix = b"event: " + shape.encode() + b"\n" if shape.startswith("response.") else b""
    event = ServerSentEvent.from_bytes(prefix + b"data: " + json.dumps(payload).encode() + b"\n\n")
    return endpoint.stream.classifier.classify_event(event)


@pytest.mark.parametrize("image", [None, {"context_compression": "enabled"}, {"context_compression": "disabled"}])
def test_reviewed_image_metadata_preserves_provider_payload(response_case, image):
    endpoint, shape, payload, target = response_case
    target["image"] = image
    original = deepcopy(payload)
    validate_case(endpoint, shape, payload)
    contract = export_guard_contract(endpoint)["response" if shape == "buffered" else "stream"]
    Draft202012Validator(contract).validate(payload)
    assert payload == original


@pytest.mark.parametrize(
    "image",
    [
        {},
        [],
        "uninspected image",
        {"context_compression": "auto"},
        {"context_compression": True},
        {"context_compression": "disabled", "url": "uninspected image"},
        {"context_compression": "disabled", "text": "uninspected content"},
        {"Context_Compression": "disabled"},
    ],
)
def test_image_metadata_rejects_unreviewed_content_and_invalid_policy(response_case, image):
    endpoint, shape, payload, target = response_case
    target["image"] = image
    with pytest.raises((ValidationError, UnsupportedProviderStream)):
        validate_case(endpoint, shape, payload)
    contract = export_guard_contract(endpoint)["response" if shape == "buffered" else "stream"]
    assert not Draft202012Validator(contract).is_valid(payload)


@pytest.mark.parametrize(
    "endpoint", [RESPONSES_ENDPOINT, LITELLM_RESPONSES_ENDPOINT, LITELLM_CHAT_COMPLETIONS_ENDPOINT]
)
def test_response_image_metadata_does_not_enable_request_options(endpoint):
    payload = {"input": "question", "store": False}
    if endpoint is LITELLM_CHAT_COMPLETIONS_ENDPOINT:
        payload = {"messages": [{"role": "user", "content": "question"}]}
    payload["image"] = {"context_compression": "disabled"}
    with pytest.raises(ValidationError):
        endpoint.guarded_request_model.validate_payload(payload)


def test_native_chat_does_not_inherit_translated_response_metadata():
    payload = {
        "choices": [{"message": {"role": "assistant", "content": "answer"}}],
        "image": {"context_compression": "disabled"},
    }
    with pytest.raises(ValidationError):
        CHAT_COMPLETIONS_ENDPOINT.guarded_response_model.validate_payload(payload)
