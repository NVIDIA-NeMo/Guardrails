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

"""Translated reasoning retains Responses constraints and closed Chat boundaries."""

from copy import deepcopy

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from nemoguardrails.server.experimental.provider.contract_export import export_guard_contract
from nemoguardrails.server.experimental.provider.types import UnknownContentFieldPolicy
from nemoguardrails.server.experimental.providers.openai.dialects.litellm.chat_completions import (
    LITELLM_CHAT_COMPLETIONS_ENDPOINT,
)
from nemoguardrails.server.experimental.providers.openai.dialects.litellm.stream import LiteLLMStreamPayloadProjection
from tests.server.experimental.test_litellm_chat_dialect import response_payload, stream_payload

REASONING = {"type": "reasoning", "id": "rs_test", "encrypted_content": "opaque-test-value", "summary": []}


def payload_with_reasoning(stream, value):
    payload = stream_payload() if stream else response_payload()
    payload["choices"][0]["delta" if stream else "message"]["reasoning_items"] = deepcopy(value)
    return payload


def model_for(stream):
    return LiteLLMStreamPayloadProjection if stream else LITELLM_CHAT_COMPLETIONS_ENDPOINT.guarded_response_model


def validator_for(stream):
    contract = export_guard_contract(LITELLM_CHAT_COMPLETIONS_ENDPOINT)
    return Draft202012Validator(contract["stream" if stream else "response"])


@pytest.mark.parametrize("stream", [False, True], ids=["buffered", "stream"])
@pytest.mark.parametrize("items", [None, [], [REASONING], [REASONING, REASONING]])
def test_reviewed_reasoning_preserves_payload_and_assistant_text(stream, items):
    payload = payload_with_reasoning(stream, items)
    original = deepcopy(payload)
    parsed = model_for(stream).validate_payload(payload)
    if stream:
        assert parsed.choices[0].delta.content == "answer"
    else:
        assert parsed.locate_guarded_message(payload).message.content == "answer"
    validator_for(stream).validate(payload)
    assert payload == original


@pytest.mark.parametrize("stream", [False, True], ids=["buffered", "stream"])
@pytest.mark.parametrize("unknown", list(UnknownContentFieldPolicy))
@pytest.mark.parametrize(
    "items",
    [
        [{**REASONING, "summary": [{"type": "summary_text", "text": "unchecked"}]}],
        [{**REASONING, "content": [{"type": "reasoning_text", "text": "unchecked"}]}],
        [{**REASONING, "text": "unchecked"}],
        [{**REASONING, "future_content": "unchecked"}],
        [{**REASONING, "Summary": []}],
        [{**REASONING, "Encrypted_Content": "unchecked"}],
        [{**REASONING, "type": "message"}],
        [{"encrypted_content": "missing discriminator"}],
        ["unchecked"],
        "unchecked",
        {},
    ],
)
def test_plaintext_unknown_fields_and_malformed_items_remain_rejected(stream, unknown, items):
    payload = payload_with_reasoning(stream, items)
    with pytest.raises(ValidationError):
        model_for(stream).validate_payload(payload, unknown_content_fields=unknown)
    assert not validator_for(stream).is_valid(payload)


@pytest.mark.parametrize("stream", [False, True], ids=["buffered", "stream"])
def test_reasoning_permissions_do_not_open_extension_bags(stream):
    payload = payload_with_reasoning(stream, [])
    payload["choices"][0]["delta" if stream else "message"]["provider_specific_fields"] = {
        "reasoning_items": [REASONING]
    }
    with pytest.raises(ValidationError):
        model_for(stream).validate_payload(payload, unknown_content_fields=UnknownContentFieldPolicy.ALLOW)
    assert not validator_for(stream).is_valid(payload)
