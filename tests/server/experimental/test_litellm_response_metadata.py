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

"""Source-backed bridge metadata must preserve the guarded text boundary."""

from copy import deepcopy

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from nemoguardrails.server.experimental.provider.contract_export import export_guard_contract
from nemoguardrails.server.experimental.providers.openai.chat_completions.endpoint import CHAT_COMPLETIONS_ENDPOINT
from nemoguardrails.server.experimental.providers.openai.dialects.litellm.chat_completions import (
    LITELLM_CHAT_COMPLETIONS_ENDPOINT,
)
from nemoguardrails.server.experimental.providers.openai.dialects.litellm.stream import LiteLLMStreamPayloadProjection
from tests.server.experimental.conformance.openai.chat_completions.support import chat_chunk, chat_response

METADATA = {
    "access_programs": {"cyber": "daybreak_blue"},
    "billing": {"payer": "developer"},
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
    "tool_usage": {"web_search": {"num_requests": 0}},
}


@pytest.mark.parametrize("field,value", METADATA.items())
def test_bridge_metadata_is_explicit_and_preserves_buffered_payload(field, value):
    payload = {**chat_response(), field: value}
    original = deepcopy(payload)
    with pytest.raises(ValidationError):
        CHAT_COMPLETIONS_ENDPOINT.guarded_response_model.validate_payload(payload)
    parsed = LITELLM_CHAT_COMPLETIONS_ENDPOINT.guarded_response_model.validate_payload(payload)
    assert parsed.locate_guarded_message(payload).message.content == payload["choices"][0]["message"]["content"]
    Draft202012Validator(export_guard_contract(LITELLM_CHAT_COMPLETIONS_ENDPOINT)["response"]).validate(payload)
    assert payload == original


@pytest.mark.parametrize("location", ["root", "root-bag"])
@pytest.mark.parametrize("field,value", {**METADATA, "service_tier": "default"}.items())
def test_stream_metadata_preserves_text_and_export(location, field, value):
    payload = chat_chunk("answer")
    target = payload if location == "root" else payload.setdefault("provider_specific_fields", {})
    target[field] = value
    original = deepcopy(payload)
    parsed = LiteLLMStreamPayloadProjection.validate_payload(payload)
    assert parsed.choices[0].delta.content == "answer"
    Draft202012Validator(export_guard_contract(LITELLM_CHAT_COMPLETIONS_ENDPOINT)["stream"]).validate(payload)
    assert payload == original


@pytest.mark.parametrize("field,value", METADATA.items())
def test_accounting_permissions_do_not_extend_to_delta_bag(field, value):
    payload = chat_chunk("answer")
    payload["choices"][0]["delta"]["provider_specific_fields"] = {field: value}
    with pytest.raises(ValidationError):
        LiteLLMStreamPayloadProjection.validate_payload(payload)


@pytest.mark.parametrize("location", ["root", "root-bag"])
def test_bridge_metadata_does_not_open_unknown_text(location):
    payload = chat_chunk("answer")
    target = payload if location == "root" else payload.setdefault("provider_specific_fields", {})
    target.update(METADATA)
    target["future_content"] = "unchecked"
    with pytest.raises(ValidationError):
        LiteLLMStreamPayloadProjection.validate_payload(payload)
