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

"""Audit archived Chat evidence separately from source-backed shape fixtures.

These tests never record or manufacture provider responses. They make the limits
of the existing evidence executable: buffered extensions contain null refusal
and an empty choice object; streams do not exercise extension bags. Runtime and
export validation must accept every successful recorded payload unchanged.
"""

import json
from copy import deepcopy
from pathlib import Path

import pytest
import yaml
from jsonschema import Draft202012Validator

from nemoguardrails.server.experimental.provider.contract_export import export_guard_contract
from nemoguardrails.server.experimental.provider.sse import ServerSentEvent
from nemoguardrails.server.experimental.provider.stream import StreamEventRole
from nemoguardrails.server.experimental.providers.openai.dialects.litellm.chat_completions import (
    LITELLM_CHAT_COMPLETIONS_ENDPOINT,
)

pytestmark = [pytest.mark.recorded]

CASSETTES = Path(__file__).parent / "cassettes/test_chat_completions"
SUCCESS_RECORDINGS = (
    "test_buffered_chat_preserves_guarded_constrained_and_opaque_contract_fields",
    "test_nvidia_chat_source_profile_preserves_allowed_relay[buffered]",
    "test_streaming_chat_preserves_declared_events_and_opaque_metadata",
    "test_nvidia_chat_source_profile_preserves_allowed_relay[streaming]",
)


def extension_values(value):
    """Find extension bags anywhere, including outside the inspected text path."""
    if isinstance(value, dict):
        for name, nested in value.items():
            if name == "provider_specific_fields":
                yield nested
            yield from extension_values(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from extension_values(nested)


@pytest.mark.parametrize("recording", SUCCESS_RECORDINGS)
def test_recorded_inventory_matches_bound_policy_and_export(recording):
    (interaction,) = yaml.safe_load((CASSETTES / f"{recording}.yaml").read_text())["interactions"]
    request = interaction["request"]["parsed_body"]
    response = interaction["response"]
    body = response["body"]["parsed_body"]
    original = deepcopy(body)
    endpoint = LITELLM_CHAT_COMPLETIONS_ENDPOINT
    document = export_guard_contract(endpoint)

    assert response["status"]["code"] == 200
    assert not {"caching", "num_retries", "guardrails", "fallbacks", "context_window_fallback_dict"} & request.keys()
    endpoint.guarded_request_model.validate_payload(request)
    Draft202012Validator(document["request"]).validate(request)

    if request["stream"]:
        assert list(extension_values(body)) == []
        assert endpoint.stream is not None
        adapter = endpoint.stream.create_adapter()
        validator = Draft202012Validator(document["stream"])
        text_events = 0
        for payload in body:
            if payload == "[DONE]":
                data = b"[DONE]"
            else:
                validator.validate(payload)
                data = json.dumps(payload).encode()
            event = ServerSentEvent.from_bytes(b"data: " + data + b"\n\n")
            classified = adapter.classify_event(event)
            if classified.role is StreamEventRole.GUARDED_TEXT:
                assert classified.text == payload["choices"][0]["delta"]["content"]
                text_events += 1
        adapter.validate_end_of_stream()
        assert text_events > 0
    else:
        assert len(list(extension_values(body))) == 2
        (choice,) = body["choices"]
        assert choice["message"]["provider_specific_fields"] == {"refusal": None}
        assert choice["provider_specific_fields"] == {}
        assert type(body["usage"]["completion_tokens_details"]["reasoning_tokens"]) is int
        endpoint.guarded_response_model.validate_payload(body)
        Draft202012Validator(document["response"]).validate(body)
    assert body == original
