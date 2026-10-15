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

"""Verify explicit dialect policy with synthetic cases, not provider recordings."""

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
import yaml
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from nemoguardrails.server.experimental.provider.contract_export import export_guard_contract
from nemoguardrails.server.experimental.provider.projection_policy import EXTENSION
from nemoguardrails.server.experimental.provider.stream import UnsupportedProviderStream
from nemoguardrails.server.experimental.providers.openai.dialects.litellm.responses import (
    LITELLM_RESPONSES_ENDPOINT,
    STREAM_RULES,
    LiteLLMLifecycleResponse,
)
from nemoguardrails.server.experimental.providers.openai.responses.endpoint import RESPONSES_ENDPOINT
from nemoguardrails.server.experimental.providers.openai.responses.stream_classifier import STREAM_RULES as BASE_RULES
from tests.server.experimental.test_openai_responses_projections import response_payload
from tests.server.experimental.test_openai_responses_streaming_projections import event

METADATA = {
    "access_programs": None,
    "billing": {"payer": "developer"},
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
    "store": False,
    "tool_usage": {"web_search": {"num_requests": 0}},
}


def test_dialect_is_explicit_and_reuses_request_policy_and_lifecycle_hooks():
    endpoint = LITELLM_RESPONSES_ENDPOINT
    assert endpoint.guarded_request_model is RESPONSES_ENDPOINT.guarded_request_model
    assert endpoint.stream.hooks_factory is RESPONSES_ENDPOINT.stream.hooks_factory
    assert endpoint.operation_paths == RESPONSES_ENDPOINT.operation_paths
    assert len(STREAM_RULES) == len(BASE_RULES) == 15
    for base, dialect in zip(BASE_RULES, STREAM_RULES):
        assert replace(dialect, model=base.model) == base
        assert issubclass(dialect.model, base.model)


@pytest.mark.parametrize("name,value", METADATA.items())
def test_reviewed_metadata_is_response_only_and_preserves_the_original_payload(name, value):
    payload = {**response_payload(), name: value}
    original = deepcopy(payload)
    with pytest.raises(ValidationError):
        RESPONSES_ENDPOINT.guarded_response_model.validate_payload(payload)
    projection = LITELLM_RESPONSES_ENDPOINT.guarded_response_model.validate_payload(payload)
    assert projection.locate_guarded_message(payload).message.content == "answer"
    assert payload == original
    LiteLLMLifecycleResponse.model_validate(payload)
    exported = export_guard_contract(LITELLM_RESPONSES_ENDPOINT)
    assert exported["response"]["properties"][name][EXTENSION]["classification"] == "opaque"
    Draft202012Validator(exported["response"]).validate(payload)


@pytest.mark.parametrize(
    "updates",
    [{"guardrails": []}, {"fallbacks": []}, {"billing": {}}, {"input": []}, {"tools": []}, {"instructions": "hidden"}],
)
def test_dialect_does_not_enable_gateway_request_controls_or_extra_content(updates):
    with pytest.raises(ValidationError):
        LITELLM_RESPONSES_ENDPOINT.guarded_request_model.validate_payload({"input": "hello", **updates})


@pytest.mark.parametrize(
    "updates",
    [{"provider_specific_fields": {}}, {"future_text": "hidden"}, {"output_text": "hidden"}, {"Model": "alias"}],
)
def test_dialect_rejects_unreviewed_response_fields_and_duplicate_text(updates):
    with pytest.raises(ValidationError):
        LITELLM_RESPONSES_ENDPOINT.guarded_response_model.validate_payload(
            {**response_payload(), **METADATA, **updates}
        )


@pytest.mark.parametrize("shape", [rule.shape for rule in STREAM_RULES])
def test_every_rebound_event_retains_role_text_and_export_consistency(shape):
    base_rule = next(rule for rule in BASE_RULES if rule.shape == shape)
    native = shape.split(":", 1)[0]
    payload = response_payload()
    if shape.endswith(":reasoning"):
        fields = {"item": payload["output"][0]}
    elif shape.endswith(":message"):
        fields = {"item": payload["output"][1]}
    elif native.startswith("response.content_part."):
        fields = {"part": payload["output"][1]["content"][0]}
    elif native == "response.output_text.delta":
        fields = {"delta": "answer"}
    elif native == "response.output_text.done":
        fields = {"text": "answer"}
    elif native == "error":
        fields = {"message": "provider failure"}
    else:
        fields = {"response": payload}
    base = RESPONSES_ENDPOINT.stream.classifier.classify_event(event(native, **fields))
    if "response" in fields:
        fields["response"].update(METADATA)
    original = deepcopy(fields)
    wire = event(native, **fields, model="gateway-model")
    with pytest.raises(UnsupportedProviderStream):
        RESPONSES_ENDPOINT.stream.classifier.classify_event(wire)
    classified = LITELLM_RESPONSES_ENDPOINT.stream.classifier.classify_event(wire)
    assert classified.role == base.role == base_rule.role
    assert classified.text == base.text
    assert fields == original
    Draft202012Validator(export_guard_contract(LITELLM_RESPONSES_ENDPOINT)["stream"]).validate(json.loads(wire.data))


@pytest.mark.parametrize(
    "native,fields",
    [
        ("response.output_text.delta", {"delta": "answer", "provider_specific_fields": {}}),
        ("response.output_text.delta", {"delta": "answer", "future_text": "hidden"}),
        ("response.output_text.delta", {"delta": "answer", "Model": "alias"}),
        ("response.output_text.delta", {"delta": "answer", "logprobs": [{"token": "hidden"}]}),
        ("response.reasoning_summary_text.delta", {"delta": "hidden"}),
        ("response.output_item.done", {"item": {"type": "reasoning", "summary": [{"text": "hidden"}]}}),
        ("error", {"error": {"message": "nested diagnostic", "code": "provider_error"}}),
    ],
)
def test_model_annotation_does_not_relax_content_or_authorize_new_event_shapes(native, fields):
    with pytest.raises(UnsupportedProviderStream):
        LITELLM_RESPONSES_ENDPOINT.stream.classifier.classify_event(event(native, **fields, model="gateway-model"))


def test_bound_dialect_retains_identity_and_native_completion_checks():
    adapter = LITELLM_RESPONSES_ENDPOINT.stream.create_adapter()
    adapter.classify_event(event("response.output_text.delta", delta="answer", item_id="first", model="gateway-model"))
    with pytest.raises(UnsupportedProviderStream):
        adapter.classify_event(
            event("response.output_text.done", text="answer", item_id="different", model="gateway-model")
        )
    with pytest.raises(UnsupportedProviderStream):
        LITELLM_RESPONSES_ENDPOINT.stream.create_adapter().validate_end_of_stream()


def test_dialect_export_matches_generated_artifact_and_format():
    root = Path(__file__).parents[3] / "nemoguardrails/server/experimental/contracts"
    document = export_guard_contract(LITELLM_RESPONSES_ENDPOINT)
    assert document == yaml.safe_load((root / "openai/_generated/litellm-responses.guard.yaml").read_text())
    Draft202012Validator(json.loads((root / "guard-contract.schema.json").read_text())).validate(document)
