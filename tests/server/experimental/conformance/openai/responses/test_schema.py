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

import hashlib
import json
from copy import deepcopy

import pytest
import yaml

from nemoguardrails.server.experimental.provider.contract_export import export_guard_contract
from nemoguardrails.server.experimental.provider.projection_policy import EXTENSION, field_coverage, model_graph
from tests.server.experimental.conformance.openai.responses.fixtures.extract_source import COMPONENTS, extract_inventory
from tests.server.experimental.conformance.openai.responses.support import ENDPOINTS, INVENTORY, INVENTORY_PATH, ROOT

NATIVE_ADDITIONS = {"store", "access_programs", "billing", "frequency_penalty", "presence_penalty", "tool_usage"}


def bound_models(endpoint):
    roots = [endpoint.guarded_request_model, endpoint.guarded_response_model]
    roots.extend(rule.model for rule in endpoint.stream.classifier.rules)
    return {model for root in roots for model in model_graph(root).values()}


def source_objects(node):
    if isinstance(node, dict):
        source = node.get(EXTENSION, {}).get("source")
        if source and "properties" in node:
            yield source.removeprefix("#/components/schemas/"), node
        for value in node.values():
            yield from source_objects(value)
    elif isinstance(node, list):
        for value in node:
            yield from source_objects(value)


def test_inventory_tracks_the_reviewed_provider_source():
    pin = yaml.safe_load((ROOT / "nemoguardrails/server/experimental/contracts/openai/source.yaml").read_text())
    assert INVENTORY["source"] == pin
    assert (
        hashlib.sha256(INVENTORY_PATH.read_bytes()).hexdigest()
        == "3166e95194393c2180bca713a440368b4760610fa422692765134401affcbb19"
    )


@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_every_bound_model_has_complete_independent_source_coverage(endpoint):
    models = bound_models(endpoint)
    assert {model.policy.source for model in models} == INVENTORY["schema_fields"].keys()
    event_models = {rule.model for rule in endpoint.stream.classifier.rules}
    for model in models:
        expected = set(INVENTORY["schema_fields"][model.policy.source])
        if model.policy.source == "Response":
            expected |= NATIVE_ADDITIONS
        if model.policy.source == "ResponseTextDeltaEvent":
            expected.add("obfuscation")
        if endpoint.contract_name == "litellm_responses" and model in event_models:
            expected.add("model")
        assert field_coverage(model).reviewed_fields == expected, model.__name__


@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_official_response_example_catches_fields_omitted_from_schema(endpoint):
    payload = deepcopy(INVENTORY["text_response_example"])
    assert "store" not in INVENTORY["schema_fields"]["Response"]
    assert payload["store"] is True
    parsed = endpoint.guarded_response_model.validate_payload(payload)
    assert parsed.locate_guarded_message(payload).message.content == payload["output"][0]["content"][0]["text"]
    assert payload == INVENTORY["text_response_example"]


@pytest.mark.parametrize(
    "field", ["store", "access_programs", "billing", "frequency_penalty", "presence_penalty", "tool_usage"]
)
@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_observed_native_metadata_is_opaque_in_buffered_and_lifecycle_responses(endpoint, field):
    models = [model for model in bound_models(endpoint) if model.policy.source == "Response"]
    assert len(models) == 2
    for model in models:
        assert field in field_coverage(model).opaque_fields


@pytest.mark.parametrize(
    "source,field,classification",
    [
        ("CreateResponse", "input", "guarded"),
        ("CreateResponse", "instructions", "constrained"),
        ("CreateResponse", "prompt", "constrained"),
        ("CreateResponse", "previous_response_id", "constrained"),
        ("CreateResponse", "conversation", "constrained"),
        ("CreateResponse", "tools", "constrained"),
        ("CreateResponse", "include", "constrained"),
        ("Reasoning", "summary", "constrained"),
        ("Response", "output", "guarded"),
        ("Response", "output_text", "constrained"),
        ("OutputMessage", "content", "guarded"),
        ("OutputTextContent", "text", "guarded"),
        ("OutputTextContent", "annotations", "constrained"),
        ("OutputTextContent", "logprobs", "constrained"),
        ("ReasoningItem", "summary", "constrained"),
        ("ReasoningItem", "content", "constrained"),
        ("ReasoningItem", "encrypted_content", "opaque"),
        ("ResponseTextDeltaEvent", "delta", "guarded"),
        ("ResponseTextDeltaEvent", "obfuscation", "opaque"),
        ("ResponseTextDoneEvent", "text", "guarded"),
        ("ResponseCompletedEvent", "response", "guarded"),
        ("ResponseIncompleteEvent", "response", "guarded"),
        ("ResponseFailedEvent", "response", "guarded"),
    ],
)
@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_reviewed_content_paths_cannot_be_reclassified_as_metadata(endpoint, source, field, classification):
    models = [model for model in bound_models(endpoint) if model.policy.source == source]
    assert models
    for model in models:
        assert field in getattr(field_coverage(model), classification + "_fields"), model.__name__
    nodes = [node for name, node in source_objects(export_guard_contract(endpoint)) if name == source]
    assert nodes
    for node in nodes:
        assert node["properties"][field][EXTENSION]["classification"] == classification


def test_extractor_rejects_unverified_source_bytes():
    with pytest.raises(ValueError, match="Source bytes"):
        extract_inventory(b"unverified", INVENTORY["source"])


def extraction_document():
    schemas = {name: {"properties": {"field": {}}} for name in COMPONENTS}
    schemas["Response"]["allOf"] = [{"$ref": "#/components/schemas/Inherited"}]
    schemas["Inherited"] = {"properties": {"store": {"type": "boolean"}}}
    return {
        "components": {"schemas": schemas},
        "paths": {
            "/responses": {
                "post": {
                    "x-oaiMeta": {
                        "examples": [{"title": "Text input", "response": json.dumps({"store": True, "output": []})}]
                    }
                }
            }
        },
    }


def test_extractor_preserves_inherited_fields_and_separate_example_evidence():
    raw = yaml.safe_dump(extraction_document()).encode()
    result = extract_inventory(raw, {"sha256": hashlib.sha256(raw).hexdigest()})
    assert result["schema_fields"]["Response"] == ["field", "store"]
    assert result["text_response_example"] == {"store": True, "output": []}


def test_extractor_rejects_nonlocal_schema_references():
    document = extraction_document()
    document["components"]["schemas"]["Response"]["allOf"] = [{"$ref": "https://provider.test/schema.json"}]
    raw = yaml.safe_dump(document).encode()
    with pytest.raises(ValueError, match="Only local"):
        extract_inventory(raw, {"sha256": hashlib.sha256(raw).hexdigest()})
