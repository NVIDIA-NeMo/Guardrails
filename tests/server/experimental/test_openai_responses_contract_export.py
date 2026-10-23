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

"""Verify that Responses documentation describes the actual typed bindings."""

import json
from pathlib import Path

import pytest
import yaml
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from nemoguardrails.server.experimental.provider.contract_export import export_guard_contract
from nemoguardrails.server.experimental.provider.projection_policy import EXTENSION
from nemoguardrails.server.experimental.provider.types import UnknownContentFieldPolicy
from nemoguardrails.server.experimental.providers.openai.responses.endpoint import RESPONSES_ENDPOINT
from tests.server.experimental.test_openai_responses_projections import output_message, response_payload

CONTRACTS = Path(__file__).parents[3] / "nemoguardrails/server/experimental/contracts"


def test_responses_export_matches_checked_in_artifact_and_format():
    exported = export_guard_contract(RESPONSES_ENDPOINT)
    assert exported == yaml.safe_load((CONTRACTS / "openai/_generated/responses.guard.yaml").read_text())
    Draft202012Validator(json.loads((CONTRACTS / "guard-contract.schema.json").read_text())).validate(exported)
    assert exported["operationId"] == "createResponse"
    assert exported["integration"]["endpoint"]["route_path"] == "/v1/responses"


@pytest.mark.parametrize(
    "output",
    [
        response_payload()["output"],
        list(reversed(response_payload()["output"])),
        [],
        [{"type": "reasoning"}],
        [output_message(), output_message()],
        [output_message(), {"type": "function_call", "arguments": "hidden"}],
    ],
)
def test_exported_variant_cardinality_matches_bound_validation(output, export_policy_validator):
    model = RESPONSES_ENDPOINT.guarded_response_model
    document = {"output": output}
    validator = export_policy_validator(
        export_guard_contract(RESPONSES_ENDPOINT)["response"], UnknownContentFieldPolicy.FORBID
    )
    if validator.is_valid(document):
        model.validate_payload(document)
    else:
        with pytest.raises(ValidationError):
            model.validate_payload(document)


def test_export_marks_reasoning_as_unselected_and_assistant_as_exactly_one():
    variants = export_guard_contract(RESPONSES_ENDPOINT)["response"]["properties"]["output"]["items"]["oneOf"]
    assert [branch[EXTENSION]["variant"]["cardinality"] for branch in variants] == ["any", "exactly_one"]
    assert variants[1][EXTENSION]["variant"]["match"] == {"type": "message", "role": "assistant"}


def test_stream_export_preserves_all_bound_event_selections_and_missing_text_rules():
    classifier = RESPONSES_ENDPOINT.stream.classifier
    exported = export_guard_contract(RESPONSES_ENDPOINT)
    events = exported["stream"]["oneOf"]
    assert len(events) == len(classifier.rules)
    for node, rule in zip(events, classifier.rules):
        declaration = node[EXTENSION]["event"]
        assert declaration["variants"] == [
            {"source_schema": rule.model.policy.source, "shape": rule.shape, "match": dict(rule.match)}
        ]
        if rule.missing_text_value == "":
            assert declaration["missing_text"] == "empty"
    assert exported["stream"][EXTENSION]["transport"] == {
        "require_sse_event": False,
        "data_discriminator": "type",
        "non_data_shape": "keepalive",
        "sentinels": {"[DONE]": "done"},
    }
    assert "hooks" not in exported["integration"]


@pytest.mark.parametrize(
    "storage_fields", [{}, {"store": False}, {"store": True}, {"store": None}, {"store": 0}, {"store": "false"}]
)
def test_exported_storage_constraint_matches_runtime(storage_fields, export_policy_validator):
    document = {"input": "question", **storage_fields}
    validator = export_policy_validator(
        export_guard_contract(RESPONSES_ENDPOINT)["request"], UnknownContentFieldPolicy.FORBID
    )
    accepted = storage_fields == {"store": False} and type(storage_fields.get("store")) is bool
    assert validator.is_valid(document) is accepted
    if accepted:
        RESPONSES_ENDPOINT.guarded_request_model.validate_payload(document)
    else:
        with pytest.raises(ValidationError):
            RESPONSES_ENDPOINT.guarded_request_model.validate_payload(document)
