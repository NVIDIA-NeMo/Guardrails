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

"""Audit the actual bound Chat declarations against an independent source pin."""

import hashlib
from copy import deepcopy
from typing import Literal, get_args

import pytest
import yaml
from jsonschema import Draft202012Validator

from nemoguardrails.server.experimental.provider.contract_export import export_guard_contract
from nemoguardrails.server.experimental.provider.projection_policy import field_coverage, model_graph
from nemoguardrails.server.experimental.providers.openai.chat_completions.endpoint import CHAT_COMPLETIONS_ENDPOINT
from tests.server.experimental.conformance.openai.chat_completions.fixtures.extract_source import (
    extract_chat_source,
    structural_fields,
)
from tests.server.experimental.conformance.openai.chat_completions.support import (
    DESCRIPTOR,
    FIXTURE,
    normalize_nullable,
    object_properties,
    resolve_pointer,
    source_document,
)

FIXTURE_SHA256 = "a84917786d64f8a04998da6f166b9b293eba20a1f093dd49ac69a1498f78454f"
REQUEST = CHAT_COMPLETIONS_ENDPOINT.guarded_request_model
USER_MESSAGE = get_args(REQUEST.model_fields["messages"].annotation)[0]
RESPONSE = CHAT_COMPLETIONS_ENDPOINT.guarded_response_model
CHOICE = get_args(RESPONSE.model_fields["choices"].annotation)[0]
ASSISTANT_MESSAGE = CHOICE.model_fields["message"].annotation
STREAM, ERROR = (rule.model for rule in CHAT_COMPLETIONS_ENDPOINT.stream.classifier.rules)
STREAM_CHOICE = get_args(STREAM.model_fields["choices"].annotation)[0]
DELTA = STREAM_CHOICE.model_fields["delta"].annotation
ERROR_DETAILS = ERROR.model_fields["error"].annotation

OBJECTS = [
    (REQUEST, "CreateChatCompletionRequest", frozenset()),
    (USER_MESSAGE, "ChatCompletionRequestUserMessage", frozenset()),
    (RESPONSE, "CreateChatCompletionResponse", frozenset()),
    (CHOICE, "CreateChatCompletionResponse/properties/choices/items", frozenset()),
    (ASSISTANT_MESSAGE, "ChatCompletionResponseMessage", frozenset({"reasoning_content"})),
    (STREAM, "CreateChatCompletionStreamResponse", frozenset()),
    (STREAM_CHOICE, "CreateChatCompletionStreamResponse/properties/choices/items", frozenset()),
    (DELTA, "ChatCompletionStreamResponseDelta", frozenset({"audio", "reasoning_content"})),
    (ERROR, "ErrorResponse", frozenset()),
    (ERROR_DETAILS, "Error", frozenset()),
]


def assert_field_review(model, schema, extensions):
    coverage = field_coverage(model)
    assert coverage.local_extension_fields == extensions
    assert coverage.reviewed_fields - extensions == object_properties(source_document(), schema).keys()


def test_source_fixture_provenance_and_operation_inventory():
    source = source_document()
    descriptor = yaml.safe_load(DESCRIPTOR.read_text(encoding="utf-8"))
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == FIXTURE_SHA256
    assert source["x-nemo-source"] == {"revision": descriptor["revision"], "sha256": descriptor["sha256"]}
    assert source["openapi"] == "3.1.0"
    assert {
        (path, method)
        for path, operations in source["paths"].items()
        for method in operations
        if method in {"get", "post", "delete"}
    } == {
        ("/chat/completions", "get"),
        ("/chat/completions", "post"),
        ("/chat/completions/{completion_id}", "get"),
        ("/chat/completions/{completion_id}", "post"),
        ("/chat/completions/{completion_id}", "delete"),
        ("/chat/completions/{completion_id}/messages", "get"),
    }
    operation = source["paths"]["/chat/completions"]["post"]
    assert operation["operationId"] == CHAT_COMPLETIONS_ENDPOINT.provider_operation_id
    assert set(operation["responses"]["200"]["content"]) == {"application/json", "text/event-stream"}


@pytest.mark.parametrize("model,component,extensions", OBJECTS, ids=[model.__name__ for model, _, _ in OBJECTS])
def test_every_inspected_source_object_has_explicit_field_review(model, component, extensions):
    schema = resolve_pointer(source_document(), "#/components/schemas/" + component)
    assert_field_review(model, schema, extensions)


def test_audit_covers_every_bound_policy_model():
    bound = {model for root in (REQUEST, RESPONSE, STREAM, ERROR) for model in model_graph(root).values()}
    assert bound == {model for model, _, _ in OBJECTS}


def test_new_source_field_requires_review():
    schema = deepcopy(resolve_pointer(source_document(), "#/components/schemas/ChatCompletionRequestUserMessage"))
    schema["properties"]["future_content"] = {"type": "string"}
    with pytest.raises(AssertionError):
        assert_field_review(USER_MESSAGE, schema, frozenset())


@pytest.mark.parametrize("direction,model", [("request", REQUEST), ("response", RESPONSE)])
def test_export_inventory_matches_the_bound_source_review(direction, model):
    exported = export_guard_contract(CHAT_COMPLETIONS_ENDPOINT)[direction]
    assert set(exported["properties"]) == field_coverage(model).reviewed_fields
    assert exported["additionalProperties"] is False


def test_source_message_variants_are_deliberately_narrowed():
    variants = source_document()["components"]["schemas"]["ChatCompletionRequestMessage"]["oneOf"]
    assert {item["$ref"].rsplit("/", 1)[-1] for item in variants} == {
        "ChatCompletionRequestDeveloperMessage",
        "ChatCompletionRequestSystemMessage",
        "ChatCompletionRequestUserMessage",
        "ChatCompletionRequestAssistantMessage",
        "ChatCompletionRequestToolMessage",
        "ChatCompletionRequestFunctionMessage",
    }
    assert USER_MESSAGE.model_fields["role"].annotation == Literal["user"]


def test_extraction_preserves_named_fields_and_literal_values():
    schema = {
        "description": "documentation",
        "properties": {
            "description": {"type": "string", "description": "documentation"},
            "title": {"type": "string"},
            "examples": {"type": "array"},
        },
        "default": {"description": "real data"},
        "const": {"title": "real data"},
    }
    assert structural_fields(schema) == {
        "properties": {"description": {"type": "string"}, "title": {"type": "string"}, "examples": {"type": "array"}},
        "default": {"description": "real data"},
        "const": {"title": "real data"},
    }
    assert structural_fields({"schemas": {"description": schema}}, ("components",))["schemas"].keys() == {"description"}


def test_extractor_rejects_unverified_source_bytes(tmp_path):
    source = tmp_path / "source.yaml"
    source.write_bytes(b"not the pinned document")
    with pytest.raises(ValueError, match="SHA-256"):
        extract_chat_source(source, DESCRIPTOR)


def test_extractor_keeps_only_chat_paths_and_transitive_components(tmp_path):
    source = {
        "openapi": "3.1.0",
        "info": {"title": "synthetic test", "version": "1", "license": {"name": "MIT"}},
        "paths": {
            "/chat/completions": {"post": {"schema": {"$ref": "#/components/schemas/Root"}}},
            "/chat/completions-other": {"get": {"schema": {"$ref": "#/components/schemas/Unused"}}},
        },
        "components": {
            "schemas": {
                "Root": {"properties": {"description": {"$ref": "#/components/schemas/Leaf"}}},
                "Leaf": {"type": "string"},
                "Unused": {"type": "string"},
            }
        },
    }
    source_path = tmp_path / "source.yaml"
    source_path.write_text(yaml.safe_dump(source))
    descriptor = tmp_path / "descriptor.yaml"
    descriptor.write_text(
        yaml.safe_dump({"revision": "synthetic", "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest()})
    )
    extracted = extract_chat_source(source_path, descriptor)
    assert set(extracted["paths"]) == {"/chat/completions"}
    assert set(extracted["components"]["schemas"]) == {"Root", "Leaf"}
    assert (
        extracted["components"]["schemas"]["Root"]["properties"]
        == source["components"]["schemas"]["Root"]["properties"]
    )


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "string", "enum": ["stop"], "nullable": True},
        {"$ref": "#/$defs/text", "nullable": True},
    ],
)
def test_source_nullable_adapter_handles_enum_and_ref_without_widening_other_values(schema):
    normalized = normalize_nullable(schema)
    validator = Draft202012Validator({**normalized, "$defs": {"text": {"type": "string"}}})
    assert validator.is_valid(None)
    assert validator.is_valid("stop")
    assert not validator.is_valid(123)
    assert schema["nullable"] is True


def test_source_nullable_adapter_does_not_rewrite_literal_data():
    schema = {"type": "object", "default": {"nullable": True}}
    assert normalize_nullable(schema) == schema


def test_reference_resolution_rejects_network_locations():
    with pytest.raises(ValueError, match="local JSON pointer"):
        resolve_pointer({}, "https://provider.test/schema.json")
