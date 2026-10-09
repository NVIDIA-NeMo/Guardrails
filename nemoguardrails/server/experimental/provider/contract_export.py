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

"""Export guard contracts from trusted Python endpoint declarations.

Provider integrations supply an endpoint and document identity, not exporter
implementations. Model annotations describe payload policy; endpoint metadata
describes the route and rejection codes. Exporting does not enable runtime
features or serialize arbitrary Python validation and transport behavior.

The CLI imports one explicitly selected module attribute. That import executes
Python code: use only trusted local endpoint declarations, never references from
requests or untrusted documents. YAML and JSON Schema tooling is loaded only by
the CLI; runtime endpoint construction does not depend on this module.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Literal, get_args, get_origin

from nemoguardrails.server.experimental.provider.endpoint import GuardedJsonEndpoint
from nemoguardrails.server.experimental.provider.payload import (
    GuardedPayloadModel,
    GuardedRequestModel,
    PayloadProjectionContract,
    validate_payload_projection_contract,
)
from nemoguardrails.server.experimental.provider.projection_policy import (
    CONTRACT_VERSION,
    EXTENSION,
    PolicyModel,
    export_payload_schema,
    payload_contract,
    text_location,
)
from nemoguardrails.server.experimental.provider.stream import StreamBinding, StreamCapabilityProfile, StreamEventRole
from nemoguardrails.server.experimental.provider.stream_classifier import (
    StreamClassifier,
    StreamClassifierDefinition,
    StreamEventRule,
)
from nemoguardrails.server.experimental.provider.stream_policy import build_policy_stream_classifier


def _export_stream_event(rule: StreamEventRule) -> dict[str, Any]:
    """Describe a typed event rule without dropping unsupported selection or fallback semantics."""
    if type(rule) is not StreamEventRule:
        raise TypeError("Stream contract export requires standard declarative event rules")
    if not issubclass(rule.model, PolicyModel) or not rule.model.policy.source:
        raise ValueError("Stream event export requires an explicit policy source schema")
    if len(dict(rule.match)) != len(rule.match) or any(type(value) not in (str, int, bool) for _, value in rule.match):
        raise ValueError("Stream event export requires unique match keys with string, integer, or boolean values")
    roles = {
        StreamEventRole.GUARDED_TEXT: "guarded_delta",
        StreamEventRole.TEXT_SNAPSHOT: "snapshot",
        StreamEventRole.OPAQUE_METADATA: "opaque",
        StreamEventRole.PROVIDER_ERROR: "provider_error",
    }
    variant: dict[str, Any] = {"source_schema": rule.model.policy.source, "shape": rule.shape}
    if rule.match:
        variant["match"] = dict(rule.match)
    if rule.required_fields:
        variant["required_fields"] = sorted(rule.required_fields)
    event: dict[str, Any] = {"classification": roles[rule.role], "variants": [variant]}
    if rule.role in {StreamEventRole.GUARDED_TEXT, StreamEventRole.TEXT_SNAPSHOT}:
        if rule.missing_text_role is None:
            event["missing_text"] = "reject"
        elif rule.missing_text_role is StreamEventRole.OPAQUE_METADATA and rule.missing_text_value is None:
            event["missing_text"] = "opaque"
        elif rule.missing_text_role is rule.role and rule.missing_text_value == "":
            event["missing_text"] = "empty"
        else:
            raise ValueError("Stream missing-text fallback cannot be represented by the contract")
        if rule.missing_text_shape is not None:
            event["missing_text_shape"] = rule.missing_text_shape
    schema = export_payload_schema(rule.model, projection_id=rule.model.__name__)
    schema[EXTENSION]["event"] = event
    return schema


def _export_stream(binding: StreamBinding) -> dict[str, Any]:
    """Export supported classifier declarations, never instantiate or serialize protocol hooks."""
    classifier = binding.classifier
    if type(classifier) is not StreamClassifier:
        raise TypeError("Stream contract export requires the standard declarative StreamClassifier")
    if classifier.contract.profile is not StreamCapabilityProfile.SINGLE_TEXT_DELTA_V1:
        raise ValueError("Stream capability profile cannot be represented by the contract")
    build_policy_stream_classifier(
        StreamClassifierDefinition(
            subject=classifier.subject,
            contract=classifier.contract,
            rules=classifier.rules,
            sentinels=classifier.sentinels,
            non_data_shape=classifier.non_data_shape,
            event_type_field=classifier.event_type_field,
            require_event_type=classifier.require_event_type,
        )
    )
    transport: dict[str, Any] = {"require_sse_event": classifier.require_event_type}
    if classifier.event_type_field is not None:
        transport["data_discriminator"] = classifier.event_type_field
    if classifier.non_data_shape is not None:
        transport["non_data_shape"] = classifier.non_data_shape
    if classifier.sentinels:
        try:
            transport["sentinels"] = {value.decode("utf-8"): shape for value, shape in classifier.sentinels}
        except UnicodeDecodeError as error:
            raise ValueError("Stream contract sentinels must be UTF-8 text") from error
    return {
        "title": classifier.contract.projection_id,
        "oneOf": [_export_stream_event(rule) for rule in classifier.rules],
        EXTENSION: {"transport": transport},
    }


def _validate_policy_binding(
    model: type[GuardedPayloadModel], direction: Literal["request", "response"]
) -> PayloadProjectionContract:
    """Reject drift between supported runtime bindings and exported field policy."""
    if not issubclass(model, PolicyModel):
        raise TypeError("Contract export requires policy-annotated endpoint models")
    contract = validate_payload_projection_contract(model, direction)
    expected = payload_contract(
        model, projection_id=contract.projection_id, direction=direction, profile=contract.profile
    )
    if contract != expected:
        raise ValueError(f"{model.__name__}: payload coverage differs from field policy")
    if model.locate_guarded_message is not GuardedPayloadModel.locate_guarded_message:
        raise ValueError("Custom text extraction cannot be represented by the buffered contract")
    if model.guarded_text_location != text_location(model):
        raise ValueError(f"{model.__name__}: text binding differs from field policy")
    return contract


def _validate_stream_selector(model: type[GuardedRequestModel]) -> None:
    """Require a declared selector to use the standard boolean-field binding."""
    selector = model.stream_selector_field
    if selector is None:
        return
    field = model.model_fields.get(selector)
    annotation = field.annotation if field is not None else None
    literal_boolean = get_origin(annotation) is Literal and all(type(value) is bool for value in get_args(annotation))
    if annotation is not bool and not literal_boolean:
        raise ValueError("The stream selector must name a boolean field")
    if model.streams_response is not GuardedRequestModel.streams_response:
        raise ValueError("Custom response-mode selection cannot be represented by the declared stream selector")


def export_guard_contract(endpoint: GuardedJsonEndpoint) -> dict[str, Any]:
    """Describe the payload and stream policy actually bound to an endpoint.

    Args:
        endpoint: A trusted endpoint with policy-annotated request and response
            models and their runtime bindings.

    Returns:
        A fresh document with payload schemas, profile, endpoint labels, and
        optional stream event rules and transport metadata. Stateful hooks,
        header/query revision bindings, alternate route ownership, and arbitrary
        Python behavior are not serialized.

    Raises:
        TypeError: The endpoint or its models do not support policy export.
        ValueError: Bound coverage, extraction, replacement, or stream selection
            disagrees with the field policy, or a schema construct is unsupported.

    No files are read or written. The CLI separately validates the document
    format; neither step proves upstream compatibility or runtime equivalence.
    """
    if not isinstance(endpoint, GuardedJsonEndpoint):
        raise TypeError("Contract export requires a GuardedJsonEndpoint")
    request_model = endpoint.guarded_request_model
    response_model = endpoint.guarded_response_model
    if not issubclass(request_model, PolicyModel) or not issubclass(response_model, PolicyModel):
        raise TypeError("Contract export requires policy-annotated endpoint models")
    request_contract = _validate_policy_binding(request_model, "request")
    response_contract = _validate_policy_binding(response_model, "response")
    if request_contract.profile != response_contract.profile:
        raise ValueError("Guarded request and response capability profiles must match")
    _validate_stream_selector(request_model)
    request = export_payload_schema(request_model, projection_id=request_contract.projection_id)
    response = export_payload_schema(response_model, projection_id=response_contract.projection_id)
    if request_model.stream_selector_field is not None:
        request[EXTENSION]["stream_selector_field"] = request_model.stream_selector_field
    integration: dict[str, Any] = {}
    if endpoint.contract_name is not None:
        integration["name"] = endpoint.contract_name
    integration["endpoint"] = {
        "route_path": endpoint.route_path,
        "operation_label": endpoint.operation,
        "unsupported_request_code": endpoint.unsupported_request_code,
        "unsupported_response_code": endpoint.unsupported_response_code,
    }
    document = {
        "version": CONTRACT_VERSION,
        "operationId": endpoint.provider_operation_id,
        "profile": request_contract.profile.value,
        "request": request,
        "response": response,
        "integration": integration,
    }
    if endpoint.stream is not None:
        document["stream"] = _export_stream(endpoint.stream)
    return document


def _load_endpoint(reference: str) -> GuardedJsonEndpoint:
    """Import one trusted module attribute for the CLI, without discovery or eval."""
    module, separator, attribute = reference.partition(":")
    if not separator or not all(part.isidentifier() for part in module.split(".")) or not attribute.isidentifier():
        raise argparse.ArgumentTypeError("Endpoint must be a Python module:attribute reference")
    try:
        endpoint = getattr(importlib.import_module(module), attribute)
    except (ImportError, AttributeError, ValueError) as error:
        raise argparse.ArgumentTypeError(f"Cannot load endpoint {reference}: {error}") from error
    if not isinstance(endpoint, GuardedJsonEndpoint):
        raise argparse.ArgumentTypeError(f"{reference} is not a GuardedJsonEndpoint")
    return endpoint


def main() -> None:
    """Validate and emit a guard contract, or check an artifact for exact drift.

    The endpoint argument imports trusted Python code. With --output, create
    parent directories and overwrite the selected file. With --check, leave files
    untouched and exit with status 1 on missing or differing content. With neither
    option, write YAML to stdout. Invalid declarations fail before file writes.
    """
    parser = argparse.ArgumentParser(description="Export a guard contract from a trusted Python endpoint.")
    parser.add_argument("endpoint", type=_load_endpoint, help="trusted Python module:attribute (imports code)")
    destination = parser.add_mutually_exclusive_group()
    destination.add_argument("--output", type=Path)
    destination.add_argument("--check", type=Path)
    args = parser.parse_args()

    import yaml
    from jsonschema import Draft202012Validator
    from jsonschema.exceptions import ValidationError

    schema_path = Path(__file__).resolve().parents[1] / "contracts" / "guard-contract.schema.json"
    try:
        contract = export_guard_contract(args.endpoint)
        Draft202012Validator(json.loads(schema_path.read_text(encoding="utf-8"))).validate(contract)
    except (TypeError, ValueError, ValidationError) as error:
        parser.error(str(error))
    rendered = yaml.safe_dump(contract, sort_keys=False, allow_unicode=True)
    if args.check:
        if not args.check.is_file() or args.check.read_text(encoding="utf-8") != rendered:
            parser.exit(1, f"Export differs from {args.check}; regenerate it with --output.\n")
    elif args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)


if __name__ == "__main__":
    main()
