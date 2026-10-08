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

"""Export buffered guard contracts from trusted Python endpoint declarations.

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
import re
import sys
from pathlib import Path
from typing import Any

from nemoguardrails.server.experimental.provider.endpoint import GuardedJsonEndpoint
from nemoguardrails.server.experimental.provider.payload import validate_payload_projection_contract
from nemoguardrails.server.experimental.provider.projection_policy import (
    CONTRACT_VERSION,
    EXTENSION,
    PolicyModel,
    export_payload_schema,
)


def export_guard_contract(
    endpoint: GuardedJsonEndpoint, *, operation_id: str, name: str | None = None
) -> dict[str, Any]:
    """Describe the buffered policy of the models actually bound to an endpoint.

    Args:
        endpoint: A trusted endpoint with policy-annotated request and response
            models and their runtime bindings.
        operation_id: Stable provider operation identity for the document.
        name: Optional integration identifier, using lowercase snake case.

    Returns:
        A fresh document with payload schemas, profile, and endpoint labels.
        The optional request stream selector is descriptive; there is no stream
        event section. Header/query revision bindings, alternate route ownership,
        and arbitrary Python behavior are not serialized.

    Raises:
        TypeError: The endpoint or its models do not support policy export.
        ValueError: Identity or projection metadata is invalid, or a payload
            schema uses a construct unsupported by the shared schema exporter.

    No files are read or written. The CLI separately validates the document
    format; neither step proves upstream compatibility or runtime equivalence.
    """
    if not isinstance(endpoint, GuardedJsonEndpoint):
        raise TypeError("Contract export requires a GuardedJsonEndpoint")
    if not operation_id.strip():
        raise ValueError("Contract operation_id must not be blank")
    if name is not None and re.fullmatch(r"[a-z][a-z0-9_]*", name) is None:
        raise ValueError("Contract name must use lowercase snake case")
    request_model = endpoint.guarded_request_model
    response_model = endpoint.guarded_response_model
    if not issubclass(request_model, PolicyModel) or not issubclass(response_model, PolicyModel):
        raise TypeError("Contract export requires policy-annotated endpoint models")
    request_contract = validate_payload_projection_contract(request_model, "request")
    response_contract = validate_payload_projection_contract(response_model, "response")
    request = export_payload_schema(request_model, projection_id=request_contract.projection_id)
    response = export_payload_schema(response_model, projection_id=response_contract.projection_id)
    if request_model.stream_selector_field is not None:
        request[EXTENSION]["stream_selector_field"] = request_model.stream_selector_field
    integration: dict[str, Any] = {}
    if name is not None:
        integration["name"] = name
    integration["endpoint"] = {
        "route_path": endpoint.route_path,
        "operation_label": endpoint.operation,
        "unsupported_request_code": endpoint.unsupported_request_code,
        "unsupported_response_code": endpoint.unsupported_response_code,
    }
    return {
        "version": CONTRACT_VERSION,
        "operationId": operation_id,
        "profile": request_contract.profile.value,
        "request": request,
        "response": response,
        "integration": integration,
    }


def _load_endpoint(reference: str) -> GuardedJsonEndpoint:
    """Import one trusted module attribute for the CLI, without discovery or eval."""
    module, separator, attribute = reference.partition(":")
    if not separator or not all(part.isidentifier() for part in module.split(".")) or not attribute.isidentifier():
        raise argparse.ArgumentTypeError("Endpoint must be a Python module:attribute reference")
    try:
        endpoint = getattr(importlib.import_module(module), attribute)
    except (ImportError, AttributeError) as error:
        raise argparse.ArgumentTypeError(f"Cannot load endpoint {reference}: {error}") from error
    if not isinstance(endpoint, GuardedJsonEndpoint):
        raise argparse.ArgumentTypeError(f"{reference} is not a GuardedJsonEndpoint")
    return endpoint


def main() -> None:
    """Validate and emit a buffered contract, or check an artifact for exact drift.

    The endpoint argument imports trusted Python code. With --output, create
    parent directories and overwrite the selected file. With --check, leave files
    untouched and exit with status 1 on missing or differing content. With neither
    option, write YAML to stdout. Invalid declarations fail before file writes.
    """
    parser = argparse.ArgumentParser(description="Export a buffered guard contract from a trusted Python endpoint.")
    parser.add_argument("endpoint", type=_load_endpoint, help="trusted Python module:attribute (imports code)")
    parser.add_argument("--operation-id", required=True, help="provider operation identity")
    parser.add_argument("--name", help="optional integration identifier")
    destination = parser.add_mutually_exclusive_group()
    destination.add_argument("--output", type=Path)
    destination.add_argument("--check", type=Path)
    args = parser.parse_args()

    import yaml
    from jsonschema import Draft202012Validator
    from jsonschema.exceptions import ValidationError

    schema_path = Path(__file__).resolve().parents[1] / "contracts" / "guard-contract.schema.json"
    try:
        contract = export_guard_contract(args.endpoint, operation_id=args.operation_id, name=args.name)
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
