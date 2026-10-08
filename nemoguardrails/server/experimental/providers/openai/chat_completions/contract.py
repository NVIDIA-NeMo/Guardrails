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

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from nemoguardrails.server.experimental.provider.payload import validate_payload_projection_contract
from nemoguardrails.server.experimental.provider.projection_policy import (
    CONTRACT_VERSION,
    EXTENSION,
    PolicyModel,
    export_payload_schema,
)
from nemoguardrails.server.experimental.providers.openai.chat_completions.endpoint import CHAT_COMPLETIONS_ENDPOINT

OPERATION_ID = "createChatCompletion"


def export_contract() -> dict[str, Any]:
    endpoint = CHAT_COMPLETIONS_ENDPOINT
    request_model = endpoint.guarded_request_model
    response_model = endpoint.guarded_response_model
    if not issubclass(request_model, PolicyModel) or not issubclass(response_model, PolicyModel):
        raise TypeError("Contract export requires policy-annotated endpoint models")
    request_contract = validate_payload_projection_contract(request_model, "request")
    response_contract = validate_payload_projection_contract(response_model, "response")
    request = export_payload_schema(request_model, projection_id=request_contract.projection_id)
    response = export_payload_schema(response_model, projection_id=response_contract.projection_id)
    request[EXTENSION]["stream_selector_field"] = endpoint.guarded_request_model.stream_selector_field
    return {
        "version": CONTRACT_VERSION,
        "operationId": OPERATION_ID,
        "profile": request_contract.profile.value,
        "request": request,
        "response": response,
        "integration": {
            "name": "chat_completions",
            "endpoint": {
                "route_path": endpoint.route_path,
                "operation_label": endpoint.operation,
                "unsupported_request_code": endpoint.unsupported_request_code,
                "unsupported_response_code": endpoint.unsupported_response_code,
            },
        },
    }


def main() -> None:
    import yaml
    from jsonschema import Draft202012Validator

    parser = argparse.ArgumentParser(
        description="Export the handwritten buffered Chat policy; no compiler is required."
    )
    destination = parser.add_mutually_exclusive_group()
    destination.add_argument("--output", type=Path)
    destination.add_argument("--check", type=Path)
    args = parser.parse_args()
    contract = export_contract()
    schema_path = Path(__file__).resolve().parents[3] / "contracts" / "guard-contract.schema.json"
    Draft202012Validator(json.loads(schema_path.read_text(encoding="utf-8"))).validate(contract)
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
