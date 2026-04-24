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

"""Generate the SMS-format JSON schema from GuardrailsUsageEvent.

Usage:
    poetry run python scripts/generate_telemetry_schema.py

    # Override output path:
    poetry run python scripts/generate_telemetry_schema.py path/to/schema.json

    # Write to stdout instead:
    poetry run python scripts/generate_telemetry_schema.py -

Produces the schema document that gets uploaded to SMS. Auto-generates all
property definitions from the GuardrailsUsageEvent Pydantic class, so the
schema stays in sync with the code.
"""

import json
import sys
from pathlib import Path
from typing import Any

from nemoguardrails.telemetry import GuardrailsUsageEvent, TelemetryEvent

DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parent.parent / "schemas" / "telemetry.json"

CLIENT_NAME = "NeMoGuardrails"
CLIENT_ID = "<assigned by SMS>"
DEFINITION_VERSION = "1.0"

SCHEMA_DESCRIPTION = "Anonymous telemetry schema for NeMo Guardrails."

EVENT_DESCRIPTION = (
    "Instance-level usage census event emitted at startup and as heartbeats "
    "by NeMo Guardrails. Contains no user content, model names, or request-level data."
)

GDPR_DESCRIPTION = (
    "Technical metadata describing the guardrails deployment "
    "(version, platform, active features). No user content or request-level data."
)


def _rewrite_refs(obj: Any) -> Any:
    """Rewrite Pydantic 2020-12 ``$defs`` refs to draft-07 ``definitions.types`` refs.

    Pydantic emits ``$ref: "#/$defs/EnumName"`` for enum types, but SMS
    expects draft-07 format with types under ``definitions.types``.

    Args:
        obj: Any JSON-serializable value (dict, list, primitive).

    Returns:
        The value with all ``$ref`` strings updated in-place-copy.
    """
    if isinstance(obj, dict):
        return {
            k: (
                v.replace("#/$defs/", "#/definitions/types/")
                if k == "$ref" and isinstance(v, str)
                else _rewrite_refs(v)
            )
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_rewrite_refs(item) for item in obj]
    return obj


def build_event_definition(event_cls: type[TelemetryEvent]) -> tuple[dict, dict]:
    """Build the event definition and extract any shared type definitions.

    Args:
        event_cls: The telemetry event class to introspect.

    Returns:
        A tuple of ``(event_definition, type_definitions)``. The second
        element contains enum type definitions extracted from Pydantic's
        ``$defs`` block, ready to place under ``definitions.types``.
    """
    schema = event_cls.model_json_schema(by_alias=True)
    properties = schema.get("properties", {})
    defs = schema.get("$defs", {})

    cleaned_properties = {}
    for name, prop in properties.items():
        cleaned = {k: v for k, v in prop.items() if k != "title"}
        cleaned_properties[name] = _rewrite_refs(cleaned)

    cleaned_properties["nemoSource"] = {
        "type": "string",
        "description": 'The NeMo product that created the event. Always "guardrails".',
        "enum": ["guardrails"],
    }

    required = list(cleaned_properties.keys())

    type_definitions = {
        name: _rewrite_refs({k: v for k, v in definition.items() if k != "title"}) for name, definition in defs.items()
    }

    event_def = {
        "description": EVENT_DESCRIPTION,
        "eventMeta": {
            "service": "telemetry",
            "gdpr": {
                "category": "functional",
                "desc": GDPR_DESCRIPTION,
            },
        },
        "type": "object",
        "properties": cleaned_properties,
        "additionalProperties": False,
        "required": required,
    }

    return event_def, type_definitions


def build_sms_schema() -> dict:
    """Build the complete SMS-format JSON schema document.

    Returns:
        A dict ready to be JSON-serialized and uploaded to SMS.
    """
    event_def, type_defs = build_event_definition(GuardrailsUsageEvent)

    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "description": SCHEMA_DESCRIPTION,
        "schemaMeta": {
            "clientName": CLIENT_NAME,
            "schemaVersion": GuardrailsUsageEvent._schema_version,
            "definitionVersion": DEFINITION_VERSION,
            "clientId": CLIENT_ID,
            "personalization": "",
        },
        "definitions": {
            "types": type_defs,
            "events": {
                GuardrailsUsageEvent._event_name: event_def,
            },
        },
        "oneOf": [
            {"$ref": f"#/definitions/events/{GuardrailsUsageEvent._event_name}"},
        ],
    }


if __name__ == "__main__":
    schema = build_sms_schema()
    payload = json.dumps(schema, indent=2) + "\n"

    if len(sys.argv) > 1:
        target = sys.argv[1]
    else:
        target = str(DEFAULT_OUTPUT_PATH)

    if target == "-":
        sys.stdout.write(payload)
    else:
        out_path = Path(target)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload, encoding="utf-8")
        print(f"wrote {out_path}", file=sys.stderr)
