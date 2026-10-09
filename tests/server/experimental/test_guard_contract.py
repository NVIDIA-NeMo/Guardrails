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

import json
import re
from copy import deepcopy
from pathlib import Path

import pytest
import yaml
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

REPOSITORY_ROOT = Path(__file__).parents[3]
CONTRACTS_ROOT = REPOSITORY_ROOT / "nemoguardrails/server/experimental/contracts"
SCHEMA_PATH = CONTRACTS_ROOT / "guard-contract.schema.json"
MINIMAL_CONTRACT_PATH = CONTRACTS_ROOT / "minimal.guard.example.yaml"
MARKDOWN_LINK = re.compile(r"\[[^]]+\]\(([^)]+)\)")
EXTENSION = "x-nemo-guardrails"


@pytest.fixture(scope="module")
def guard_contract_schema() -> dict:
    """Load the guard contract schema shared by the validation tests."""
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def minimal_guard_contract() -> dict:
    """Load the minimal guard contract example."""
    return yaml.safe_load(MINIMAL_CONTRACT_PATH.read_text(encoding="utf-8"))


@pytest.fixture
def streaming_guard_contract(minimal_guard_contract: dict) -> dict:
    """Describe a text event and SSE sentinel without an endpoint implementation."""
    contract = deepcopy(minimal_guard_contract)
    event = deepcopy(contract["response"])
    event[EXTENSION] = {
        "event": {
            "classification": "guarded_delta",
            "variants": [
                {
                    "source_schema": "ExampleTextEvent",
                    "shape": "text.delta",
                    "required_fields": ["text"],
                }
            ],
            "missing_text": "reject",
        }
    }
    contract["stream"] = {
        "oneOf": [event],
        EXTENSION: {
            "transport": {
                "require_sse_event": False,
                "non_data_shape": "[DONE]",
                "sentinels": {"[DONE]": "[DONE]"},
            }
        },
    }
    return contract


def test_guard_contract_schema_is_valid(guard_contract_schema: dict) -> None:
    """The guard contract schema is a valid JSON Schema 2020-12 document."""
    Draft202012Validator.check_schema(guard_contract_schema)


def test_minimal_guard_contract_matches_schema(guard_contract_schema: dict, minimal_guard_contract: dict) -> None:
    """The minimal guard contract demonstrates a schema-valid document."""
    Draft202012Validator(guard_contract_schema).validate(minimal_guard_contract)


@pytest.mark.parametrize("projection", ["request", "response"])
@pytest.mark.parametrize("payload", [None, True, 42, "text", []])
def test_minimal_guard_projection_rejects_non_objects(
    minimal_guard_contract: dict, projection: str, payload: object
) -> None:
    """Both example projections require object payloads."""
    with pytest.raises(ValidationError):
        Draft202012Validator(minimal_guard_contract[projection]).validate(payload)


@pytest.mark.parametrize(("projection", "field"), [("request", "prompt"), ("response", "text")])
def test_minimal_guard_projection_requires_its_text_field(
    minimal_guard_contract: dict, projection: str, field: str
) -> None:
    """Object payloads pass only when the required text field is present."""
    validator = Draft202012Validator(minimal_guard_contract[projection])
    validator.validate({field: "hello"})

    with pytest.raises(ValidationError, match="required property"):
        validator.validate({})


def test_guard_contract_rejects_unknown_version(guard_contract_schema: dict, minimal_guard_contract: dict) -> None:
    """The schema rejects documents with an unsupported format version."""
    contract = minimal_guard_contract | {"version": "1.0.0"}

    with pytest.raises(ValidationError):
        Draft202012Validator(guard_contract_schema).validate(contract)


def test_guard_contract_rejects_unknown_top_level_key(
    guard_contract_schema: dict, minimal_guard_contract: dict
) -> None:
    """The schema rejects misspelled or unsupported top-level keys."""
    contract = minimal_guard_contract | {"requests": {}}

    with pytest.raises(ValidationError):
        Draft202012Validator(guard_contract_schema).validate(contract)


def test_guard_contract_allows_non_guardrails_schema_extensions(
    guard_contract_schema: dict, minimal_guard_contract: dict
) -> None:
    """Non-Guardrails schema extensions remain available to other tooling."""
    contract = deepcopy(minimal_guard_contract)
    contract["request"]["x-provider-annotation"] = {"value": "example"}

    Draft202012Validator(guard_contract_schema).validate(contract)


@pytest.mark.parametrize("key", ["x-nemo-guardrail", "x-nemo-guardrails-extra", "x-nemoguardrails"])
@pytest.mark.parametrize("on_field", [False, True])
def test_guard_contract_rejects_guardrails_extension_lookalikes(
    guard_contract_schema: dict, minimal_guard_contract: dict, key: str, on_field: bool
) -> None:
    """The reserved x-nemo namespace rejects misspelled Guardrails keys on objects and fields."""
    contract = deepcopy(minimal_guard_contract)
    target = contract["request"]["properties"]["prompt"] if on_field else contract["request"]
    target[key] = {"classification": "opaque"}

    with pytest.raises(ValidationError):
        Draft202012Validator(guard_contract_schema).validate(contract)


def test_stream_contract_matches_schema(guard_contract_schema: dict, streaming_guard_contract: dict) -> None:
    """The format supports a stream section independently of runtime support."""
    Draft202012Validator(guard_contract_schema).validate(streaming_guard_contract)


@pytest.mark.parametrize("section", ["transport", "event"])
def test_stream_contract_requires_framing_and_event_policy(
    guard_contract_schema: dict, streaming_guard_contract: dict, section: str
) -> None:
    """Stream roots need transport metadata and each family needs event policy."""
    stream = streaming_guard_contract["stream"]
    metadata = stream[EXTENSION] if section == "transport" else stream["oneOf"][0][EXTENSION]
    del metadata[section]

    with pytest.raises(ValidationError):
        Draft202012Validator(guard_contract_schema).validate(streaming_guard_contract)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("event_schema_binding", "operation_response"),
        ("event_schema_reason", "provider_integrity.component_source"),
    ],
)
def test_stream_contract_rejects_source_resolution_settings(
    guard_contract_schema: dict, streaming_guard_contract: dict, key: str, value: str
) -> None:
    """Source resolution settings are not part of the exported policy vocabulary."""
    streaming_guard_contract["stream"][EXTENSION][key] = value

    with pytest.raises(ValidationError):
        Draft202012Validator(guard_contract_schema).validate(streaming_guard_contract)


def test_contract_documentation_has_no_broken_local_links() -> None:
    """Relative links in the contract documentation resolve locally."""
    broken_links: list[str] = []

    for document in CONTRACTS_ROOT.rglob("*.md"):
        for link in MARKDOWN_LINK.findall(document.read_text(encoding="utf-8")):
            if "://" in link or link.startswith("#"):
                continue
            target, _, _fragment = link.partition("#")
            if target and not (document.parent / target).exists():
                broken_links.append(f"{document.name}: {link}")

    assert broken_links == []


@pytest.mark.parametrize("unknown_fields", ["forbid", "configurable"])
def test_contract_accepts_explicit_object_member_policy(guard_contract_schema, minimal_guard_contract, unknown_fields):
    contract = deepcopy(minimal_guard_contract)
    contract["request"][EXTENSION].update(unknown_fields=unknown_fields, reject_case_aliases=True)
    Draft202012Validator(guard_contract_schema).validate(contract)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("unknown_fields", "allow"),
        ("unknown_fields", True),
        ("reject_case_aliases", False),
        ("reject_case_aliases", "true"),
    ],
)
def test_contract_rejects_invalid_object_member_policy(guard_contract_schema, minimal_guard_contract, key, value):
    contract = deepcopy(minimal_guard_contract)
    contract["request"][EXTENSION][key] = value
    with pytest.raises(ValidationError):
        Draft202012Validator(guard_contract_schema).validate(contract)
