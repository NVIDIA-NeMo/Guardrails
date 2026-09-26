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


@pytest.fixture(scope="module")
def guard_contract_schema() -> dict:
    """Load the guard contract schema shared by the validation tests."""
    return json.loads(SCHEMA_PATH.read_text())


@pytest.fixture(scope="module")
def minimal_guard_contract() -> dict:
    """Load the minimal guard contract example."""
    return yaml.safe_load(MINIMAL_CONTRACT_PATH.read_text())


def test_guard_contract_schema_is_valid(guard_contract_schema: dict) -> None:
    """The guard contract schema is a valid JSON Schema 2020-12 document."""
    Draft202012Validator.check_schema(guard_contract_schema)


def test_minimal_guard_contract_matches_schema(guard_contract_schema: dict, minimal_guard_contract: dict) -> None:
    """The minimal guard contract demonstrates a schema-valid document."""
    Draft202012Validator(guard_contract_schema).validate(minimal_guard_contract)


def test_guard_contract_rejects_unknown_version(guard_contract_schema: dict, minimal_guard_contract: dict) -> None:
    """The schema rejects contracts authored for an unsupported version."""
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


def test_contract_documentation_has_no_broken_local_links() -> None:
    """Relative links in the contract documentation resolve locally."""
    broken_links: list[str] = []

    for document in CONTRACTS_ROOT.glob("*.md"):
        for link in MARKDOWN_LINK.findall(document.read_text()):
            if "://" in link or link.startswith("#"):
                continue
            target, _, _fragment = link.partition("#")
            if target and not (document.parent / target).exists():
                broken_links.append(f"{document.name}: {link}")

    assert broken_links == []
