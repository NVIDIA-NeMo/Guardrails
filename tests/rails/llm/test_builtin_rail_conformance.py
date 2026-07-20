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

"""Cross-artifact conformance gates for built-in rail manifests."""

import inspect
import json
from pathlib import Path

from nemoguardrails.actions.rail_outcome import RailOutcome
from nemoguardrails.manifests import all_rail_manifests, resolve_import_ref
from nemoguardrails.rails.llm.config import RailsConfigData

CONFIG_SCHEMA_SNAPSHOT = Path(__file__).resolve().parents[3] / "schemas" / "rails_config.snapshot.json"
NON_PORTABLE_DECLARED_FLOWS = {
    "clavata": {"clavata check for"},
    "hallucination": {"hallucination warning"},
}


def test_projected_config_schema_matches_migration_snapshot():
    """Manifest projection preserves the complete pre-migration config schema."""
    expected = json.loads(CONFIG_SCHEMA_SNAPSHOT.read_text(encoding="utf-8"))

    assert RailsConfigData.model_json_schema() == expected


def test_projected_config_defaults_round_trip():
    """Every projected built-in config default survives JSON serialization."""
    config = RailsConfigData()

    assert RailsConfigData.model_validate_json(config.model_dump_json()) == config


def test_builtin_requirement_declarations_are_unique():
    """Requirement identifiers remain unambiguous within each rail manifest."""
    violations = []

    for rail_name, manifest in sorted(all_rail_manifests().items()):
        requirements = manifest.requirements
        groups = {
            "environment variables": [item.name for item in requirements.env_vars],
            "services": [item.name for item in requirements.services],
            "models": [item.type for item in requirements.models],
            "extras": list(requirements.extras),
            "optional dependencies": list(requirements.optional_dependencies),
        }
        for label, names in groups.items():
            duplicates = sorted({name for name in names if names.count(name) > 1})
            if duplicates:
                violations.append(f"{rail_name}: duplicate {label}: {duplicates}")

    assert not violations, "\n".join(violations)


def test_builtin_remote_services_are_declared_requirements():
    """Privacy metadata names only services declared by the same manifest."""
    violations = []

    for rail_name, manifest in sorted(all_rail_manifests().items()):
        declared = {service.name for service in manifest.requirements.services}
        undeclared = sorted(set(manifest.privacy.remote_services) - declared)
        if undeclared:
            violations.append(f"{rail_name}: undeclared remote services: {undeclared}")

    assert not violations, "\n".join(violations)


def test_declared_public_flows_are_portable_or_explicitly_exempted():
    """Every declared public flow has an execution surface or a named exception."""
    violations = []

    for rail_name, manifest in sorted(all_rail_manifests().items()):
        declared = set(manifest.flows.flow_names if manifest.flows is not None else ())
        portable = {surface.name for surface in manifest.surfaces}
        exempted = NON_PORTABLE_DECLARED_FLOWS.get(rail_name, set())
        if declared != portable | exempted:
            violations.append(
                f"{rail_name}: missing surfaces {sorted(declared - portable - exempted)}, "
                f"undeclared surfaces {sorted(portable - declared)}, "
                f"stale exemptions {sorted(exempted - declared)}"
            )

    assert not violations, "\n".join(violations)


def test_portable_surface_actions_declare_rail_outcome_returns():
    """Every portable surface action advertises the backend-neutral verdict contract."""
    violations = []

    for rail_name, manifest in sorted(all_rail_manifests().items()):
        for surface in manifest.surfaces:
            action = resolve_import_ref(surface.action)
            if inspect.signature(action).return_annotation is not RailOutcome:
                violations.append(f"{rail_name}: {surface.name!r} does not return RailOutcome")

    assert not violations, "\n".join(violations)
