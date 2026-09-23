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

"""Catalog construction and discovery for rail manifests and their surfaces.

Collects `RailManifestRecord` entries for built-in rails discovered under
`nemoguardrails/library` into an immutable `RailCatalog` that enforces global
uniqueness of rail names, config keys, flow names, action names, and surface
keys.
"""

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Tuple, TypeVar

from nemoguardrails.manifests.manifest import ActionRef, RailDirection, RailManifest, RailSurface


@dataclass(frozen=True, slots=True)
class RailManifestRecord:
    """Manifest plus provenance recorded in a rail catalog."""

    manifest: RailManifest
    source: str


_OwnerKey = TypeVar("_OwnerKey")


def _claim(owners: Dict[_OwnerKey, str], key: _OwnerKey, manifest_name: str, label: str) -> None:
    owner = owners.get(key)
    if owner is not None:
        raise ValueError(f"{label} is already provided by {owner!r}; cannot also provide it from {manifest_name!r}.")
    owners[key] = manifest_name


class RailCatalog:
    """Immutable index of rail manifests keyed by name and surface.

    Constructing a catalog validates the combined set of records and raises
    `ValueError` on any collision: duplicate manifest names, duplicate config
    keys, flow names, action names, or two rails claiming the same `(direction,
    surface name)`. It also rejects a surface whose action is not declared in
    that surface's own manifest.
    """

    def __init__(self, records: Iterable[RailManifestRecord] = ()) -> None:
        records_by_name: Dict[str, RailManifestRecord] = {}
        surfaces: Dict[Tuple[RailDirection, str], RailSurface] = {}
        surface_owners: Dict[Tuple[RailDirection, str], str] = {}
        config_owners: Dict[str, str] = {}
        flow_owners: Dict[str, str] = {}
        action_owners: Dict[str, Tuple[str, ActionRef]] = {}
        for record in records:
            manifest = record.manifest
            declared_actions = set(manifest.actions.refs if manifest.actions is not None else ())
            existing = records_by_name.get(manifest.name)
            if existing is not None:
                raise ValueError(
                    f"Rail manifest {manifest.name!r} is already provided by {existing.source!r}; "
                    f"cannot also provide it from {record.source!r}."
                )
            if manifest.config_schema is not None:
                key = manifest.config_schema.key
                _claim(config_owners, key, manifest.name, f"Rail config key {key!r}")
            if manifest.flows is not None:
                for flow_name in manifest.flows.flow_names:
                    _claim(flow_owners, flow_name, manifest.name, f"Rail flow {flow_name!r}")
            if manifest.actions is not None:
                for action_ref in manifest.actions.refs:
                    existing_action = action_owners.get(action_ref.name)
                    if existing_action is not None:
                        raise ValueError(
                            f"Rail action {action_ref.name!r} is already provided by {existing_action[0]!r}; "
                            f"cannot also provide it from {manifest.name!r}."
                        )
                    action_owners[action_ref.name] = (manifest.name, action_ref)
            for surface in manifest.surfaces:
                if surface.action not in declared_actions:
                    raise ValueError(
                        f"Rail surface {surface.name!r} from {manifest.name!r} references action "
                        f"{surface.action.name!r}, which is not declared in that manifest."
                    )
                key = (surface.direction, surface.name)
                owner = surface_owners.get(key)
                if owner is not None:
                    if owner == manifest.name:
                        raise ValueError(
                            f"Rail manifest {manifest.name!r} declares duplicate surface {surface.name!r} "
                            f"for direction {surface.direction.value!r}."
                        )
                _claim(
                    surface_owners,
                    key,
                    manifest.name,
                    f"Rail surface {surface.name!r} for direction {surface.direction.value!r}",
                )
                surfaces[key] = surface
            records_by_name[manifest.name] = record
        self._records = records_by_name
        self._surfaces = surfaces
        self._flow_owners = flow_owners

    @classmethod
    def discover_built_ins(cls, library_path: Optional[Path] = None) -> "RailCatalog":
        """Discover built-in manifests from `rail.py` modules under the library."""
        if library_path is None:
            library_path = Path(__file__).resolve().parents[1] / "library"
        records = []
        for manifest_file in sorted(library_path.rglob("rail.py")):
            relative_module = manifest_file.relative_to(library_path).with_suffix("")
            module_name = ".".join(("nemoguardrails", "library", *relative_module.parts))
            module = importlib.import_module(module_name)
            manifest = getattr(module, "RAIL", None)
            if not isinstance(manifest, RailManifest):
                raise TypeError(f"Rail manifest module {module_name!r} must define RAIL as a RailManifest.")
            manifest = manifest.model_copy(update={"origin": module_name})
            records.append(RailManifestRecord(manifest=manifest, source=module_name))
        return cls(records)

    @property
    def records(self) -> Mapping[str, RailManifestRecord]:
        """Return catalog records keyed by manifest name."""
        return dict(self._records)

    @property
    def manifests(self) -> Mapping[str, RailManifest]:
        """Return rail manifests keyed by manifest name."""
        return {name: record.manifest for name, record in self._records.items()}

    def surfaces(self, direction: Optional[RailDirection] = None) -> Dict[Tuple[RailDirection, str], RailSurface]:
        """Return declared surfaces, optionally filtered by direction."""
        return {key: surface for key, surface in self._surfaces.items() if direction is None or key[0] == direction}

    def owner_for_flow(self, flow_name: str) -> Optional[str]:
        """Return the manifest that owns a declared public flow name."""
        return self._flow_owners.get(flow_name)
