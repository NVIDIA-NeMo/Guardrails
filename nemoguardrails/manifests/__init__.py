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

"""Public API for the rail manifest contract, catalog, and config schema.

Re-exports the manifest types (`RailManifest`, `RailMetadata`,
`RailSpec`, and friends) and the `RailCatalog`, and provides
process-wide accessors for the default catalog of built-in rails.
"""

from nemoguardrails.manifests.catalog import RailCatalog, RailManifestRecord
from nemoguardrails.manifests.manifest import (
    ActionRef,
    Binding,
    ConfigSpecRef,
    EnvVar,
    ExampleRef,
    ImportRef,
    ModelRequirement,
    RailActions,
    RailCapability,
    RailCategory,
    RailConfigSchema,
    RailDirection,
    RailFlows,
    RailLifecycle,
    RailManifest,
    RailMetadata,
    RailPrivacy,
    RailRequirements,
    RailSpec,
    RailStatus,
    RailSurface,
    ServiceRequirement,
    TransformTarget,
    import_ref_target,
    iter_manifest_import_refs,
    iter_manifest_import_targets,
    normalize_configured_surface_name,
    parse_configured_surface,
    resolve_import_ref,
)
from nemoguardrails.manifests.manifest import (
    configured_rail_surfaces as _configured_rail_surfaces,
)

_catalog: RailCatalog | None = None
_discovering = False


def default_rail_catalog() -> RailCatalog:
    global _catalog, _discovering
    if _catalog is not None:
        return _catalog
    if _discovering:
        raise RuntimeError("Built-in rail manifest discovery re-entered while loading rail modules.")
    _discovering = True
    try:
        catalog = RailCatalog.discover_built_ins()
        _catalog = catalog
        return catalog
    finally:
        _discovering = False


def all_rail_manifests():
    return dict(default_rail_catalog().manifests)


def rail_surfaces(direction: RailDirection | str | None = None):
    parsed_direction = RailDirection(direction) if direction is not None else None
    return default_rail_catalog().surfaces(parsed_direction)


def surface_names(direction: RailDirection | str):
    parsed_direction = RailDirection(direction)
    return tuple(sorted(name for _direction, name in rail_surfaces(parsed_direction)))


def configured_rail_surfaces(direction: RailDirection | str, flows):
    parsed_direction = RailDirection(direction)
    return _configured_rail_surfaces(parsed_direction, flows, rail_surfaces(parsed_direction))


selected_rail_surfaces = configured_rail_surfaces


def _reset_rail_manifest_cache() -> None:
    global _catalog, _discovering
    _catalog = None
    _discovering = False


__all__ = [
    "ActionRef",
    "Binding",
    "ConfigSpecRef",
    "EnvVar",
    "ExampleRef",
    "ImportRef",
    "ModelRequirement",
    "RailActions",
    "RailCapability",
    "RailCatalog",
    "RailCategory",
    "RailConfigSchema",
    "RailDirection",
    "RailFlows",
    "RailLifecycle",
    "RailManifest",
    "RailManifestRecord",
    "RailMetadata",
    "RailPrivacy",
    "RailRequirements",
    "RailSpec",
    "RailStatus",
    "RailSurface",
    "ServiceRequirement",
    "TransformTarget",
    "all_rail_manifests",
    "configured_rail_surfaces",
    "default_rail_catalog",
    "import_ref_target",
    "iter_manifest_import_refs",
    "iter_manifest_import_targets",
    "normalize_configured_surface_name",
    "parse_configured_surface",
    "rail_surfaces",
    "resolve_import_ref",
    "selected_rail_surfaces",
    "surface_names",
]
