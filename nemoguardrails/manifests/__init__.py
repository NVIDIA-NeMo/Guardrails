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

"""Public API for the rail manifest contract and catalog.

Re-exports the manifest types (`RailManifest`, `RailMetadata`,
`RailSpec`, and friends) and the `RailCatalog`, and provides
process-wide accessors for the default catalog of built-in rails
(`default_rail_catalog` and `rail_catalog`).
"""

from nemoguardrails.manifests.catalog import RailCatalog, RailManifestRecord
from nemoguardrails.manifests.manifest import (
    ActionRef,
    Binding,
    ConfigSpecRef,
    EnvVar,
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
    RailSurface,
    ServiceRequirement,
    TransformTarget,
    import_ref_target,
    iter_manifest_import_refs,
    normalize_configured_surface_name,
    parse_configured_surface,
    resolve_import_ref,
)

_catalog: RailCatalog | None = None
_discovering = False
_plugin_catalogs = {}


def default_rail_catalog() -> RailCatalog:
    """Return the cached catalog of built-in rail manifests."""
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
    """Return built-in rail manifests keyed by manifest name."""
    return dict(default_rail_catalog().manifests)


def rail_catalog(enabled_plugins=()):
    """Return the built-in catalog extended with the named rail plugins."""
    names = tuple(sorted(set(enabled_plugins)))
    if not names:
        return default_rail_catalog()
    catalog = _plugin_catalogs.get(names)
    if catalog is None:
        catalog = default_rail_catalog().with_plugins(names)
        _plugin_catalogs[names] = catalog
    return catalog


def _reset_rail_manifest_cache() -> None:
    global _catalog, _discovering
    _catalog = None
    _discovering = False
    _plugin_catalogs.clear()


__all__ = [
    "ActionRef",
    "Binding",
    "ConfigSpecRef",
    "EnvVar",
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
    "RailSurface",
    "ServiceRequirement",
    "TransformTarget",
    "all_rail_manifests",
    "default_rail_catalog",
    "import_ref_target",
    "iter_manifest_import_refs",
    "normalize_configured_surface_name",
    "parse_configured_surface",
    "rail_catalog",
    "resolve_import_ref",
]
