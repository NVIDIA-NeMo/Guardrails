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

"""Process-wide access to discovered rail manifest catalogs."""

from nemoguardrails.manifests.catalog import RailCatalog

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
