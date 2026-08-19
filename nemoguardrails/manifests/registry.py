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

import json
import threading
from importlib import resources
from typing import Any, Dict, cast

from pydantic import ValidationError

from nemoguardrails.manifests.catalog import RailCatalog, RailManifestRecord
from nemoguardrails.manifests.manifest import RailManifest

_catalog: RailCatalog | None = None
_discovering = False
_lock = threading.RLock()
_BUILTIN_CATALOG_RESOURCE = "builtin_rails.json"
_BUILTIN_CATALOG_FORMAT_VERSION = 1


def _load_builtin_catalog() -> RailCatalog:
    resource = resources.files("nemoguardrails.manifests").joinpath(_BUILTIN_CATALOG_RESOURCE)
    try:
        content = resource.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise RuntimeError(f"Built-in rail catalog artifact {_BUILTIN_CATALOG_RESOURCE!r} is missing.") from exc
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Built-in rail catalog artifact {_BUILTIN_CATALOG_RESOURCE!r} is malformed.") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Built-in rail catalog artifact {_BUILTIN_CATALOG_RESOURCE!r} must contain an object.")
    version = payload.get("format_version")
    if version != _BUILTIN_CATALOG_FORMAT_VERSION:
        raise RuntimeError(
            f"Built-in rail catalog artifact {_BUILTIN_CATALOG_RESOURCE!r} has unsupported format version "
            f"{version!r}; expected {_BUILTIN_CATALOG_FORMAT_VERSION}."
        )
    records_payload = payload.get("records")
    if not isinstance(records_payload, list):
        raise RuntimeError(f"Built-in rail catalog artifact {_BUILTIN_CATALOG_RESOURCE!r} must contain a records list.")
    records = []
    for index, record_payload in enumerate(records_payload):
        if not isinstance(record_payload, dict):
            raise RuntimeError(
                f"Built-in rail catalog artifact {_BUILTIN_CATALOG_RESOURCE!r} record {index} must be an object."
            )
        record_data = cast(Dict[str, Any], record_payload)
        if set(record_data) != {"manifest", "source"} or not isinstance(record_data["source"], str):
            raise RuntimeError(
                f"Built-in rail catalog artifact {_BUILTIN_CATALOG_RESOURCE!r} record {index} is malformed."
            )
        try:
            manifest = RailManifest.model_validate(record_data["manifest"])
        except ValidationError as exc:
            raise RuntimeError(
                f"Built-in rail catalog artifact {_BUILTIN_CATALOG_RESOURCE!r} record {index} has an invalid manifest."
            ) from exc
        source = cast(str, record_data["source"])
        manifest = manifest.model_copy(update={"origin": source})
        records.append(RailManifestRecord(manifest=manifest, source=source))
    return RailCatalog(records)


def default_rail_catalog() -> RailCatalog:
    """Return the cached catalog of built-in rail manifests."""
    global _catalog, _discovering
    if _catalog is not None:
        return _catalog
    with _lock:
        if _catalog is not None:
            return _catalog
        if _discovering:
            raise RuntimeError("Built-in rail catalog loading re-entered.")
        _discovering = True
        try:
            catalog = _load_builtin_catalog()
            _catalog = catalog
            return catalog
        finally:
            _discovering = False


def all_rail_manifests():
    """Return built-in rail manifests keyed by manifest name."""
    return dict(default_rail_catalog().manifests)


def _reset_rail_manifest_cache() -> None:
    global _catalog, _discovering
    with _lock:
        _catalog = None
        _discovering = False
