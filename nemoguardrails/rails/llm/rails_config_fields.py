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

from collections.abc import Collection
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple, Type

from pydantic import BaseModel, create_model

from nemoguardrails.manifests import RailManifest, all_rail_manifests, import_ref_target, resolve_import_ref
from nemoguardrails.manifests.config_schema import RailConfigSpec


@dataclass(frozen=True)
class RailsConfigField:
    rail_name: str
    config_key: str
    config: RailConfigSpec
    origin: str = ""


setattr(RailsConfigField, "__hash__", None)

_FIELDS: Dict[str, RailsConfigField] = {}
_EXPORTS: Dict[str, Tuple[str, Any]] = {}
_discovered = False
_discovering = False


def _add_config_field(
    fields: Dict[str, RailsConfigField], exports: Dict[str, Tuple[str, Any]], field: RailsConfigField
) -> None:
    existing = fields.get(field.rail_name)
    if existing is not None:
        if existing.origin == field.origin:
            return
        raise ValueError(
            f"Rail {field.rail_name!r} already contributed rails.config fields from {existing.origin!r}; "
            f"cannot contribute them from {field.origin!r}."
        )

    existing_config_rail = next(
        (rail_name for rail_name, existing_field in fields.items() if existing_field.config_key == field.config_key),
        None,
    )
    if existing_config_rail is not None and existing_config_rail != field.rail_name:
        raise ValueError(
            f"Rail config key {field.config_key!r} is already contributed by rail "
            f"{existing_config_rail!r}; cannot contribute it from {field.rail_name!r}."
        )

    for export_name, export_value in field.config.exports.items():
        existing_export = exports.get(export_name)
        if existing_export is not None and existing_export[0] != field.rail_name:
            raise ValueError(
                f"Rail config export {export_name!r} is already contributed by rail "
                f"{existing_export[0]!r}; cannot contribute it from {field.rail_name!r}."
            )
        exports[export_name] = (field.rail_name, export_value)

    fields[field.rail_name] = field


def _project_config_fields(
    manifests: Mapping[str, RailManifest], *, include_exports: bool = True
) -> Tuple[Dict[str, RailsConfigField], Dict[str, Tuple[str, Any]]]:
    fields: Dict[str, RailsConfigField] = {}
    exports: Dict[str, Tuple[str, Any]] = {}
    for manifest in manifests.values():
        if manifest.config_schema is None:
            continue
        build_config_spec = resolve_import_ref(manifest.config_schema.spec)
        config = build_config_spec()
        if not isinstance(config, RailConfigSpec):
            raise TypeError(f"Rail manifest {manifest.name!r} config schema factory must return RailConfigSpec.")
        if config.key is not None and config.key != manifest.config_schema.key:
            raise ValueError(
                f"Rail manifest {manifest.name!r} declares config key {manifest.config_schema.key!r}, "
                f"but its config spec returned {config.key!r}."
            )
        if manifest.config_schema.export_names and set(config.exports) != set(manifest.config_schema.export_names):
            raise ValueError(f"Rail manifest {manifest.name!r} export names do not match its config spec exports.")
        origin = import_ref_target(manifest.config_schema.spec).partition(":")[0]
        if not include_exports:
            config = RailConfigSpec(annotation=config.annotation, field_info=config.field_info, key=config.key)
        _add_config_field(
            fields,
            exports,
            RailsConfigField(
                rail_name=manifest.name,
                config_key=manifest.config_schema.key,
                config=config,
                origin=origin,
            ),
        )
    return fields, exports


def discover_rails_config_fields() -> None:
    """Project the rails.config field specs from the rail manifests once and cache them.

    Like discover_rail_manifests, the explicit two-flag guard must stay: functools.cache
    cannot express the in-progress re-entrancy check.
    """
    global _FIELDS, _EXPORTS, _discovered, _discovering

    if _discovered:
        return
    if _discovering:
        raise RuntimeError("rails.config field discovery re-entered while reading rail manifests.")

    _discovering = True
    try:
        fields, exports = _project_config_fields(all_rail_manifests())

        _FIELDS = fields
        _EXPORTS = exports
        _discovered = True
    finally:
        _discovering = False


def build_rails_config_data(
    base: Optional[Type[BaseModel]] = None,
    module: str = "nemoguardrails.rails.llm.config",
    manifests: Optional[Mapping[str, RailManifest]] = None,
    model_name: str = "RailsConfigData",
) -> Type[BaseModel]:
    """Build the RailsConfigData model from the projected config fields.

    config.py calls this at import time and bakes the result into a module-level class
    and __all__, so the field set cannot be varied by mutating discovery state at runtime.
    A test needing a different built-in field set must install substitute manifests, call
    the _reset_* seams, then importlib.reload the config module.
    """
    if manifests is None:
        discover_rails_config_fields()
        projected_fields = _FIELDS
    else:
        projected_fields, _ = _project_config_fields(manifests, include_exports=False)
    fields = {
        field.config_key: (field.config.annotation, field.config.field_info) for field in projected_fields.values()
    }
    kwargs: Dict[str, Any] = {
        "__module__": module,
        "__doc__": "Configuration data for specific rails that are supported out-of-the-box.",
        **fields,
    }
    if base is not None:
        kwargs["__base__"] = base
    return create_model(model_name, **kwargs)


def resolve_config_export(name: str) -> Any:
    discover_rails_config_fields()
    export = _EXPORTS.get(name)
    if export is None:
        raise KeyError(name)
    return export[1]


def config_exported_names() -> Tuple[str, ...]:
    discover_rails_config_fields()
    return tuple(_EXPORTS)


def validate_no_config_export_shadowing(existing_names: Collection[str]) -> None:
    discover_rails_config_fields()
    collisions = sorted(set(existing_names) & set(_EXPORTS))
    if not collisions:
        return

    details = []
    for export_name in collisions:
        rail_name, _export_value = _EXPORTS[export_name]
        field = _FIELDS.get(rail_name)
        origin = field.origin if field is not None else ""
        details.append(f"{export_name!r} from rail {rail_name!r} ({origin or 'unknown origin'})")

    raise ValueError(
        "Rail export name collision in nemoguardrails.rails.llm.config: "
        + "; ".join(details)
        + ". Rename the rail export or remove it from legacy config exports."
    )


def config_key_to_rail_name() -> Dict[str, str]:
    discover_rails_config_fields()
    return {field.config_key: field.rail_name for field in _FIELDS.values() if field.config_key != field.rail_name}


def all_config_fields() -> Dict[str, RailsConfigField]:
    discover_rails_config_fields()
    return dict(_FIELDS)


def _reset_rails_config_fields_cache() -> None:
    """Clear projected config fields so the next read reprojects from the manifests.

    Test seam only; production never calls this. The in-progress flag is reset too so a
    projection aborted by the re-entrancy guard cannot stay latched and poison later reads.
    """
    global _discovered, _discovering
    _FIELDS.clear()
    _EXPORTS.clear()
    _discovered = False
    _discovering = False
