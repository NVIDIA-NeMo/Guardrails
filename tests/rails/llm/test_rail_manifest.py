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

import pytest
from pydantic import ValidationError

from nemoguardrails.manifests import (
    ActionRef,
    Binding,
    ConfigSpecRef,
    RailActions,
    RailCatalog,
    RailConfigSchema,
    RailDirection,
    RailManifest,
    RailManifestRecord,
    RailMetadata,
    RailSpec,
    RailSurface,
    import_ref_target,
    iter_manifest_import_targets,
    resolve_import_ref,
)


def _action(name: str = "check") -> ActionRef:
    return ActionRef(name=name, target="pathlib:Path.cwd")


def _record(name: str, *, action: ActionRef | None = None, surface_name: str | None = None) -> RailManifestRecord:
    action = action or _action(f"{name}_check")
    surfaces = ()
    if surface_name is not None:
        surfaces = (
            RailSurface(
                name=surface_name,
                direction=RailDirection.INPUT,
                action=action,
                bindings=(Binding.context("text", "user_message"),),
            ),
        )
    manifest = RailManifest(
        name=name,
        spec=RailSpec(actions=RailActions(refs=(action,)), surfaces=surfaces),
    )
    return RailManifestRecord(manifest=manifest, source=f"test:{name}")


def test_manifest_round_trips_with_typed_refs():
    action = _action()
    manifest = RailManifest(
        name="test",
        spec=RailSpec(
            config_schema=RailConfigSchema(key="test", spec=ConfigSpecRef(target="pathlib:Path.cwd")),
            actions=RailActions(refs=(action,)),
            surfaces=(RailSurface(name="check input", direction="input", action=action),),
        ),
    )

    assert RailManifest.model_validate(manifest.model_dump()) == manifest
    assert iter_manifest_import_targets(manifest) == ("pathlib:Path.cwd", "pathlib:Path.cwd", "pathlib:Path.cwd")


def test_metadata_retains_unknown_keys():
    metadata = RailMetadata.model_validate({"display_name": "Acme", "catalog_id": "acme-42"})

    assert metadata.catalog_id == "acme-42"
    assert RailMetadata.model_validate(metadata.model_dump()) == metadata


def test_spec_rejects_unknown_keys():
    with pytest.raises(ValidationError):
        RailSpec.model_validate({"unknown_field": 1})


@pytest.mark.parametrize(
    "factory",
    (
        lambda: ConfigSpecRef(target="missing_colon"),
        lambda: ActionRef(name="", target="pathlib:Path"),
        lambda: ActionRef(name="path", target="pathlib"),
    ),
)
def test_import_refs_reject_invalid_targets(factory):
    with pytest.raises(ValueError):
        factory()


def test_import_refs_resolve_nested_attributes():
    ref = ActionRef(name="cwd", target="pathlib:Path.cwd")

    assert import_ref_target(ref) == "pathlib:Path.cwd"
    assert callable(resolve_import_ref(ref))


def test_catalog_indexes_manifests_and_surfaces():
    catalog = RailCatalog((_record("alpha", surface_name="check alpha"), _record("beta")))

    assert set(catalog.manifests) == {"alpha", "beta"}
    assert set(catalog.surfaces()) == {(RailDirection.INPUT, "check alpha")}


def test_catalog_rejects_duplicate_manifest_names():
    with pytest.raises(ValueError, match="already provided"):
        RailCatalog((_record("duplicate"), _record("duplicate")))


def test_catalog_rejects_duplicate_action_names():
    action = _action("shared")

    with pytest.raises(ValueError, match="already provided"):
        RailCatalog((_record("alpha", action=action), _record("beta", action=action)))


def test_catalog_rejects_duplicate_surface_keys():
    with pytest.raises(ValueError, match="already provided"):
        RailCatalog((_record("alpha", surface_name="shared"), _record("beta", surface_name="shared")))


def test_catalog_rejects_surface_with_undeclared_action():
    declared = _action("declared")
    undeclared = _action("undeclared")
    manifest = RailManifest(
        name="invalid",
        spec=RailSpec(
            actions=RailActions(refs=(declared,)),
            surfaces=(RailSurface(name="invalid", direction="input", action=undeclared),),
        ),
    )

    with pytest.raises(ValueError, match="not declared"):
        RailCatalog((RailManifestRecord(manifest=manifest, source="test:invalid"),))
