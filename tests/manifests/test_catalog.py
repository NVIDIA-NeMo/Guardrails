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

import types

import pytest

import nemoguardrails.manifests.catalog as catalog_module
from nemoguardrails.manifests import (
    ActionRef,
    Binding,
    ConfigSpecRef,
    RailActions,
    RailCatalog,
    RailConfigSchema,
    RailDirection,
    RailFlows,
    RailManifest,
    RailManifestRecord,
    RailSpec,
    RailSurface,
)


def _action(name: str = "check") -> ActionRef:
    return ActionRef(name=name, target="pathlib:Path.cwd")


def _record(
    name: str,
    *,
    action: ActionRef | None = None,
    surface_name: str | None = None,
    flow_names: tuple[str, ...] = (),
) -> RailManifestRecord:
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
        spec=RailSpec(
            flows=RailFlows(flow_names=flow_names) if flow_names else None,
            actions=RailActions(refs=(action,)),
            surfaces=surfaces,
        ),
    )
    return RailManifestRecord(manifest=manifest, source=f"test:{name}")


def test_catalog_indexes_manifests_and_surfaces():
    catalog = RailCatalog((_record("alpha", surface_name="check alpha", flow_names=("check alpha",)), _record("beta")))

    assert set(catalog.manifests) == {"alpha", "beta"}
    assert set(catalog.surfaces()) == {(RailDirection.INPUT, "check alpha")}
    assert catalog.owner_for_flow("check alpha") == "alpha"
    assert catalog.owner_for_flow("custom flow") is None


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


def test_catalog_reports_duplicate_surface_within_manifest():
    action = _action("shared")
    surface = RailSurface(name="shared", direction=RailDirection.INPUT, action=action)
    manifest = RailManifest(
        name="alpha",
        spec=RailSpec(actions=RailActions(refs=(action,)), surfaces=(surface, surface)),
    )

    with pytest.raises(ValueError, match="declares duplicate surface"):
        RailCatalog((RailManifestRecord(manifest=manifest, source="test:alpha"),))


def test_catalog_rejects_duplicate_config_key():
    shared_spec = ConfigSpecRef(target="pathlib:Path.cwd")

    def record(name):
        manifest = RailManifest(
            name=name, spec=RailSpec(config_schema=RailConfigSchema(key="shared", spec=shared_spec))
        )
        return RailManifestRecord(manifest=manifest, source=f"test:{name}")

    with pytest.raises(ValueError, match="config key"):
        RailCatalog((record("alpha"), record("beta")))


def test_catalog_rejects_duplicate_flow_name():
    with pytest.raises(ValueError, match="flow.*already provided"):
        RailCatalog((_record("alpha", flow_names=("shared",)), _record("beta", flow_names=("shared",))))


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


def test_discover_built_ins_loads_rail_modules(tmp_path, monkeypatch):
    rail_package = tmp_path / "content_safety"
    rail_package.mkdir()
    (rail_package / "rail.py").write_text("RAIL = None\n")

    discovered_manifest = RailManifest(name="content_safety")
    fake_module = types.SimpleNamespace(RAIL=discovered_manifest)

    def fake_import(module_name):
        assert module_name == "nemoguardrails.library.content_safety.rail"
        return fake_module

    monkeypatch.setattr(catalog_module.importlib, "import_module", fake_import)

    catalog = RailCatalog.discover_built_ins(library_path=tmp_path)

    assert set(catalog.manifests) == {"content_safety"}
    assert catalog.manifests["content_safety"].origin == "nemoguardrails.library.content_safety.rail"


def test_discover_built_ins_rejects_non_manifest_rail(tmp_path, monkeypatch):
    rail_package = tmp_path / "broken"
    rail_package.mkdir()
    (rail_package / "rail.py").write_text("RAIL = 1\n")

    monkeypatch.setattr(
        catalog_module.importlib,
        "import_module",
        lambda module_name: types.SimpleNamespace(RAIL="not-a-manifest"),
    )

    with pytest.raises(TypeError, match="must define RAIL"):
        RailCatalog.discover_built_ins(library_path=tmp_path)
