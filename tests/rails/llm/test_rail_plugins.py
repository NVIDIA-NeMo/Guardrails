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

from types import SimpleNamespace

import pytest

from nemoguardrails.actions import action
from nemoguardrails.actions.action_dispatcher import ActionDispatcher
from nemoguardrails.actions.rail_outcome import RailOutcome
from nemoguardrails.colang.v1_0.runtime.runtime import RuntimeV1_0
from nemoguardrails.manifests import (
    ActionRef,
    ConfigSpecRef,
    RailActions,
    RailConfigSchema,
    RailDirection,
    RailFlows,
    RailManifest,
    RailManifestRecord,
    RailSpec,
    RailSurface,
    rail_catalog,
    registry,
)
from nemoguardrails.manifests.catalog import RailCatalog
from nemoguardrails.manifests.config_schema import Field, RailConfigBaseModel, RailConfigSpec, rail_field
from nemoguardrails.rails.llm.config import RailsConfig


class ExamplePluginConfig(RailConfigBaseModel):
    threshold: float = Field(default=0.5)


def build_example_plugin_config() -> RailConfigSpec:
    return RailConfigSpec(
        annotation=ExamplePluginConfig,
        field_info=rail_field(default_factory=ExamplePluginConfig),
        exports={"ExamplePluginConfig": ExamplePluginConfig},
    )


@action(name="example_plugin_check")
async def example_plugin_action():
    return RailOutcome.allow(metadata={"plugin": "example_plugin"})


PLUGIN = RailManifest(
    name="example_plugin",
    spec=RailSpec(
        config_schema=RailConfigSchema(
            key="example_plugin",
            spec=ConfigSpecRef(target="tests.rails.llm.test_rail_plugins:build_example_plugin_config"),
        ),
        flows=RailFlows(v1_files=("example_plugin_flows.co",)),
        actions=RailActions(
            refs=(
                ActionRef(
                    name="example_plugin_check",
                    target="tests.rails.llm.test_rail_plugins:example_plugin_action",
                ),
            )
        ),
        surfaces=(
            RailSurface(
                name="example plugin check input",
                direction=RailDirection.INPUT,
                action=ActionRef(
                    name="example_plugin_check",
                    target="tests.rails.llm.test_rail_plugins:example_plugin_action",
                ),
            ),
        ),
    ),
)


class FakeEntryPoint:
    name = "example_plugin"
    value = "tests.rails.llm.test_rail_plugins:PLUGIN"
    dist = SimpleNamespace(name="example-distribution")

    def __init__(self):
        self.loads = 0

    def load(self):
        self.loads += 1
        return PLUGIN


class FakeEntryPoints(list):
    def select(self, *, group):
        return self if group == "nemoguardrails.rails" else []


@pytest.fixture
def plugin_entry_point(monkeypatch):
    entry_point = FakeEntryPoint()
    monkeypatch.setattr(
        "nemoguardrails.manifests.catalog.importlib.metadata.entry_points",
        lambda: FakeEntryPoints([entry_point]),
    )
    registry._reset_rail_manifest_cache()
    yield entry_point
    registry._reset_rail_manifest_cache()


def test_disabled_plugin_is_not_loaded(plugin_entry_point):
    rail_catalog()
    assert plugin_entry_point.loads == 0


def test_enabled_plugin_contributes_config_per_configuration(plugin_entry_point):
    config = RailsConfig.from_content(
        config={
            "models": [],
            "rails": {
                "plugins": {"enabled": ["example_plugin"]},
                "config": {"example_plugin": {"threshold": 0.8}},
            },
        }
    )
    assert plugin_entry_point.loads == 1
    assert config.rails.config.example_plugin.threshold == 0.8
    assert type(config) is RailsConfig.for_plugins(["example_plugin"])
    schema = type(config).model_json_schema()
    assert "example_plugin" in schema["$defs"][type(config.rails.config).__name__]["properties"]


def test_model_validate_builds_plugin_specific_config(plugin_entry_point):
    config = RailsConfig.model_validate(
        {
            "models": [],
            "rails": {
                "plugins": {"enabled": ["example_plugin"]},
                "config": {"example_plugin": {"threshold": 0.8}},
            },
        }
    )

    assert type(config) is RailsConfig.for_plugins(["example_plugin"])
    assert config.rails.config.example_plugin.threshold == 0.8


def test_direct_constructor_rejects_enabled_plugins(plugin_entry_point):
    with pytest.raises(ValueError, match=r"use RailsConfig\.for_plugins"):
        RailsConfig(
            models=[],
            rails={
                "plugins": {"enabled": ["example_plugin"]},
                "config": {"example_plugin": {"threshold": 0.8}},
            },
        )


def test_unknown_enabled_plugin_fails_before_config_validation(plugin_entry_point):
    with pytest.raises(ValueError, match="not installed"):
        RailsConfig.from_content(config={"models": [], "rails": {"plugins": {"enabled": ["missing_plugin"]}}})


def test_enabled_plugin_contributes_lazy_action_and_surface(plugin_entry_point):
    catalog = rail_catalog(["example_plugin"])
    dispatcher = ActionDispatcher(rail_catalog=catalog)
    resolved_action = dispatcher.get_action("example_plugin_check")
    assert getattr(resolved_action, "action_meta")["name"] == "example_plugin_check"
    assert (RailDirection.INPUT, "example plugin check input") in catalog.surfaces()


def test_enabled_plugin_action_is_registered_with_colang_runtime(plugin_entry_point):
    config = RailsConfig.from_content(config={"models": [], "rails": {"plugins": {"enabled": ["example_plugin"]}}})
    runtime = RuntimeV1_0(config)

    assert runtime.action_dispatcher.has_registered("example_plugin_check")


def test_enabled_plugin_contributes_colang_flow_resources(plugin_entry_point):
    config = RailsConfig.from_content(
        config={
            "models": [],
            "rails": {
                "plugins": {"enabled": ["example_plugin"]},
                "input": {"flows": ["example plugin check input"]},
            },
        }
    )

    assert "example plugin check input" in {flow["id"] for flow in config.flows}


def test_from_path_loads_enabled_plugin_flow_resources(plugin_entry_point, tmp_path):
    config_path = tmp_path / "config.yml"
    config_path.write_text(
        "models: []\n"
        "rails:\n"
        "  plugins:\n"
        "    enabled:\n"
        "      - example_plugin\n"
        "  input:\n"
        "    flows:\n"
        "      - example plugin check input\n"
    )

    config = RailsConfig.from_path(str(config_path))

    assert "example plugin check input" in {flow["id"] for flow in config.flows}


def test_missing_plugin_flow_resource_fails_during_config_loading(plugin_entry_point, monkeypatch):
    missing = PLUGIN.model_copy(
        update={"spec": PLUGIN.spec.model_copy(update={"flows": RailFlows(v1_files=("missing.co",))})}
    )
    monkeypatch.setattr(plugin_entry_point, "load", lambda: missing)

    with pytest.raises(FileNotFoundError, match="missing.co"):
        RailsConfig.from_content(config={"models": [], "rails": {"plugins": {"enabled": ["example_plugin"]}}})


def test_catalog_rejects_duplicate_public_action_names():
    action_ref = ActionRef(name="duplicate", target="pathlib:Path.cwd")
    records = (
        RailManifestRecord(
            manifest=RailManifest(name="first", spec=RailSpec(actions=RailActions(refs=(action_ref,)))),
            source="first",
        ),
        RailManifestRecord(
            manifest=RailManifest(name="second", spec=RailSpec(actions=RailActions(refs=(action_ref,)))),
            source="second",
        ),
    )

    with pytest.raises(ValueError, match="Rail action 'duplicate'.*first.*second"):
        RailCatalog(records)


def test_catalog_rejects_duplicate_config_keys():
    config_schema = RailConfigSchema(
        key="duplicate",
        spec=ConfigSpecRef(target="tests.rails.llm.test_rail_plugins:build_example_plugin_config"),
    )
    records = (
        RailManifestRecord(
            manifest=RailManifest(name="first", spec=RailSpec(config_schema=config_schema)), source="first"
        ),
        RailManifestRecord(
            manifest=RailManifest(name="second", spec=RailSpec(config_schema=config_schema)), source="second"
        ),
    )

    with pytest.raises(ValueError, match="Rail config key 'duplicate'.*first.*second"):
        RailCatalog(records)


def test_catalog_rejects_surface_actions_not_declared_by_manifest():
    action_ref = ActionRef(name="missing", target="pathlib:Path.cwd")
    record = RailManifestRecord(
        manifest=RailManifest(
            name="invalid",
            spec=RailSpec(
                surfaces=(RailSurface(name="invalid input", direction=RailDirection.INPUT, action=action_ref),)
            ),
        ),
        source="invalid",
    )

    with pytest.raises(ValueError, match="not declared"):
        RailCatalog((record,))
