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
    ConfigSpecRef,
    RailActions,
    RailConfigSchema,
    RailDirection,
    RailManifest,
    RailMetadata,
    RailSpec,
    RailSurface,
    import_ref_target,
    iter_manifest_import_refs,
    iter_manifest_import_targets,
    normalize_configured_surface_name,
    parse_configured_surface,
    resolve_import_ref,
)
from nemoguardrails.manifests.manifest import configured_rail_surfaces


def _action(name: str = "check") -> ActionRef:
    return ActionRef(name=name, target="pathlib:Path.cwd")


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


def test_import_ref_target_rejects_non_ref():
    with pytest.raises(TypeError):
        import_ref_target("pathlib:Path.cwd")


def test_iter_manifest_import_refs_covers_config_actions_and_surfaces():
    action = _action("cwd")
    manifest = RailManifest(
        name="sample",
        spec=RailSpec(
            config_schema=RailConfigSchema(key="cfg", spec=ConfigSpecRef(target="pathlib:Path.cwd")),
            actions=RailActions(refs=(action,)),
            surfaces=(RailSurface(name="surface", direction=RailDirection.INPUT, action=action),),
        ),
    )

    assert len(iter_manifest_import_refs(manifest)) == 3


def test_parse_configured_surface_parenthesized_form():
    name, parameters = parse_configured_surface("content safety check($model=abc, $threshold=0.5)")

    assert name == "content safety check"
    assert parameters == {"model": "abc", "threshold": "0.5"}


def test_parse_configured_surface_dollar_form():
    name, parameters = parse_configured_surface("self check input $model=gpt-4o")

    assert name == "self check input"
    assert parameters == {"model": "gpt-4o"}
    assert normalize_configured_surface_name("self check input $model=gpt-4o") == "self check input"


def test_parse_configured_surface_bare_name():
    assert parse_configured_surface("plain flow") == ("plain flow", {})


@pytest.mark.parametrize(
    ("configured", "expected"),
    (
        (
            'content safety check input $model="content safety" $mode=strict',
            ("content safety check input", {"model": "content safety", "mode": "strict"}),
        ),
        (
            'content safety check input(model="content,safety", mode=strict)',
            ("content safety check input", {"model": "content,safety", "mode": "strict"}),
        ),
        (
            'check $pattern="foo$bar" $mode=strict',
            ("check", {"pattern": "foo$bar", "mode": "strict"}),
        ),
    ),
)
def test_configured_surface_parser_supports_quoted_parameters(configured, expected):
    assert parse_configured_surface(configured) == expected


@pytest.mark.parametrize(
    "configured",
    (
        "",
        "content safety check input $model",
        "content safety check input $model=",
        "content safety check input $model=first $model=second",
        "content safety check input $model=first$mode=strict",
        "content safety check input(model)",
        "content safety check input(model=first, model=second)",
        'content safety check input $model="unterminated',
        "content safety check input(model=value",
    ),
)
def test_configured_surface_parser_rejects_malformed_parameters(configured):
    with pytest.raises(ValueError):
        parse_configured_surface(configured)


def test_configured_rail_surfaces_selects_unique_declared_surfaces():
    action = _action()
    surface = RailSurface(name="check", direction=RailDirection.INPUT, action=action)
    surfaces = {(RailDirection.INPUT, "check"): surface}

    selected = configured_rail_surfaces(
        RailDirection.INPUT,
        ["check($model=gpt-4)", "check($model=claude)", "unknown"],
        surfaces,
    )

    assert selected == {"check": surface}


def test_flat_manifest_is_normalized_into_spec():
    flat_manifest = {
        "name": "flat",
        "actions": {"refs": [{"name": "act", "target": "pathlib:Path.cwd"}]},
        "surfaces": [
            {
                "name": "surface",
                "direction": "input",
                "action": {"name": "act", "target": "pathlib:Path.cwd"},
            }
        ],
    }

    manifest = RailManifest.model_validate(flat_manifest)

    assert manifest.actions.refs[0].name == "act"
    assert manifest.surfaces[0].name == "surface"
