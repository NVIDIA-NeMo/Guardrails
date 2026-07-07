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
    PythonPackage,
    RailActions,
    RailConfigSchema,
    RailDirection,
    RailManifest,
    RailMetadata,
    RailRequirements,
    RailSpec,
    RailSurface,
    import_ref_target,
    iter_manifest_import_refs,
    resolve_import_ref,
)


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
    assert tuple(import_ref_target(ref) for ref in iter_manifest_import_refs(manifest)) == (
        "pathlib:Path.cwd",
        "pathlib:Path.cwd",
        "pathlib:Path.cwd",
    )


def test_metadata_retains_unknown_keys():
    metadata = RailMetadata.model_validate({"display_name": "Acme", "catalog_id": "acme-42"})

    assert metadata.catalog_id == "acme-42"
    assert RailMetadata.model_validate(metadata.model_dump()) == metadata


def test_config_schema_preserves_export_names_when_serialized():
    schema = RailConfigSchema(
        key="test",
        spec=ConfigSpecRef(target="pathlib:Path.cwd"),
        export_names=("model", "threshold"),
    )

    assert RailConfigSchema.model_validate(schema.model_dump()) == schema


def test_spec_rejects_unknown_keys():
    with pytest.raises(ValidationError):
        RailSpec.model_validate({"unknown_field": 1})


def test_python_package_requirement_round_trip():
    package = PythonPackage(
        distribution="example-package",
        import_name="example_package",
        version=">=1,<2",
        marker="python_version < '3.13'",
        description="Example runtime",
    )

    assert PythonPackage.model_validate(package.model_dump()) == package


@pytest.mark.parametrize("field,value", [("version", ">>1"), ("marker", "python_version ??? '3.12'")])
def test_python_package_rejects_invalid_constraints(field, value):
    with pytest.raises(ValueError):
        PythonPackage(distribution="example", import_name="example", **{field: value})


def test_requirements_reject_duplicate_python_packages():
    with pytest.raises(ValueError, match="cannot be declared more than once"):
        RailRequirements(
            python_packages=(
                PythonPackage(distribution="Example_Package", import_name="example"),
                PythonPackage(distribution="example-package", import_name="example"),
            )
        )


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


def test_flat_manifest_is_rejected():
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

    with pytest.raises(ValidationError):
        RailManifest.model_validate(flat_manifest)


@pytest.mark.parametrize(
    "factory",
    (
        lambda: Binding(kind="context", action_param="text"),
        lambda: Binding(kind="surface_param", action_param="model"),
        lambda: Binding(kind="literal", action_param="mode", key="source", value="strict"),
    ),
)
def test_binding_source_matches_binding_kind(factory):
    with pytest.raises(ValidationError, match="source key"):
        factory()


def test_surface_rejects_duplicate_action_parameter_bindings():
    with pytest.raises(ValidationError, match="more than once"):
        RailSurface(
            name="duplicate bindings",
            direction="input",
            action=_action(),
            bindings=(Binding.context("text", "user_message"), Binding.literal("text", "value")),
        )
