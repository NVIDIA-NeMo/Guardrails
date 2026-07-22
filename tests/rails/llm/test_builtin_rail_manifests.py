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

import ast
import inspect
from pathlib import Path

from nemoguardrails.manifests import (
    ActionRef,
    Binding,
    ConfigSpecRef,
    RailDirection,
    TransformTarget,
    all_rail_manifests,
    default_rail_catalog,
    iter_manifest_import_refs,
    resolve_import_ref,
)

LEGACY_UNMANIFESTED_PACKAGES = frozenset({"attention", "utils"})


def test_library_action_packages_declare_manifests():
    """Every library package with actions must declare a rail.py manifest.

    Unmanifested rails fall back to legacy loading and bypass every
    catalog-keyed gate (manifest completeness, flow files, requirements).
    LEGACY_UNMANIFESTED_PACKAGES is the frozen set of pre-manifest packages;
    do not extend it for new rails.
    """
    library_root = Path("nemoguardrails/library")
    violations = []

    for path in sorted(library_root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        if "@action(" not in path.read_text(encoding="utf-8"):
            continue
        if path.relative_to(library_root).parts[0] in LEGACY_UNMANIFESTED_PACKAGES:
            continue
        package_dir = path.parent
        covered = False
        while package_dir != library_root:
            if (package_dir / "rail.py").exists():
                covered = True
                break
            package_dir = package_dir.parent
        if not covered:
            violations.append(str(path))

    assert not violations, "Library action packages without a rail.py manifest:\n" + "\n".join(violations)


def test_builtin_manifests_are_discovered_from_rail_modules():
    manifests = all_rail_manifests()

    assert manifests
    for name, manifest in manifests.items():
        assert manifest.name == name
        assert manifest.origin.endswith(".rail")
        assert manifest.metadata.display_name
        assert manifest.metadata.description


def test_builtin_manifest_docs_urls_resolve():
    manifests = all_rail_manifests()

    for manifest in manifests.values():
        docs_url = manifest.metadata.docs_url
        if docs_url is None or docs_url.startswith(("http://", "https://")):
            continue
        assert Path(docs_url).is_file(), f"{manifest.name}: {docs_url}"

    assert manifests["clavata"].metadata.docs_url == ("docs/configure-rails/guardrail-catalog/community/clavata.mdx")


def test_builtin_rail_modules_only_import_manifest_types():
    allowed_imports = {"nemoguardrails.manifests"}
    disallowed_imports = []

    for path in sorted(Path("nemoguardrails/library").rglob("rail.py")):
        module = ast.parse(path.read_text(encoding="utf-8"))
        for node in module.body:
            if isinstance(node, ast.Import):
                disallowed_imports.extend(
                    f"{path}: import {alias.name}" for alias in node.names if alias.name not in allowed_imports
                )
            elif isinstance(node, ast.ImportFrom):
                module_name = node.module or ""
                if module_name not in allowed_imports:
                    disallowed_imports.append(f"{path}: from {module_name} import ...")

    assert disallowed_imports == []


def test_builtin_manifest_refs_resolve_lazily():
    for manifest in all_rail_manifests().values():
        for ref in iter_manifest_import_refs(manifest):
            assert isinstance(ref, (ActionRef, ConfigSpecRef))
            assert callable(resolve_import_ref(ref))


def test_builtin_action_refs_match_decorated_names_and_bindings():
    for manifest in all_rail_manifests().values():
        if manifest.actions is None:
            continue
        for action_ref in manifest.actions.refs:
            action = resolve_import_ref(action_ref)
            assert action.action_meta["name"] == action_ref.name
        for surface in manifest.surfaces:
            action = resolve_import_ref(surface.action)
            parameters = inspect.signature(action).parameters
            for binding in surface.bindings:
                assert binding.action_param in parameters


def test_self_check_surfaces_bind_optional_variant():
    manifests = all_rail_manifests()

    for rail_name in ("self_check.input_check", "self_check.output_check"):
        surface = manifests[rail_name].surfaces[0]
        assert surface.bindings == (Binding.surface_param(action_param="variant", name="variant", required=False),)


def test_configurable_api_key_requirements_document_shipped_names():
    manifests = all_rail_manifests()

    expected = {
        "gliner": "NVIDIA_API_KEY",
        "jailbreak_detection": "NVIDIA_API_KEY",
        "trend_micro": "V1_API_KEY",
    }
    for rail_name, env_var_name in expected.items():
        env_vars = {env_var.name: env_var for env_var in manifests[rail_name].requirements.env_vars}
        assert env_vars[env_var_name].required is False
        assert env_vars[env_var_name].description


def test_jailbreak_requirements_document_local_environment_variables():
    env_vars = {env_var.name: env_var for env_var in all_rail_manifests()["jailbreak_detection"].requirements.env_vars}

    expected = {
        "HF_TOKEN",
        "HF_HOME",
        "HF_HUB_OFFLINE",
        "JAILBREAK_CHECK_DEVICE",
        "EMBEDDING_CLASSIFIER_PATH",
    }
    for env_var_name in expected:
        assert env_vars[env_var_name].required is False
        assert env_vars[env_var_name].description


def test_polygraf_manifest_declares_remote_endpoint_consistently():
    manifest = all_rail_manifests()["polygraf"]

    assert tuple(service.name for service in manifest.requirements.services) == ("Polygraf endpoint",)
    assert manifest.privacy.remote_services == ("Polygraf endpoint",)


def test_builtin_surfaces_preserve_flow_contracts():
    surfaces = default_rail_catalog().surfaces()

    assert surfaces[(RailDirection.INPUT, "clavata check input")].bindings == (
        Binding.context("text", "user_message"),
        Binding.literal("rail", "input"),
    )
    assert surfaces[(RailDirection.OUTPUT, "clavata check output")].bindings == (
        Binding.context("text", "bot_message"),
        Binding.literal("rail", "output"),
    )
    assert surfaces[(RailDirection.INPUT, "context bloat detection on input")].bindings == (
        Binding.literal("source", "input"),
        Binding.context("text", "user_message"),
    )
    assert surfaces[(RailDirection.INPUT, "context bloat detection on input")].transform_target == (
        TransformTarget.USER_MESSAGE
    )
    assert surfaces[(RailDirection.RETRIEVAL, "context bloat detection on retrieval")].bindings == (
        Binding.literal("source", "retrieval"),
        Binding.context("text", "relevant_chunks"),
    )
    assert surfaces[(RailDirection.RETRIEVAL, "context bloat detection on retrieval")].transform_target == (
        TransformTarget.RELEVANT_CHUNKS
    )
    assert (
        surfaces[(RailDirection.INPUT, "jailbreak detection heuristics")].action.name
        == "jailbreak_detection_heuristics"
    )
    assert surfaces[(RailDirection.OUTPUT, "autoalign groundedness output")].action.name == (
        "autoalign_groundedness_output_api"
    )
    assert surfaces[(RailDirection.INPUT, "activefence moderation on input")].bindings[-1] == Binding.literal(
        "threshold_mode", "simple"
    )
    assert surfaces[(RailDirection.INPUT, "activefence moderation on input detailed")].bindings[-1] == (
        Binding.literal("threshold_mode", "detailed")
    )
    assert surfaces[(RailDirection.INPUT, "gcpnlp moderation")].bindings == (
        Binding.literal("threshold_mode", "simple"),
    )
    assert surfaces[(RailDirection.INPUT, "gcpnlp moderation detailed")].bindings == (
        Binding.literal("threshold_mode", "detailed"),
    )
    assert surfaces[(RailDirection.RETRIEVAL, "hf classifier check retrieval")].transform_target == (
        TransformTarget.RELEVANT_CHUNKS
    )
    assert surfaces[(RailDirection.RETRIEVAL, "regex check retrieval")].transform_target == (
        TransformTarget.RELEVANT_CHUNKS
    )
