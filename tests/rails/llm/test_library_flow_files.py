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

"""Gate: library rail flow files stay valid and consistent in both Colang dialects.

Every manifest-declared rail must ship flow files that parse, define the flow
names the manifest declares, and invoke only actions declared by the same
manifest. Parameterized flows are a Colang 2 feature, so a declared flow whose
Colang 2 definition takes parameters is exempt from the Colang 1 presence
requirement.
"""

import re
from pathlib import Path

from nemoguardrails.actions.action_dispatcher import ActionDispatcher
from nemoguardrails.colang import parse_colang_file
from nemoguardrails.manifests import all_rail_manifests
from nemoguardrails.utils import camelcase_to_snakecase

LIBRARY_ROOT = Path("nemoguardrails/library")

V1_EXECUTE_RE = re.compile(r"execute\s+([A-Za-z_][\w ]*?)\s*(?:\(|$)", re.MULTILINE)
V2_ACTION_RE = re.compile(r"(?:await|start)\s+([A-Z]\w*Action)\b")
V2_LOWERCASE_CALL_RE = re.compile(r"(?:await|start)\s+([a-z][a-z0-9_]*)\s*\(")


def _package_dir(manifest) -> Path:
    parts = manifest.origin.split(".")
    return LIBRARY_ROOT.joinpath(*parts[parts.index("library") + 1 : -1])


def _base_flow_name(name: str) -> str:
    return name.split("$")[0].strip()


def _manifests_with_flows():
    for name, manifest in sorted(all_rail_manifests().items()):
        if manifest.spec.flows is not None:
            yield name, manifest


def _parse_flow_files(manifest, violations: list):
    package_dir = _package_dir(manifest)
    parsed = {}
    for version, file_names in (("1.0", manifest.spec.flows.v1_files), ("2.x", manifest.spec.flows.files)):
        parsed[version] = []
        for file_name in file_names:
            path = package_dir / file_name
            if not path.exists():
                violations.append(f"{manifest.name}: missing {path}")
                continue
            try:
                result = parse_colang_file(file_name, content=path.read_text(encoding="utf-8"), version=version)
            except Exception as error:
                message = str(error).splitlines()[0]
                violations.append(f"{manifest.name}: {path} does not parse as Colang {version}: {message}")
                continue
            if not result:
                violations.append(f"{manifest.name}: {path} does not match Colang {version}")
                continue
            parsed[version].append((path, result))
    return parsed


def test_parse_flow_files_rejects_dialect_mismatches(monkeypatch):
    _, manifest = next(_manifests_with_flows())
    monkeypatch.setitem(_parse_flow_files.__globals__, "parse_colang_file", lambda *args, **kwargs: {})
    violations = []

    parsed = _parse_flow_files(manifest, violations)

    assert all(not files for files in parsed.values())
    assert violations
    assert all("does not match Colang" in violation for violation in violations)


def test_library_flow_files_parse_and_define_declared_flows():
    violations = []

    for rail_name, manifest in _manifests_with_flows():
        parsed = _parse_flow_files(manifest, violations)

        v1_defined = {flow["id"] for _, result in parsed["1.0"] for flow in result["flows"]}
        v2_flows = [flow for _, result in parsed["2.x"] for flow in result["flows"]]
        v2_defined = {flow.name for flow in v2_flows}
        v2_parameterized = {flow.name for flow in v2_flows if flow.parameters}

        for declared in manifest.spec.flows.flow_names:
            base_name = _base_flow_name(declared)
            if base_name not in v2_defined:
                violations.append(
                    f"{rail_name}: declared flow {base_name!r} is not defined in {manifest.spec.flows.files}"
                )
            if base_name not in v1_defined and base_name not in v2_parameterized:
                violations.append(
                    f"{rail_name}: declared flow {base_name!r} is not defined in {manifest.spec.flows.v1_files} "
                    "and its Colang 2 definition is not parameterized"
                )

    assert not violations, "\n".join(violations)


def test_library_flows_do_not_invoke_actions_as_flows():
    """A lowercase name after await/start is a FLOW reference in Colang 2.

    Awaiting an action by its snake_case registered name parses fine but
    resolves to a nonexistent flow at runtime; the action must be invoked as
    CamelCase plus the Action suffix.
    """
    dispatcher = ActionDispatcher()
    violations = []

    for rail_name, manifest in _manifests_with_flows():
        package_dir = _package_dir(manifest)
        for file_name in manifest.spec.flows.files:
            path = package_dir / file_name
            if not path.exists():
                continue
            for name in sorted(set(V2_LOWERCASE_CALL_RE.findall(path.read_text(encoding="utf-8")))):
                if dispatcher.has_registered(name):
                    violations.append(
                        f"{rail_name}: {path} awaits registered action {name!r} as a flow; "
                        "invoke it as CamelCase plus the Action suffix"
                    )

    assert not violations, "\n".join(violations)


def test_library_flow_actions_are_declared_by_owning_manifest():
    violations = []

    for rail_name, manifest in _manifests_with_flows():
        action_refs = manifest.actions.refs if manifest.actions is not None else ()
        declared_action_names = {action_ref.name for action_ref in action_refs}
        package_dir = _package_dir(manifest)
        for file_name in manifest.spec.flows.v1_files + manifest.spec.flows.files:
            path = package_dir / file_name
            if not path.exists():
                continue
            content = path.read_text(encoding="utf-8")
            invoked = {match.strip() for match in V1_EXECUTE_RE.findall(content)}
            invoked |= set(V2_ACTION_RE.findall(content))
            for action_name in sorted(invoked):
                declared_name = action_name
                if declared_name not in declared_action_names:
                    declared_name = camelcase_to_snakecase(declared_name.removesuffix("Action"))
                if declared_name not in declared_action_names:
                    violations.append(
                        f"{rail_name}: {path} invokes action {action_name!r}, which its manifest does not declare"
                    )

    assert not violations, "\n".join(violations)
