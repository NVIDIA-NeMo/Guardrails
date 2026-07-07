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

import importlib
import importlib.metadata

from nemoguardrails.manifests import (
    EnvVar,
    PythonPackage,
    RailCatalog,
    RailManifest,
    RailManifestRecord,
    RailRequirements,
    RailSpec,
    RequirementStatus,
    configured_rail_manifests,
    validate_rail_requirements,
)
from nemoguardrails.rails.llm.config import RailsConfig


def _manifest(name="test", *, package=None, env_vars=(), flow_names=(), config_schema=None):
    requirements = RailRequirements(python_packages=(package,) if package else (), env_vars=env_vars)
    from nemoguardrails.manifests import RailFlows

    return RailManifest(
        name=name,
        spec=RailSpec(
            flows=RailFlows(flow_names=flow_names) if flow_names else None,
            config_schema=config_schema,
            requirements=requirements,
        ),
    )


def test_static_validation_uses_distribution_metadata_without_importing(monkeypatch):
    package = PythonPackage(distribution="example", import_name="example", version=">=2")
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "2.1")
    imported = []
    monkeypatch.setattr(importlib, "import_module", lambda name: imported.append(name))

    report = validate_rail_requirements([_manifest(package=package)])

    assert report.valid
    assert report.results[0].checks[0].status == RequirementStatus.OK
    assert imported == []


def test_missing_required_and_optional_packages(monkeypatch):
    def missing(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", missing)
    required = _manifest(package=PythonPackage(distribution="required", import_name="required"))
    optional = _manifest(
        "optional", package=PythonPackage(distribution="optional", import_name="optional", required=False)
    )

    report = validate_rail_requirements([required, optional])

    assert not report.valid
    assert report.results[0].checks[0].status == RequirementStatus.WARNING
    assert report.results[1].checks[0].status == RequirementStatus.ERROR


def test_required_unsupported_environment_is_an_error():
    package = PythonPackage(
        distribution="example",
        import_name="example",
        marker="python_version < '1'",
    )

    report = validate_rail_requirements([_manifest(package=package)])

    assert not report.valid
    assert report.results[0].checks[0].message == "required package does not support this environment"


def test_runtime_validation_reports_transitive_import_failure(monkeypatch):
    requirement = PythonPackage(distribution="example", import_name="example")
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "1.0")

    def missing(name):
        raise ModuleNotFoundError("missing", name="transitive_dependency")

    monkeypatch.setattr(importlib, "import_module", missing)

    report = validate_rail_requirements([_manifest(package=requirement)], runtime=True)

    assert not report.valid
    assert report.results[0].checks[0].message == "runtime import failed"


def test_environment_validation_does_not_expose_values(monkeypatch):
    monkeypatch.setenv("SECRET_TOKEN", "top-secret")
    manifest = _manifest(env_vars=(EnvVar(name="SECRET_TOKEN", required=True),))

    report = validate_rail_requirements([manifest])

    assert report.valid
    assert "top-secret" not in repr(report)


def test_configured_manifest_selection_normalizes_parameters_and_ignores_custom_flows():
    manifest = _manifest("owned", flow_names=("owned flow",))
    catalog = RailCatalog((RailManifestRecord(manifest=manifest, source="test"),))
    config = RailsConfig.from_content(
        yaml_content="""
models: []
rails:
  input:
    flows:
      - owned flow $mode=\"strict\"
      - custom flow
"""
    )

    assert configured_rail_manifests(config, catalog) == (manifest,)


def test_configured_manifest_selection_includes_explicit_config_sections():
    from nemoguardrails.manifests import default_rail_catalog

    catalog = default_rail_catalog()
    config = RailsConfig.from_content(
        yaml_content="""
models: []
rails:
  config:
    sensitive_data_detection:
      input:
        entities: [EMAIL_ADDRESS]
"""
    )

    assert tuple(item.name for item in configured_rail_manifests(config, catalog)) == ("sensitive_data_detection",)
