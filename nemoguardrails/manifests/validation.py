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
import os
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Tuple

from packaging.markers import Marker
from packaging.specifiers import SpecifierSet

from nemoguardrails.manifests.manifest import PythonPackage, RailManifest


class RequirementStatus(str, Enum):
    OK = "ok"
    WARNING = "warning"
    ERROR = "error"
    INFO = "info"


@dataclass(frozen=True, slots=True)
class RequirementCheck:
    kind: str
    name: str
    status: RequirementStatus
    required: bool
    message: str


@dataclass(frozen=True, slots=True)
class RailValidationResult:
    rail_name: str
    checks: Tuple[RequirementCheck, ...]

    @property
    def valid(self) -> bool:
        return all(check.status != RequirementStatus.ERROR for check in self.checks)


@dataclass(frozen=True, slots=True)
class RailValidationReport:
    results: Tuple[RailValidationResult, ...]

    @property
    def valid(self) -> bool:
        return all(result.valid for result in self.results)

    @property
    def packages_to_install(self) -> Tuple[str, ...]:
        packages = []
        for result in self.results:
            for check in result.checks:
                if check.kind == "python_package" and check.status in {
                    RequirementStatus.ERROR,
                    RequirementStatus.WARNING,
                }:
                    packages.append(check.name)
        return tuple(sorted(set(packages), key=str.lower))


class RailDependencyError(ImportError):
    def __init__(self, rail_name: str, requirement: PythonPackage) -> None:
        self.rail_name = rail_name
        self.requirement = requirement
        package = requirement.distribution + (requirement.version or "")
        super().__init__(
            f"Rail {rail_name!r} requires Python package {package!r}. Install it with: pip install '{package}'"
        )


def _package_name(requirement: PythonPackage) -> str:
    return requirement.distribution + (requirement.version or "")


def _package_check(requirement: PythonPackage, runtime: bool) -> RequirementCheck:
    package_name = _package_name(requirement)
    marker_matches = requirement.marker is None or Marker(requirement.marker).evaluate()
    if not marker_matches:
        status = RequirementStatus.ERROR if requirement.required else RequirementStatus.INFO
        message = (
            "required package does not support this environment"
            if requirement.required
            else "optional package does not apply to this environment"
        )
        return RequirementCheck("python_package", package_name, status, requirement.required, message)
    try:
        installed = importlib.metadata.version(requirement.distribution)
    except importlib.metadata.PackageNotFoundError:
        status = RequirementStatus.ERROR if requirement.required else RequirementStatus.WARNING
        return RequirementCheck("python_package", package_name, status, requirement.required, "not installed")
    if requirement.version and installed not in SpecifierSet(requirement.version):
        status = RequirementStatus.ERROR if requirement.required else RequirementStatus.WARNING
        return RequirementCheck(
            "python_package",
            package_name,
            status,
            requirement.required,
            f"installed version {installed} is incompatible",
        )
    if runtime:
        try:
            importlib.import_module(requirement.import_name)
        except ModuleNotFoundError as error:
            if error.name == requirement.import_name or requirement.import_name.startswith(f"{error.name}."):
                status = RequirementStatus.ERROR if requirement.required else RequirementStatus.WARNING
                return RequirementCheck(
                    "python_package", package_name, status, requirement.required, "runtime import failed"
                )
            raise
    return RequirementCheck(
        "python_package", package_name, RequirementStatus.OK, requirement.required, f"installed ({installed})"
    )


def validate_rail_requirements(manifests: Iterable[RailManifest], *, runtime: bool = False) -> RailValidationReport:
    results = []
    for manifest in sorted(manifests, key=lambda item: item.name):
        checks = [_package_check(requirement, runtime) for requirement in manifest.requirements.python_packages]
        for env_var in manifest.requirements.env_vars:
            present = env_var.name in os.environ
            if present:
                status = RequirementStatus.OK
                message = "set"
            elif env_var.required:
                status = RequirementStatus.ERROR
                message = "not set"
            else:
                status = RequirementStatus.INFO
                message = "optional and not set"
            checks.append(RequirementCheck("environment_variable", env_var.name, status, env_var.required, message))
        for service in manifest.requirements.services:
            checks.append(
                RequirementCheck(
                    "service", service.name, RequirementStatus.INFO, service.required, "declared; not verified"
                )
            )
        for model in manifest.requirements.models:
            checks.append(
                RequirementCheck("model", model.type, RequirementStatus.INFO, model.required, "declared; not verified")
            )
        results.append(RailValidationResult(manifest.name, tuple(checks)))
    return RailValidationReport(tuple(results))


def require_python_package(rail_name: str, requirement: PythonPackage):
    try:
        return importlib.import_module(requirement.import_name)
    except ModuleNotFoundError as error:
        if error.name == requirement.import_name or requirement.import_name.startswith(f"{error.name}."):
            raise RailDependencyError(rail_name, requirement) from error
        raise
