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

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, Mapping, Optional, Tuple

from packaging.markers import Marker

from nemoguardrails.manifests.manifest import RailManifest


@dataclass(frozen=True, slots=True)
class InstallationPackage:
    distribution: str
    import_name: str
    version: Optional[str]
    marker: Optional[str]
    required: bool
    applicable: bool
    description: Optional[str]
    rail_names: Tuple[str, ...]

    @property
    def requirement(self) -> str:
        value = self.distribution + (self.version or "")
        if self.marker is not None:
            value = f"{value}; {self.marker}"
        return value

    def to_dict(self) -> dict:
        value = asdict(self)
        value["requirement"] = self.requirement
        value["rail_names"] = list(self.rail_names)
        return value


@dataclass(frozen=True, slots=True)
class InstallationResource:
    name: str
    required: bool
    description: Optional[str]
    rail_names: Tuple[str, ...]

    def to_dict(self) -> dict:
        value = asdict(self)
        value["rail_names"] = list(self.rail_names)
        return value


@dataclass(frozen=True, slots=True)
class InstallationPlan:
    rail_names: Tuple[str, ...]
    python_packages: Tuple[InstallationPackage, ...]
    environment_variables: Tuple[InstallationResource, ...]
    services: Tuple[InstallationResource, ...]
    models: Tuple[InstallationResource, ...]

    @property
    def required_python_packages(self) -> Tuple[InstallationPackage, ...]:
        return tuple(package for package in self.python_packages if package.required and package.applicable)

    @property
    def optional_python_packages(self) -> Tuple[InstallationPackage, ...]:
        return tuple(package for package in self.python_packages if not package.required and package.applicable)

    @property
    def unsupported_python_packages(self) -> Tuple[InstallationPackage, ...]:
        return tuple(package for package in self.python_packages if not package.applicable)

    def to_dict(self) -> dict:
        return {
            "rail_names": list(self.rail_names),
            "python_packages": {
                "required": [package.to_dict() for package in self.required_python_packages],
                "optional": [package.to_dict() for package in self.optional_python_packages],
                "unsupported": [package.to_dict() for package in self.unsupported_python_packages],
            },
            "environment_variables": _resource_groups(self.environment_variables),
            "services": _resource_groups(self.services),
            "models": _resource_groups(self.models),
        }


def _resource_groups(resources: Tuple[InstallationResource, ...]) -> dict:
    return {
        "required": [resource.to_dict() for resource in resources if resource.required],
        "optional": [resource.to_dict() for resource in resources if not resource.required],
    }


def _installation_packages(
    manifests: Tuple[RailManifest, ...], environment: Optional[Mapping[str, str]]
) -> Tuple[InstallationPackage, ...]:
    declarations: Dict[tuple, set[str]] = {}
    for manifest in manifests:
        for package in manifest.requirements.python_packages:
            applicable = package.marker is None or Marker(package.marker).evaluate(environment=environment)
            key = (
                package.distribution,
                package.import_name,
                package.version,
                package.marker,
                package.required,
                applicable,
                package.description,
            )
            declarations.setdefault(key, set()).add(manifest.name)

    packages = []
    for declaration, rail_names in declarations.items():
        packages.append(InstallationPackage(*declaration, rail_names=tuple(sorted(rail_names))))
    return tuple(
        sorted(
            packages,
            key=lambda package: (
                package.distribution.lower(),
                package.version or "",
                package.marker or "",
                not package.required,
            ),
        )
    )


def _installation_resources(manifests: Tuple[RailManifest, ...], attribute: str) -> Tuple[InstallationResource, ...]:
    declarations: Dict[tuple, set[str]] = {}
    for manifest in manifests:
        for resource in getattr(manifest.requirements, attribute):
            name = resource.type if attribute == "models" else resource.name
            key = (name, resource.required, resource.description)
            declarations.setdefault(key, set()).add(manifest.name)

    resources = [
        InstallationResource(*declaration, rail_names=tuple(sorted(rail_names)))
        for declaration, rail_names in declarations.items()
    ]
    return tuple(sorted(resources, key=lambda resource: (resource.name.lower(), not resource.required)))


def build_installation_plan(
    manifests: Iterable[RailManifest], *, environment: Optional[Mapping[str, str]] = None
) -> InstallationPlan:
    manifests_by_name = {manifest.name: manifest for manifest in manifests}
    selected = tuple(manifests_by_name[name] for name in sorted(manifests_by_name))
    return InstallationPlan(
        rail_names=tuple(manifest.name for manifest in selected),
        python_packages=_installation_packages(selected, environment),
        environment_variables=_installation_resources(selected, "env_vars"),
        services=_installation_resources(selected, "services"),
        models=_installation_resources(selected, "models"),
    )
