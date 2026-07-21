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

from nemoguardrails.manifests import (
    EnvVar,
    ModelRequirement,
    PythonPackage,
    RailManifest,
    RailRequirements,
    RailSpec,
    ServiceRequirement,
    build_installation_plan,
)


def _manifest(
    name: str,
    *,
    packages=(),
    env_vars=(),
    services=(),
    models=(),
) -> RailManifest:
    return RailManifest(
        name=name,
        spec=RailSpec(
            requirements=RailRequirements(
                python_packages=packages,
                env_vars=env_vars,
                services=services,
                models=models,
            )
        ),
    )


def test_installation_plan_classifies_packages_and_preserves_declarations():
    manifest = _manifest(
        "example",
        packages=(
            PythonPackage(
                distribution="required",
                import_name="required",
                version=">=2,<3",
                marker="python_version >= '3.10'",
            ),
            PythonPackage(distribution="optional", import_name="optional", required=False),
            PythonPackage(
                distribution="unsupported",
                import_name="unsupported",
                marker="python_version < '3.10'",
            ),
        ),
    )

    plan = build_installation_plan([manifest], environment={"python_version": "3.12"})

    assert tuple(package.requirement for package in plan.required_python_packages) == (
        "required>=2,<3; python_version >= '3.10'",
    )
    assert tuple(package.requirement for package in plan.optional_python_packages) == ("optional",)
    assert tuple(package.requirement for package in plan.unsupported_python_packages) == (
        "unsupported; python_version < '3.10'",
    )


def test_installation_plan_aggregates_matching_declarations_with_provenance():
    package = PythonPackage(distribution="shared", import_name="shared", version=">=1")

    plan = build_installation_plan([_manifest("zeta", packages=(package,)), _manifest("alpha", packages=(package,))])

    assert plan.rail_names == ("alpha", "zeta")
    assert len(plan.python_packages) == 1
    assert plan.python_packages[0].rail_names == ("alpha", "zeta")


def test_installation_plan_groups_declared_resources_without_values():
    manifest = _manifest(
        "example",
        env_vars=(EnvVar(name="REQUIRED_TOKEN", required=True), EnvVar(name="OPTIONAL_TOKEN")),
        services=(ServiceRequirement(name="Example API", required=True),),
        models=(ModelRequirement(type="spacy:example", required=True),),
    )

    plan = build_installation_plan([manifest])
    serialized = plan.to_dict()

    assert serialized["environment_variables"] == {
        "required": [
            {
                "name": "REQUIRED_TOKEN",
                "required": True,
                "description": None,
                "rail_names": ["example"],
            }
        ],
        "optional": [
            {
                "name": "OPTIONAL_TOKEN",
                "required": False,
                "description": None,
                "rail_names": ["example"],
            }
        ],
    }
    assert serialized["services"]["required"][0]["name"] == "Example API"
    assert serialized["models"]["required"][0]["name"] == "spacy:example"


def test_installation_plan_is_empty_for_rails_without_requirements():
    plan = build_installation_plan([_manifest("empty")])

    assert plan.rail_names == ("empty",)
    assert plan.python_packages == ()
    assert plan.environment_variables == ()
    assert plan.services == ()
    assert plan.models == ()
