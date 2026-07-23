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

import json
import shlex
from enum import Enum
from pathlib import Path
from typing import Iterable, Optional

import typer
from packaging.markers import Marker

from nemoguardrails.manifests import (
    InstallationPackage,
    InstallationPlan,
    InstallationResource,
    RequirementStatus,
    build_installation_plan,
    configured_rail_manifests,
    default_rail_catalog,
    validate_rail_requirements,
)
from nemoguardrails.rails.llm.config import RailsConfig

app = typer.Typer()


class RequirementsOutputFormat(str, Enum):
    HUMAN = "human"
    REQUIREMENTS = "requirements"
    JSON = "json"


def _rail_names(rail_names: Iterable[str]) -> str:
    return ", ".join(rail_names)


def _description_suffix(description: Optional[str]) -> str:
    return f": {' '.join(description.split())}" if description else ""


def _package_lines(title: str, packages: Iterable[InstallationPackage]) -> list[str]:
    packages = tuple(packages)
    lines = [f"{title}:"]
    if not packages:
        return [*lines, "  none"]
    for package in packages:
        lines.append(
            f"  - {package.requirement} (rails: {_rail_names(package.rail_names)})"
            f"{_description_suffix(package.description)}"
        )
    return lines


def _resource_lines(title: str, resources: Iterable[InstallationResource]) -> list[str]:
    resources = tuple(resources)
    lines = [f"{title}:"]
    if not resources:
        return [*lines, "  none"]
    for required in (True, False):
        selected = tuple(resource for resource in resources if resource.required is required)
        if not selected:
            continue
        lines.append(f"  {'Required' if required else 'Optional'}:")
        for resource in selected:
            lines.append(
                f"    - {resource.name} (rails: {_rail_names(resource.rail_names)})"
                f"{_description_suffix(resource.description)}"
            )
    return lines


def _install_command(packages: Iterable[InstallationPackage]) -> str:
    requirements = []
    for package in packages:
        requirement = package.distribution + (package.version or "")
        if package.marker is not None:
            requirement = f"{requirement}; {Marker(package.marker)}"
        requirements.append(shlex.quote(requirement))
    return "pip install " + " ".join(requirements)


def _render_human_installation_plan(plan: InstallationPlan) -> str:
    lines = [f"Rails: {_rail_names(plan.rail_names) or 'none'}"]
    lines.extend(_package_lines("Required Python packages", plan.required_python_packages))
    if plan.required_python_packages:
        lines.append(f"Required install command: {_install_command(plan.required_python_packages)}")
    lines.extend(_package_lines("Optional Python packages", plan.optional_python_packages))
    if plan.optional_python_packages:
        lines.append(f"Optional install command: {_install_command(plan.optional_python_packages)}")
    lines.extend(_package_lines("Unsupported Python packages in this environment", plan.unsupported_python_packages))
    lines.extend(_resource_lines("Environment variables", plan.environment_variables))
    lines.extend(_resource_lines("Services", plan.services))
    lines.extend(_resource_lines("Models", plan.models))
    return "\n".join(lines)


def _requirements_comments(title: str, resources: Iterable[InstallationResource]) -> list[str]:
    resources = tuple(resources)
    if not resources:
        return []
    lines = ["", f"# {title}"]
    for resource in resources:
        qualifier = "required" if resource.required else "optional"
        lines.append(
            f"# {resource.name} ({qualifier}; rails: {_rail_names(resource.rail_names)})"
            f"{_description_suffix(resource.description)}"
        )
    return lines


def _render_requirements_file(plan: InstallationPlan) -> str:
    lines = [f"# Rails: {_rail_names(plan.rail_names) or 'none'}", "", "# Required Python packages"]
    if plan.required_python_packages:
        lines.extend(package.requirement for package in plan.required_python_packages)
    else:
        lines.append("# None")
    if plan.optional_python_packages:
        lines.extend(("", "# Optional Python packages (commented out)"))
        lines.extend(f"# {package.requirement}" for package in plan.optional_python_packages)
    if plan.unsupported_python_packages:
        lines.extend(("", "# Unsupported Python packages in this environment"))
        lines.extend(f"# {package.requirement}" for package in plan.unsupported_python_packages)
    lines.extend(_requirements_comments("Environment variables", plan.environment_variables))
    lines.extend(_requirements_comments("Services", plan.services))
    lines.extend(_requirements_comments("Models", plan.models))
    return "\n".join(lines)


def _select_installation_manifests(rail: Optional[list[str]], config: Optional[Path]):
    if rail and config is not None:
        raise typer.BadParameter("Use either --rail or --config, not both.")
    if not rail and config is None:
        raise typer.BadParameter("Provide at least one --rail or a --config path.")
    catalog = default_rail_catalog()
    if config is not None:
        rails_config = RailsConfig.from_path(str(config))
        return configured_rail_manifests(rails_config, catalog)
    missing = sorted(set(rail or ()) - set(catalog.manifests))
    if missing:
        raise typer.BadParameter(f"Unknown rail names: {', '.join(missing)}.", param_hint="--rail")
    return tuple(catalog.manifests[name] for name in sorted(set(rail or ())))


@app.command("requirements")
def show_rail_requirements(
    rail: Optional[list[str]] = typer.Option(None, "--rail", help="Rail name. Repeat to select multiple rails."),
    config: Optional[Path] = typer.Option(None, "--config", exists=True, help="Guardrails configuration path."),
    output_format: RequirementsOutputFormat = typer.Option(
        RequirementsOutputFormat.HUMAN,
        "--format",
        help="Output format: human, requirements, or json.",
    ),
):
    """Print installation requirements declared by selected rails."""
    manifests = _select_installation_manifests(rail, config)
    plan = build_installation_plan(manifests)
    if output_format == RequirementsOutputFormat.JSON:
        typer.echo(json.dumps(plan.to_dict(), indent=2, sort_keys=True))
    elif output_format == RequirementsOutputFormat.REQUIREMENTS:
        typer.echo(_render_requirements_file(plan))
    else:
        typer.echo(_render_human_installation_plan(plan))


@app.command("validate")
def validate_rails(
    config: Path = typer.Option(Path("config"), "--config", exists=True, help="Guardrails configuration path."),
    runtime: bool = typer.Option(False, "--runtime", help="Import declared Python packages after static checks."),
):
    """Validate dependencies declared by configured rails."""
    rails_config = RailsConfig.from_path(str(config))
    catalog = default_rail_catalog()
    manifests = configured_rail_manifests(rails_config, catalog)
    report = validate_rail_requirements(manifests, runtime=runtime)
    if not report.results:
        typer.echo("No catalog rails are configured.")
    for result in report.results:
        typer.echo(f"{result.rail_name}:")
        if not result.checks:
            typer.echo("  [OK] no declared requirements")
        for check in result.checks:
            label = {
                RequirementStatus.OK: "OK",
                RequirementStatus.WARNING: "WARNING",
                RequirementStatus.ERROR: "ERROR",
                RequirementStatus.INFO: "INFO",
            }[check.status]
            typer.echo(f"  [{label}] {check.kind}: {check.name} ({check.message})")
    if report.required_packages_to_install:
        packages = " ".join(shlex.quote(package) for package in report.required_packages_to_install)
        typer.echo(f"Install required packages with: pip install {packages}")
    if report.optional_packages_to_install:
        packages = " ".join(shlex.quote(package) for package in report.optional_packages_to_install)
        typer.echo(f"Install optional packages to enable additional functionality with: pip install {packages}")
    if not report.valid:
        raise typer.Exit(1)
