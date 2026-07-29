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

import shlex
from pathlib import Path

import typer

from nemoguardrails.manifests import (
    RequirementStatus,
    configured_rail_manifests,
    default_rail_catalog,
    validate_rail_requirements,
)
from nemoguardrails.rails.llm.config import RailsConfig

app = typer.Typer()


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
