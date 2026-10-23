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

"""Server-owned manifest commands, independent of the library's CLI.

Validation and inspection construct routes but never enter application lifespan.
Serving uses the same application assembly and its existing resource ownership.
"""

import json
from pathlib import Path

import httpx
import typer
from fastapi import FastAPI

from nemoguardrails.server.experimental._application_manifest import (
    ApplicationManifest,
    ManifestError,
    load_manifest,
)
from nemoguardrails.server.experimental._configuration import ProxyConfig, resolve_provider
from nemoguardrails.server.experimental._configuration_error import DeploymentValidationError
from nemoguardrails.server.experimental.application import create_proxy_application
from nemoguardrails.server.experimental.provider.payload import validate_payload_projection_contract

app = typer.Typer(
    no_args_is_help=True, pretty_exceptions_enable=False, help="Validate, inspect or serve a proxy manifest."
)


def _prepare(path: Path) -> tuple[ProxyConfig, FastAPI]:
    """Validate through existing assembly, suppressing raw validation inputs."""
    try:
        config = load_manifest(path)
        application = create_proxy_application(config)
        return config, application
    except ManifestError as error:
        typer.echo(str(error), err=True)
    except DeploymentValidationError as error:
        typer.echo(f"Invalid application: {error}", err=True)
    except (ValueError, TypeError, OSError, RuntimeError, httpx.InvalidURL):
        typer.echo(
            "Invalid application: check Rails directories, provider URLs, instance IDs and route mounts.", err=True
        )
    raise typer.Exit(2)


@app.command()
def validate(manifest: Path = typer.Argument(..., help="Deployment-owned YAML file.")) -> None:
    """Check manifest data, directories and routing without loading Rails code."""
    _prepare(manifest)
    typer.echo("Valid declaration. Rails contents, checker compatibility and startup have not been validated.")


def inspect_config(config: ProxyConfig) -> dict:
    """Describe selected bindings, not inferred checker or provider capabilities."""
    providers = []
    for declaration in config.providers:
        provider = resolve_provider(declaration)
        prefix = config.path_prefix.rstrip("/") + provider.upstream.mount_path
        operations = []
        for endpoint in provider.endpoints:
            request = validate_payload_projection_contract(endpoint.guarded_request_model, "request")
            response = validate_payload_projection_contract(endpoint.guarded_response_model, "response")
            operations.append(
                {
                    "operation": endpoint.operation_name,
                    "routes": [
                        {
                            "path": prefix + route.route_path,
                            "guarded_methods": sorted(route.methods),
                            "passthrough_methods": sorted(route.passthrough_methods),
                        }
                        for route in endpoint.operation_paths
                        if route.route_path == endpoint.route_path
                    ],
                    "reserved_route_patterns": [
                        prefix + route.route_path
                        for route in endpoint.operation_paths
                        if route.route_path != endpoint.route_path
                    ],
                    "buffered": True,
                    "streaming": endpoint.stream is not None,
                    "profiles": {
                        "request": request.profile.value,
                        "response": response.profile.value,
                        "stream": endpoint.stream.classifier.contract.profile.value if endpoint.stream else None,
                    },
                }
            )
        providers.append(
            {
                "provider": provider.provider_type,
                "instance_id": provider.instance_id,
                "mount_path": prefix,
                "upstream": provider.upstream.base_url,
                "dialect": declaration.dialect or "native",
                "rails_config_id": declaration.rails_config_id,
                "operations": operations,
            }
        )
    return {
        "providers": providers,
        "rails_config_ids": [entry.id for entry in config.rails_configs],
        "limits": {
            "max_request_body_bytes": config.max_request_body_bytes,
            "max_response_body_bytes": config.max_response_body_bytes,
            "timeout": config.timeout,
            "config_resolution_timeout": config.config_resolution_timeout,
        },
        "routing": {
            "unselected_routes": "passthrough within provider mounts, excluding reserved application routes",
            "unsupported_guarded_shapes": "rejected when guard checks apply; not retried as passthrough",
            "mount_precedence": "most-specific mount first for passthrough",
        },
        "checks": "Not loaded. Profiles describe bound policy, not enabled Rails checks or checker compatibility.",
    }


@app.command(name="inspect")
def inspect_manifest(manifest: Path = typer.Argument(..., help="Deployment-owned YAML file.")) -> None:
    """Show effective routes, bound profiles and fixed Rails selections as JSON."""
    config, _ = _prepare(manifest)
    typer.echo(json.dumps(inspect_config(config), indent=2))


@app.command()
def schema() -> None:
    """Emit editor JSON Schema from the manifest model, without a second schema."""
    typer.echo(json.dumps(ApplicationManifest.model_json_schema(), indent=2))


@app.command()
def serve(
    manifest: Path = typer.Argument(..., help="Deployment-owned YAML file."),
    host: str = typer.Option("127.0.0.1", help="Listen address; non-loopback requires deployment access controls."),
    port: int = typer.Option(8000, min=0, max=65535, help="Listen port; 0 selects a free port."),
    timeout_graceful_shutdown: int = typer.Option(
        30, min=1, help="Seconds to drain in-flight requests before cancellation and application cleanup."
    ),
) -> None:
    """Run the existing application with fixed bindings and lifespan ownership."""
    import uvicorn

    _, application = _prepare(manifest)
    uvicorn.run(
        application,
        host=host,
        port=port,
        access_log=False,
        proxy_headers=False,
        timeout_graceful_shutdown=timeout_graceful_shutdown,
    )


def main() -> None:
    """Enter the server CLI from module invocation or a future console script."""
    app()
