#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""
Typer CLI wrapper for running Locust load tests against NeMo Guardrails server.

This module provides a command-line interface for running load tests, supporting
both direct CLI arguments and YAML configuration files.
"""

import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx
import typer
import yaml
from pydantic import ValidationError

from benchmark.locust.locust_models import LocustConfig

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(formatter)

log.addHandler(console_handler)

app = typer.Typer(
    help="Locust load testing application for NeMo Guardrails",
    add_completion=False,
)


class LocustRunner:
    """Run Locust load tests against NeMo Guardrails server."""

    def __init__(self, config: LocustConfig):
        self.config = config
        self.locustfile_path = Path(__file__).parent / "locustfile.py"

    def _check_service(self) -> None:
        """Check if the NeMo Guardrails server is up before running tests."""
        url = f"{self.config.host}/health"
        log.debug("Checking service is up at %s", url)

        try:
            # Try a simple request to verify the server is accessible
            response = httpx.get(url, timeout=5)
        except httpx.ConnectError as e:
            raise RuntimeError(f"ConnectError accessing {url}: {e}")
        except httpx.TimeoutException as e:
            raise RuntimeError(f"HTTP Timeout accessing {url}: {e}")

        if response.is_error:
            raise RuntimeError(f"Error {response.status_code} connecting to {url}: {response.text}")

        try:
            if response.json().get("status") != "healthy":
                raise RuntimeError(f"Service at {url} is unhealthy: {response.text}")
        except json.decoder.JSONDecodeError as e:
            raise RuntimeError(f"Error: response {response.text} couldn't be parsed as JSON: {e}")

        log.info("Successfully connected to server at %s", self.config.host)

    def _build_locust_command(self, output_dir: Optional[Path] = None) -> list[str]:
        """Build the Locust command with all parameters."""
        cmd = ["locust", "-f", str(self.locustfile_path)]

        # Host
        cmd.extend(["--host", self.config.host])

        # User and spawn rate
        cmd.extend(["--users", str(self.config.users)])
        cmd.extend(["--spawn-rate", str(self.config.spawn_rate)])
        cmd.extend(["--run-time", f"{self.config.run_time}s"])

        # Headless mode
        if self.config.headless:
            cmd.append("--headless")
            cmd.append("--only-summary")  # only print last latency table

            # Add output files for headless mode
            if output_dir:
                html_file = output_dir / "report.html"
                csv_prefix = output_dir / "stats"
                cmd.extend(["--html", str(html_file)])
                cmd.extend(["--csv", str(csv_prefix)])

        log.debug("Locust command: %s", " ".join(cmd))
        return cmd

    def _save_run_metadata(self, output_dir: Path, command: list[str], start_time: datetime) -> None:
        """Save metadata about the load test run."""
        metadata = {
            "start_time": start_time.isoformat(),
            "config": self.config.model_dump(),
            "command": " ".join([str(c) for c in command]),
        }

        metadata_file = output_dir / "run_metadata.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        log.debug("Saved run metadata to %s", metadata_file)

    def _create_output_path(self, base_dir: str) -> Path:
        """Create timestamped output directory for test results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(base_dir) / Path(timestamp)
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path

    def run(self, dry_run: bool) -> int:
        """Run the Locust load test."""

        # For dry-run, print command without creating directories or metadata
        if dry_run:
            command = self._build_locust_command()
            env_vars = (
                f"LOCUST_CONFIG_ID={self.config.config_id} "
                f"LOCUST_MODEL={self.config.model} "
                f"LOCUST_MESSAGE='{self.config.message}'"
            )
            log.info("Dry run mode. Command: %s %s", env_vars, " ".join(command))
            return 0

        # Check service availability
        try:
            self._check_service()
        except RuntimeError as e:
            log.error(str(e))
            return 1

        # Build command with output directory
        output_path = self._create_output_path(self.config.output_base_dir)
        command = self._build_locust_command(output_path)

        # Save metadata
        start_time = datetime.now()
        self._save_run_metadata(output_path, command, start_time)
        log.info("Saving metadata to: %s", output_path)

        # Set environment variables for the locustfile
        env = os.environ.copy()
        env["LOCUST_CONFIG_ID"] = self.config.config_id
        env["LOCUST_MODEL"] = self.config.model
        env["LOCUST_MESSAGE"] = self.config.message

        # Log test configuration
        log.info("Starting Locust load test")
        log.info("Config: %s", self.config.model_dump_json())

        rampup_seconds = min(int(self.config.users / self.config.spawn_rate), self.config.run_time)
        steady_state_seconds = self.config.run_time - rampup_seconds
        log.info("Duration: rampup: %is, steady-state %is", rampup_seconds, steady_state_seconds)

        if not self.config.headless:
            log.info("Web UI will be available at: http://localhost:8089")

        try:
            result = subprocess.run(command, env=env, check=False)

            if result.returncode == 0:
                log.info("Load test completed successfully")
                log.info("Results saved to: %s", output_path)
            else:
                log.error("Load test failed with exit code %s", result.returncode)

            return result.returncode

        except KeyboardInterrupt:
            log.warning("Load test interrupted by user")
            return 130
        except Exception as e:
            log.error("Error running load test: %s", e)
            return 1


def _load_config_from_yaml(config_file: Path) -> LocustConfig:
    """Load and validate configuration from YAML file."""
    try:
        with open(config_file, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)

        if config_data is None:
            config_data = {}

        config = LocustConfig(**config_data)
        return config

    except FileNotFoundError:
        log.error("Configuration file not found: %s", config_file)
        sys.exit(1)
    except yaml.YAMLError as e:
        log.error("Error parsing YAML configuration: %s", e)
        sys.exit(1)
    except ValidationError as e:
        log.error("Configuration validation error:\n%s", e)
        sys.exit(1)


@app.command()
def run(
    config_file: Path = typer.Argument(
        help="Path to YAML configuration file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print commands without executing them",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        help="Print additional debugging information during run",
    ),
):
    """
    Run Locust load test using provided config file
    """
    if verbose:
        log.setLevel(logging.DEBUG)

    locust_config = _load_config_from_yaml(config_file)

    # Create and run the test
    runner = LocustRunner(locust_config)
    exit_code = runner.run(dry_run)

    raise typer.Exit(code=exit_code)


if __name__ == "__main__":
    app()
