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
        url = f"{self.config.host}/v1/chat/completions"
        log.debug("Checking service is up at %s", url)

        try:
            # Try a simple request to verify the server is accessible
            response = httpx.get(self.config.host, timeout=5)
        except httpx.ConnectError as e:
            raise RuntimeError(
                f"Can't connect to {self.config.host}: {e}\nPlease ensure the NeMo Guardrails server is running."
            )

        log.info("Successfully connected to server at %s", self.config.host)

    def _build_locust_command(self, output_dir: Optional[Path] = None) -> list[str]:
        """Build the Locust command with all parameters."""
        cmd = ["locust", "-f", str(self.locustfile_path)]

        # Host
        cmd.extend(["--host", self.config.host])

        # User and spawn rate
        cmd.extend(["--users", str(self.config.users)])
        cmd.extend(["--spawn-rate", str(self.config.spawn_rate)])

        # Run time
        if self.config.run_time:
            cmd.extend(["--run-time", f"{self.config.run_time}s"])

        # Headless mode
        if self.config.headless:
            cmd.append("--headless")

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
            "command": " ".join(command),
        }

        metadata_file = output_dir / "run_metadata.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        log.debug("Saved run metadata to %s", metadata_file)

    def _create_output_dir(self, base_dir: Path) -> Path:
        """Create timestamped output directory for test results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = base_dir / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def run(self, dry_run: bool) -> int:
        """Run the Locust load test."""
        # Check service availability
        try:
            self._check_service()
        except RuntimeError as e:
            log.error(str(e))
            return 1

        # Create output directory if in headless mode
        output_dir = None
        if self.config.headless:
            base_dir = Path("locust_results")
            output_dir = self._create_output_dir(base_dir)
            log.info("Results will be saved to: %s", output_dir)

        # Build command
        command = self._build_locust_command(output_dir)

        # Save metadata
        start_time = datetime.now()
        if output_dir:
            self._save_run_metadata(output_dir, command, start_time)

        # Set environment variables for the locustfile
        env = os.environ.copy()
        env["LOCUST_CONFIG_ID"] = self.config.config_id
        env["LOCUST_MODEL"] = self.config.model
        env["LOCUST_MESSAGE"] = self.config.message

        # Log test configuration
        log.info("Starting Locust load test")
        log.info("Host: %s", self.config.host)
        log.info("Config ID: %s", self.config.config_id)
        log.info("Model: %s", self.config.model)
        log.info("Users: %s", self.config.users)
        log.info("Spawn rate: %s users/second", self.config.spawn_rate)
        log.info("Run time: %s seconds", self.config.run_time or "unlimited")
        log.info("Mode: %s", "headless" if self.config.headless else "web UI")

        rampup_seconds = round(self.config.users / self.config.spawn_rate, 2)
        steady_state_seconds = self.config.run_time - rampup_seconds
        log.info("Duration: rampup: %fs, steady-state %fs", rampup_seconds, steady_state_seconds)

        if not self.config.headless:
            log.info("Web UI will be available at: http://localhost:8089")

        try:
            # For dry-run, just print out the command
            if dry_run:
                log.info("Dry run mode. Command: %s", " ".join(command))
                return 0

            result = subprocess.run(command, env=env, check=False)

            if result.returncode == 0:
                log.info("Load test completed successfully")
                if output_dir:
                    log.info("Results saved to: %s", output_dir)
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
    except Exception as e:
        log.error("Unexpected error loading configuration: %s", e)
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

    # Load config from file if provided
    if config_file:
        locust_config = _load_config_from_yaml(config_file)

    # Create and run the test
    runner = LocustRunner(locust_config)
    exit_code = runner.run(dry_run)

    raise typer.Exit(code=exit_code)


if __name__ == "__main__":
    app()
