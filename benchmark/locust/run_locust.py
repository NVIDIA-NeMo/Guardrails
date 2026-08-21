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
import math
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

from benchmark.locust.locust_models import LocustConfig, LocustSweepConfig

# The mock LLM servers serve `/health`; the Guardrails server serves
# `/v1/health` (and `/healthz`). Probe both so either target works.
HEALTH_PATHS = ("/health", "/v1/health")

# `/health` reports "healthy"; the Guardrails health endpoint reports "pass".
HEALTHY_STATUSES = frozenset({"healthy", "pass"})

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

    def _get(self, url: str) -> httpx.Response:
        """Issue a preflight GET, translating transport failures into RuntimeError."""
        log.debug("Checking service is up at %s", url)

        try:
            # Try a simple request to verify the server is accessible
            return httpx.get(url, timeout=5)
        except httpx.ConnectError as e:
            raise RuntimeError(f"ConnectError accessing {url}: {e}") from e
        except httpx.TimeoutException as e:
            raise RuntimeError(f"HTTP Timeout accessing {url}: {e}") from e
        except httpx.HTTPError as e:
            # Any other transport failure is still the benchmark failing to
            # start, so report it the same way rather than as a traceback.
            raise RuntimeError(f"HTTP error accessing {url}: {type(e).__name__}: {e}") from e

    def _check_service(self) -> None:
        """Check the server under test is up before running tests.

        The mock LLM servers serve `/health` and the Guardrails server serves
        `/v1/health`, so try each path and use the first one the host answers.
        """
        for path in HEALTH_PATHS:
            url = f"{self.config.host}{path}"
            response = self._get(url)

            if response.status_code == httpx.codes.NOT_FOUND:
                log.debug("No %s endpoint at %s", path, self.config.host)
                continue

            self._check_health_response(url, response)
            log.info("Successfully connected to server at %s", self.config.host)
            return

        raise RuntimeError(f"No health endpoint found at {self.config.host}: tried {', '.join(HEALTH_PATHS)}")

    def _check_health_response(self, url: str, response: httpx.Response) -> None:
        """Raise unless the health response reports a healthy service."""
        if response.is_error:
            raise RuntimeError(f"Error {response.status_code} connecting to {url}: {response.text}")

        try:
            status = response.json().get("status")
        except json.decoder.JSONDecodeError as e:
            raise RuntimeError(f"Error: response {response.text} couldn't be parsed as JSON: {e}") from e

        if status not in HEALTHY_STATUSES:
            raise RuntimeError(f"Service at {url} is unhealthy: {response.text}")

    @property
    def ramp_seconds(self) -> int:
        """Seconds Locust spends spawning users before the measured window starts."""
        return math.ceil(self.config.users / self.config.spawn_rate)

    @property
    def total_run_seconds(self) -> int:
        """Total wall-clock seconds to ask Locust for.

        ``run_time`` is the measured duration, so the ramp is added on top of it
        rather than taken out of it. Otherwise the measured window shrinks as
        concurrency rises, which is where the measurement matters most.
        """
        return self.ramp_seconds + self.config.run_time

    def _build_locust_command(self, output_dir: Optional[Path] = None) -> list[str]:
        """Build the Locust command with all parameters."""
        cmd = ["locust", "-f", str(self.locustfile_path)]

        # Host
        cmd.extend(["--host", self.config.host])

        # User and spawn rate
        cmd.extend(["--users", str(self.config.users)])
        cmd.extend(["--spawn-rate", str(self.config.spawn_rate)])
        cmd.extend(["--run-time", f"{self.total_run_seconds}s"])

        # Discard the statistics gathered while ramping, so the reported numbers
        # describe the requested concurrency rather than the climb towards it.
        cmd.append("--reset-stats")

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
                # Keep the time series, so a level that never reached a plateau
                # can be told apart from one that did.
                cmd.append("--csv-full-history")

        log.debug("Locust command: %s", " ".join(cmd))
        return cmd

    def _save_run_metadata(
        self,
        output_dir: Path,
        command: list[str],
        start_time: datetime,
        exit_code: Optional[int] = None,
    ) -> None:
        """Save metadata about the load test run.

        ``exit_code`` is None until the run finishes. A sweep treats metadata
        without an exit code as an incomplete level.
        """
        metadata = {
            "start_time": start_time.isoformat(),
            "config": self.config.model_dump(),
            "command": " ".join([str(c) for c in command]),
            "exit_code": exit_code,
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
        """Run a single Locust load test into a timestamped directory."""

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

        output_path = self._create_output_path(self.config.output_base_dir)
        return self.run_level(output_path)

    def run_level(self, output_path: Path) -> int:
        """Run Locust once, writing results to ``output_path``.

        Returns Locust's exit code, which is non-zero when any request failed.
        The service health check is not repeated here; a sweep performs it once
        for the whole batch.
        """
        output_path.mkdir(parents=True, exist_ok=True)
        command = self._build_locust_command(output_path)

        # Save metadata up front so an interrupted run still leaves a record.
        # The exit code is filled in once the run finishes.
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
        log.info(
            "Duration: rampup: %is, measured %is",
            self.ramp_seconds,
            self.config.run_time,
        )

        if not self.config.headless:
            log.info("Web UI will be available at: http://localhost:8089")

        try:
            result = subprocess.run(command, env=env, check=False)

            if result.returncode == 0:
                log.info("Load test completed successfully")
                log.info("Results saved to: %s", output_path)
            else:
                log.error("Load test failed with exit code %s", result.returncode)

            self._save_run_metadata(output_path, command, start_time, exit_code=result.returncode)
            return result.returncode

        except KeyboardInterrupt:
            log.warning("Load test interrupted by user")
            return 130
        except Exception as e:
            log.error("Error running load test: %s", e)
            return 1


class LocustSweepRunner:
    """Run every level of a sweep, keeping per-level results and exit codes.

    Each level is an independent Locust invocation. A level whose requests
    failed is a result, not a runner error: its exit code is recorded and the
    sweep continues, because the levels past saturation are the ones a stress
    test exists to measure.
    """

    def __init__(self, config: LocustSweepConfig):
        self.config = config

    @property
    def batch_path(self) -> Path:
        """Directory holding every level of this batch."""
        return Path(self.config.output_base_dir) / self.config.batch_name

    @staticmethod
    def is_complete(level_path: Path, expected_config: Optional[LocustConfig] = None) -> bool:
        """True when a level already holds a finished result for this configuration.

        Metadata is written before the run starts, so an exit code is what
        distinguishes a finished level from an interrupted one.

        A level is identified by its swept values alone, so a result can predate
        an edit to any setting that is not swept. When ``expected_config`` is
        given, a level recorded under different settings is treated as
        incomplete rather than silently mixed into the batch.
        """
        metadata_file = level_path / "run_metadata.json"
        if not metadata_file.is_file():
            return False

        try:
            metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return False

        if metadata.get("exit_code") is None:
            return False

        if expected_config is not None:
            expected = json.loads(expected_config.model_dump_json())
            if metadata.get("config") != expected:
                log.info("Configuration changed since %s was recorded; re-running it", level_path.name)
                return False

        return True

    @staticmethod
    def _hosts_to_check(levels: list[tuple[str, LocustConfig]]) -> list[LocustConfig]:
        """One config per distinct host in the batch, in the order levels run."""
        seen: dict[str, LocustConfig] = {}
        for _, level_config in levels:
            seen.setdefault(level_config.host, level_config)
        return list(seen.values())

    def run(self, dry_run: bool, resume: bool = False) -> int:
        """Run the whole sweep.

        Returns non-zero only when the benchmark could not be run, such as a
        failed health check. Request failures within a level are reported in
        that level's metadata instead.
        """
        levels = self.config.expand()
        log.info("Sweep %s: %i levels", self.config.batch_name, len(levels))

        if dry_run:
            for label, level_config in levels:
                LocustRunner(level_config).run(dry_run=True)
            return 0

        # One health check per distinct host rather than one per level. A sweep
        # over `host` targets several servers, and each must be reachable before
        # its levels run; otherwise an unreachable target looks like a level
        # whose requests merely failed.
        try:
            for level_config in self._hosts_to_check(levels):
                LocustRunner(level_config)._check_service()
        except RuntimeError as e:
            log.error(str(e))
            return 1

        exit_codes: dict[str, int] = {}
        for index, (label, level_config) in enumerate(levels, start=1):
            level_path = self.batch_path / label

            if resume and self.is_complete(level_path, level_config):
                log.info("Level %i/%i (%s) already complete, keeping it", index, len(levels), label)
                continue

            log.info("Level %i/%i: %s", index, len(levels), label)
            exit_codes[label] = LocustRunner(level_config).run_level(level_path)

        failed = sorted(label for label, code in exit_codes.items() if code != 0)
        if failed:
            log.warning(
                "Levels with request failures (recorded, not fatal): %s",
                ", ".join(failed),
            )
        log.info("Sweep complete. Results in %s", self.batch_path)
        return 0


def _load_config_from_yaml(config_file: Path) -> LocustSweepConfig:
    """Load and validate configuration from YAML file.

    Both the flat single-run format and the nested ``base_config``/``sweeps``
    batch format are accepted, so configuration files written before sweeps
    existed keep working.
    """
    try:
        with open(config_file, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)

        if config_data is None:
            config_data = {}

        config = LocustSweepConfig(**config_data)
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
    resume: bool = typer.Option(
        False,
        "--resume",
        help="Skip sweep levels that already hold a completed result",
    ),
):
    """
    Run Locust load test using provided config file
    """
    if verbose:
        log.setLevel(logging.DEBUG)

    locust_config = _load_config_from_yaml(config_file)

    # Without a sweep this is a single run, which keeps its timestamped
    # output directory rather than a per-level one.
    if not locust_config.sweeps:
        if resume:
            log.warning("--resume applies to sweeps; ignoring it for a single run")
        exit_code = LocustRunner(locust_config.base_config).run(dry_run)
    else:
        exit_code = LocustSweepRunner(locust_config).run(dry_run, resume)

    raise typer.Exit(code=exit_code)


if __name__ == "__main__":
    app()
