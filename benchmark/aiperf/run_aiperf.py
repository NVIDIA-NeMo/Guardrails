#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import itertools
import json
import logging
import os
import subprocess
import sys
import urllib.parse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from subprocess import CompletedProcess
from typing import Any, Dict, List, Optional, Union

import httpx
import typer
import yaml
from pydantic import ValidationError

from benchmark.aiperf.aiperf_models import AIPerfConfig

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(formatter)

log.addHandler(console_handler)

app = typer.Typer()


@dataclass
class AIPerfSummary:
    total: int
    completed: int
    failed: int


class AIPerfRunner:
    """Run batches of AIPerf benchmarks using YAML config and optional parameter sweeps"""

    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> AIPerfConfig:
        """Load and validate the YAML configuration file using Pydantic."""
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f)

            # Validate with Pydantic model
            config = AIPerfConfig(**config_data)
            return config

        except FileNotFoundError:
            log.error("Configuration file not found: %s", self.config_path)
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

    def _get_sweep_combinations(self) -> Optional[List[Dict[str, Union[int, str]]]]:
        """Create cartesian-product of parameter sweep values for benchmarks"""

        if not self.config.sweeps:
            # No sweeps, return single empty combination
            return None

        # Extract parameter names and their values
        param_names = list(self.config.sweeps.keys())
        param_values = [self.config.sweeps[name] for name in param_names]

        num_runs = 1
        for _, sweep_values in self.config.sweeps.items():
            num_runs *= len(sweep_values)

        max_runs = 100
        if num_runs > max_runs:
            raise RuntimeError(f"Requested {num_runs} runs, max is {max_runs}")

        # Generate all combinations
        combinations = []
        for combination in itertools.product(*param_values):
            combinations.append(dict(zip(param_names, combination)))

        return combinations

    @staticmethod
    def _sanitize_command_for_logging(cmd: List[str]) -> str:
        """Convert command list to string with API key redacted.

        Args:
            cmd: List of command-line arguments

        Returns:
            String with --api-key value replaced with * apart from last N chars
        """
        last_n_chars = 6  # Show the last 6 characters

        sanitized = []
        i = 0
        while i < len(cmd):
            current = cmd[i]
            sanitized.append(current)

            # If this is --api-key, replace the next value with <removed>
            if current == "--api-key" and i + 1 < len(cmd):
                api_key = cmd[i + 1]
                len_api_key = len(api_key)
                sanitized_api_key = "*" * (len_api_key - last_n_chars)
                sanitized_api_key += api_key[-last_n_chars:]
                sanitized.append(sanitized_api_key)
                i += 2  # Skip the actual API key value
            else:
                i += 1

        return " ".join(sanitized)

    def _build_command(self, sweep_params: Optional[Dict[str, Union[str, int]]], output_dir: Path) -> List[str]:
        """Create a list of strings with the aiperf command and arguments to execute"""

        # Run aiperf in profile mode: `aiperf profile`
        cmd = ["aiperf", "profile"]

        # Get base config as dictionary
        base_params = self.config.base_config.model_dump()

        # Merge base config with sweep params (sweep params override base)
        params = base_params if not sweep_params else {**base_params, **sweep_params}
        log.debug("Building command-line with params: %s", params)

        # Add output directory
        params["output-artifact-dir"] = str(output_dir)

        # Use the --verbose CLI option (which changes log.level to debug) to enable more debugging
        params["ui_type"] = "simple" if log.level == logging.DEBUG else "none"

        # Convert parameters to command line arguments
        for key, value in params.items():
            # If an optional field isn't provided, don't pass that argument to aiperf
            if value is None:
                continue

            # If `api_key_env_var` is provided, get the value of the env var and add it
            # to the command
            if key == "api_key_env_var":
                api_key = os.environ.get(value)
                if not api_key:
                    raise RuntimeError(
                        f"Environment variable '{value}' is not set. Please set it: export {value}='your-api-key'"
                    )
                cmd.extend(["--api-key", str(api_key)])
                continue

            # Convert underscores to hyphens for CLI arguments
            arg_name = key.replace("_", "-")

            # Handle different value types
            if isinstance(value, bool):
                if value:
                    cmd.append(f"--{arg_name}")
            elif isinstance(value, list):
                # For list values, add multiple arguments
                for item in value:
                    cmd.extend([f"--{arg_name}", str(item)])
            elif value is not None:
                cmd.extend([f"--{arg_name}", str(value)])

        log.debug("Final command-line: %s", self._sanitize_command_for_logging(cmd))
        return cmd

    @staticmethod
    def _create_output_dir(
        base_dir: Path,
        sweep_params: Optional[Dict[str, Union[str, int]]],
    ) -> Path:
        """Create directory in which to store AIPerf outputs."""

        # Early-out if we're not sweeping anything
        if not sweep_params:
            base_dir.mkdir(parents=True, exist_ok=True)
            return base_dir

        param_parts = [f"{key}{value}" for key, value in sorted(sweep_params.items())]
        param_dir = "_".join(param_parts)

        output_dir = base_dir / param_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def _save_run_metadata(
        self,
        output_dir: Path,
        sweep_params: Optional[Dict[str, Any]],
        command: List[str],
        run_index: int,
    ):
        """Save metadata about the run for future reruns or analysis"""
        metadata = {
            "run_index": run_index,
            "timestamp": datetime.now().isoformat(),
            "config_file": str(self.config_path),
            "sweep_params": sweep_params,
            "base_config": self.config.base_config.model_dump(),
            "command": self._sanitize_command_for_logging(command),
        }

        metadata_file = output_dir / "run_metadata.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    @staticmethod
    def _save_subprocess_result_json(output_dir: Path, result: CompletedProcess) -> None:
        """Save the subprocess result to the given filename"""

        process_result_file = output_dir / "process_result.json"
        save_data = result.__dict__

        try:
            with open(process_result_file, "w", encoding="utf-8") as f:
                json.dump(save_data, f, indent=2)

        except (IOError, OSError) as e:
            log.error("Could not write %s to file %s: %s", save_data, process_result_file, e)
            raise

        except TypeError as e:
            log.error("Couldn't serialize %s to %s: %s", save_data, process_result_file, e)
            raise

    def _check_service(self, endpoint: Optional[str] = "/v1/models") -> None:
        """Check if the service is up before we run the benchmarks"""
        url = urllib.parse.urljoin(self.config.base_config.url, endpoint)
        log.debug("Checking service is up using endpoint %s", url)

        # If the user has an API Key stored in an env var, use that in the /v1/models call
        api_key_env_var = self.config.base_config.api_key_env_var
        api_key = None
        if api_key_env_var:
            api_key = os.environ.get(api_key_env_var)
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else None

        try:
            response = httpx.get(url, timeout=5, headers=headers)
        except httpx.ConnectError as e:
            raise RuntimeError(f"Can't connect to {url}: {e}")

        if response.status_code != 200:
            raise RuntimeError(f"Can't access {url}: {response}")

    def run(self, dry_run: bool = False) -> int:
        """Run benchmarks with AIPerf"""

        # Check the service is up before running anything
        self._check_service()

        # Get the directory under which all benchmarks will store results
        batch_dir = self._get_batch_dir()

        log.info("Running AIPerf with configuration: %s", self.config_path)
        log.info("Results root directory: %s", batch_dir)
        log.info("Sweeping parameters: %s", self.config.sweeps)

        benchmark_result: AIPerfSummary = (
            self.run_batch_benchmarks(batch_dir, dry_run)
            if self.config.sweeps
            else self.run_single_benchmark(batch_dir, dry_run)
        )

        # Log summary
        log.info("SUMMARY")
        log.info("Total runs : %s", benchmark_result.total)
        log.info("Completed  : %s", benchmark_result.completed)
        log.info("Failed     : %s", benchmark_result.failed)

        return 1 if benchmark_result.failed > 0 else 0

    def _get_batch_dir(self) -> Path:
        # Get base output directory
        base_output_dir = self.config.get_output_base_path()
        batch_name = self.config.batch_name

        # Create timestamped batch directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_dir = base_output_dir / batch_name / timestamp
        return batch_dir

    def run_single_benchmark(
        self,
        run_directory: Path,
        dry_run: bool,
    ) -> AIPerfSummary:
        """Run a single benchmark. Return OS exit code."""

        run_output_dir = self._create_output_dir(run_directory, sweep_params=None)

        log.info("Running AIPerf with configuration: %s", self.config_path)
        log.info("Output directory: %s", run_output_dir)

        # Build command
        command = self._build_command(sweep_params=None, output_dir=run_output_dir)

        # Save metadata
        self._save_run_metadata(run_output_dir, None, command, 0)

        log.info("Single Run")
        log.debug("Output directory: %s", run_output_dir)
        log.debug("Command: %s", self._sanitize_command_for_logging(command))
        if dry_run:
            log.info("Dry-run mode. Commands will not be executed")
            return AIPerfSummary(total=0, completed=0, failed=0)

        try:
            capture_output = log.level != logging.DEBUG
            # Execute the command
            result = subprocess.run(
                command,
                check=True,
                capture_output=capture_output,
                text=True,
            )
            log.info("Run completed successfully")
            self._save_subprocess_result_json(run_output_dir, result)
            run_completed = 1 if result.returncode == 0 else 0
            return AIPerfSummary(total=1, completed=run_completed, failed=1 - run_completed)

        except subprocess.CalledProcessError as e:
            log.error("Run failed with exit code %s", e.returncode)
            return AIPerfSummary(total=1, completed=0, failed=1)

        except KeyboardInterrupt:
            log.warning("Interrupted by user")
            raise

    def run_batch_benchmarks(
        self,
        run_directory: Path,
        dry_run: bool,
    ) -> AIPerfSummary:
        """Run a batch of benchmarks using sweeps values. Return OS exit code."""

        # Generate all sweep combinations
        combinations = self._get_sweep_combinations()
        if not combinations:
            raise RuntimeError(f"Can't generate sweep combinations from {self.config.sweeps}")

        num_combinations = len(combinations)
        log.info("Running %s benchmarks", num_combinations)

        # Early-out if it's a dry-run
        if dry_run:
            log.info("Dry-run mode. Commands will not be executed")
            return AIPerfSummary(total=0, completed=0, failed=0)

        # If logging isn't set to DEBUG, we'll capture the AIPerf stdout and stderr to a file
        capture_output = log.level != logging.DEBUG

        # Execute each combination
        failed_runs = 0

        # Iterate over the sweep combinations, saving out results in separate directories
        for i, sweep_params in enumerate(combinations):
            run_num = i + 1  # 1-indexed for run status printouts

            # Create output directory for this run
            run_output_dir = self._create_output_dir(run_directory, sweep_params)

            # Create the command-line for this sweep param
            command = self._build_command(sweep_params, run_output_dir)

            # Save metadata to reproduce benchmark results later if needed
            self._save_run_metadata(run_output_dir, sweep_params, command, i)

            log.info("Run %s/%s", run_num, num_combinations)
            log.info("Sweep parameters: %s", sweep_params)
            log.debug("Output directory: %s", run_output_dir)
            log.debug("Command: %s", " ".join(command))

            try:
                # Execute the command
                result = subprocess.run(
                    command,
                    check=True,
                    capture_output=capture_output,
                    text=True,
                )
                log.info("Run %s completed successfully", run_num)

                self._save_subprocess_result_json(run_output_dir, result)
                if result.returncode != 0:
                    failed_runs += 1

            except subprocess.CalledProcessError as e:
                log.error(
                    "Run %s with sweep params %s failed with exit code %s",
                    i,
                    sweep_params,
                    e.returncode,
                )
                failed_runs += 1

            except KeyboardInterrupt:
                log.warning("Interrupted by user")
                raise

        return AIPerfSummary(
            total=num_combinations,
            completed=num_combinations - failed_runs,
            failed=failed_runs,
        )


# Create typer app
app = typer.Typer(
    help="AIPerf application to run, analyze, and compare benchmarks",
    add_completion=False,
)


@app.command()
def run(
    config_file: Path = typer.Option(
        ...,
        "--config-file",
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
    """Run AIPerf benchmark using the provided YAML config file"""

    if verbose:
        log.setLevel(logging.DEBUG)

    # Create and run the benchmark runner
    runner = AIPerfRunner(config_file)
    exit_code = runner.run(dry_run=dry_run)

    raise typer.Exit(code=exit_code)


if __name__ == "__main__":
    app()
