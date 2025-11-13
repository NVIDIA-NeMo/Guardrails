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
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from subprocess import CompletedProcess
from typing import Any, Dict, List, Optional, Union

import typer
import yaml
from aiperf_models import AIPerfConfig
from pydantic import ValidationError

# Set up logging
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)  # Set the lowest level to capture all messages

formatter = logging.Formatter(
    "%(asctime)s %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)  # DEBUG and higher will go to the console
console_handler.setFormatter(formatter)

# Add the console handler for logging
log.addHandler(console_handler)


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

    def _get_sweep_combinations(self) -> List[Dict[str, Any]]:
        """Create cartesian-product of parameter sweep values for benchmarks"""

        if not self.config.sweeps:
            # No sweeps, return single empty combination
            return [{}]

        # Extract parameter names and their values
        param_names = list(self.config.sweeps.keys())
        param_values = [self.config.sweeps[name] for name in param_names]

        # Generate all combinations
        combinations = []
        for combination in itertools.product(*param_values):
            combinations.append(dict(zip(param_names, combination)))

        return combinations

    def _build_command(
        self, sweep_params: Optional[Dict[str, Union[str, int]]], output_dir: Path
    ) -> List[str]:
        """Build the aiperf command with given parameters."""

        # Run aiperf in profile mode: `aiperf profile`
        cmd = ["aiperf", "profile"]

        # Get base config as dictionary
        base_params = self.config.base_config.model_dump()

        # Merge base config with sweep params (sweep params override base)
        params = base_params if not sweep_params else {**base_params, **sweep_params}

        # Add output directory
        params["output-artifact-dir"] = str(output_dir)

        # Convert parameters to command line arguments
        for key, value in params.items():
            item_key = key
            # Rampup seconds is used to derive `request_rate` in the BaseConfig model, don't pass
            # it to the aiperf invocation
            # if key == "rampup_seconds":
            #     continue

            # Convert the `benchmark_seconds` in config file to `benchmark_duration` key
            if key == "benchmark_seconds":
                item_key = "benchmark_duration"

            # Convert underscores to hyphens for CLI arguments
            arg_name = item_key.replace("_", "-")

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

        return cmd

    @staticmethod
    def _create_output_dir(
        base_dir: Path,
        sweep_params: Optional[Dict[str, Union[str, int]]],
    ) -> Path:
        """Create directory in which to place AIPerf outputs."""

        # Early-out if we're not sweeping anything
        if not sweep_params:
            return base_dir

        param_parts = [f"{key}{value}" for key, value in sorted(sweep_params.items())]
        param_dir = "_".join(param_parts)

        output_dir = base_dir / param_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def _save_run_metadata(
        self,
        output_dir: Path,
        sweep_params: Dict[str, Any],
        command: List[str],
        run_index: int,
    ):
        """
        Save metadata about this run for reference.

        Args:
            output_dir: Directory where results are stored
            sweep_params: Parameters for this run
            command: Full command that was executed
            run_index: Index of this run
        """
        metadata = {
            "run_index": run_index,
            "timestamp": datetime.now().isoformat(),
            "config_file": str(self.config_path),
            "sweep_params": sweep_params,
            "base_config": self.config.base_config.model_dump(),
            "command": " ".join(command),
        }

        metadata_file = output_dir / "run_metadata.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    def _save_subprocess_result_json(
        self, output_dir: Path, result: CompletedProcess
    ) -> None:
        """Save the subprocess result to the given filename"""

        process_result_file = output_dir / "process_result.json"
        with open(process_result_file, "w", encoding="utf-8") as f:
            json.dump(result.__dict__, f, indent=2)

    def run(self, dry_run: bool = False) -> int:
        """
        Run benchmarks with AIPerf
        """
        # Get base output directory
        base_output_dir = self.config.get_output_base_path()

        # Create timestamped batch directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_name = self.config.batch_name
        batch_dir = base_output_dir / batch_name / timestamp

        # Generate all sweep combinations
        combinations = self._get_sweep_combinations()

        log.info("Running AIPerf with configuration: %s", self.config_path)
        log.info("Batch directory: %s", batch_dir)
        log.info("Sweep parameters: %s", combinations)
        log.info("Number of runs: %s", len(combinations))

        if dry_run:
            log.info("DRY RUN MODE - Commands will not be executed")

        # Execute each combination
        failed_runs = []

        for i, sweep_params in enumerate(combinations):
            run_num = i + 1
            # Create output directory for this run
            run_output_dir = self._create_output_dir(batch_dir, sweep_params)

            # Build command
            command = self._build_command(sweep_params, run_output_dir)

            # Save metadata
            self._save_run_metadata(run_output_dir, sweep_params, command, i)

            log.info("Run %s/%s", run_num, len(combinations))
            log.info(
                "Parameters: %s", sweep_params if sweep_params else "base config only"
            )
            log.debug("Output directory: %s", run_output_dir)
            log.debug("Command: %s", " ".join(command))

            if not dry_run:
                try:
                    # Execute the command
                    result = subprocess.run(
                        command, check=True, capture_output=True, text=True
                    )
                    log.info("Run %s completed successfully", run_num)

                    self._save_subprocess_result_json(run_output_dir, result)

                except subprocess.CalledProcessError as e:
                    log.error("Run %s failed with exit code %s", i, e.returncode)
                    failed_runs.append((i, sweep_params))
                except KeyboardInterrupt:
                    log.warning("Interrupted by user")
                    return 130

        # Log summary
        log.info("SUMMARY")
        log.info("Total runs: %s", len(combinations))
        log.info("Successful: %s", len(combinations) - len(failed_runs))
        log.info("Failed: %s", len(failed_runs))

        if failed_runs:
            log.warning("Failed runs:")
            for run_index, params in failed_runs:
                log.warning("  - Run %s: %s", run_index, params)

        log.info("Results stored in: %s", batch_dir)

        return 1 if failed_runs else 0


# Create typer app
app = typer.Typer(
    help="Run aiperf benchmarks with configurable parameters and sweeps",
    add_completion=False,
)


@app.command()
def main(
    config_file: Path = typer.Argument(
        ...,
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
    no_progress: bool = typer.Option(
        False,
        "--no-progress",
        help="Disable progress bar display",
    ),
):
    """
    Run aiperf benchmarks with configurable parameters and sweeps.

    Example configuration file (config.yaml):

      batch_name: my_benchmark
      output_base_dir: ./benchmark_results

      base_config:
        model-names: gpt-3.5-turbo
        url: localhost:8000
        request-count: 100
        random-seed: 42
        prompt-input-tokens-mean: 100
        prompt-input-tokens-stddev: 10
        prompt-output-tokens-mean: 50
        prompt-output-tokens-stddev: 5

      sweeps:
        request-rate: [10, 20, 50]
        concurrency: [1, 5, 10]

    This configuration will run 9 benchmark tests (3 request rates × 3 concurrency levels).
    """
    # Create and run the benchmark runner
    runner = AIPerfRunner(config_file)
    exit_code = runner.run(dry_run=dry_run)

    raise typer.Exit(code=exit_code)


if __name__ == "__main__":
    app()
