# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
Tests for AIPerf run_aiperf module.
"""

import json
import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import Mock, patch

import httpx
import pytest
import yaml
from typer.testing import CliRunner

from benchmark.aiperf.aiperf_models import AIPerfConfig, BaseConfig
from benchmark.aiperf.run_aiperf import AIPerfRunner, AIPerfSummary


@pytest.fixture
def create_config_data():
    """Returns a function with sample basic config, and allows mutation of fields to cover
    more cases or add extra fields"""

    def _create_config(
        batch_name="test_batch",
        output_base_dir="test_output",
        model="test-model",
        tokenizer="test-tokenizer",
        url="http://localhost:8000",
        warmup_request_count=10,
        benchmark_duration=60,
        concurrency=5,
        sweeps=None,
        **extra_base_config,
    ):
        base_config = {
            "model": model,
            "tokenizer": tokenizer,
            "url": url,
            "warmup_request_count": warmup_request_count,
            "benchmark_duration": benchmark_duration,
            "concurrency": concurrency,
        }

        config_data = {
            "batch_name": batch_name,
            "output_base_dir": output_base_dir,
            "base_config": base_config,
        }

        # Add sweeps if provided
        if sweeps:
            config_data["sweeps"] = sweeps

        # Merge any extra base_config parameters
        if extra_base_config:
            base_config.update(extra_base_config)

        return config_data

    return _create_config


@pytest.fixture
def create_config_file(tmp_path, create_config_data):
    """Fixture to write config data to a file and return the path."""

    def _write_config_file(
        extra_base_config: Optional[Dict[str, Any]] = None,
        filename: Optional[str] = "config.yml",
        sweeps: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Apply extra base config to config data, write to file and return the path."""

        # Unpack extra_base_config as kwargs if provided
        if extra_base_config:
            config_data = create_config_data(sweeps=sweeps, **extra_base_config)
        else:
            config_data = create_config_data(sweeps=sweeps)

        config_file = tmp_path / filename
        config_file.write_text(yaml.dump(config_data))
        return config_file

    return _write_config_file


class TestAIPerfSummary:
    """Test the AIPerfSummary dataclass."""

    def test_aiperf_summary_creation(self):
        """Test creating an AIPerfSummary instance."""
        summary = AIPerfSummary(total=10, completed=8, failed=2)
        assert summary.total == 10
        assert summary.completed == 8
        assert summary.failed == 2


class TestAIPerfRunnerInit:
    """Test AIPerfRunner initialization and config loading."""

    def test_init_with_valid_config(self, create_config_file):
        """Test initialization with a valid config file."""
        config_file = create_config_file()

        runner = AIPerfRunner(config_file)

        assert runner.config_path == config_file
        assert isinstance(runner.config, AIPerfConfig)
        assert runner.config.batch_name == "test_batch"
        assert runner.config.output_base_dir == "test_output"
        assert runner.config.base_config.model == "test-model"
        assert runner.config.base_config.tokenizer == "test-tokenizer"
        assert runner.config.base_config.url == "http://localhost:8000"
        assert runner.config.base_config.warmup_request_count == 10
        assert runner.config.base_config.benchmark_duration == 60
        assert runner.config.base_config.concurrency == 5
        assert runner.config.sweeps is None

    def test_init_with_nonexistent_config(self, tmp_path):
        """Test initialization with a nonexistent config file."""
        config_file = tmp_path / "nonexistent.yaml"

        with pytest.raises(SystemExit):
            AIPerfRunner(config_file)

    def test_init_with_invalid_yaml(self, tmp_path):
        """Test initialization with invalid YAML syntax."""
        config_file = tmp_path / "invalid.yaml"
        config_file.write_text("invalid: yaml: syntax: [")

        with pytest.raises(SystemExit):
            AIPerfRunner(config_file)

    def test_init_with_validation_error(self, tmp_path):
        """Test initialization with config that fails Pydantic validation."""

        config_file = tmp_path / "invalid.yaml"
        config_file.write_text(
            yaml.dump(
                {
                    "batch_name": "test_batch",
                    "base_config": {
                        "model": "test-model",
                        # Missing required fields
                    },
                }
            )
        )

        with pytest.raises(SystemExit):
            AIPerfRunner(config_file)

    def test_init_with_unexpected_error(self, create_config_file):
        """Test initialization with an unexpected error."""
        config_file = create_config_file()

        # Mock yaml.safe_load to raise an unexpected exception
        with patch("yaml.safe_load", side_effect=RuntimeError("Unexpected error")):
            with pytest.raises(SystemExit):
                AIPerfRunner(config_file)


class TestGetSweepCombinations:
    """Test the _get_sweep_combinations method."""

    def test_no_sweeps_returns_none(self, create_config_file):
        """Test that no sweeps returns None."""
        config_file = create_config_file()

        runner = AIPerfRunner(config_file)
        combinations = runner._get_sweep_combinations()

        assert combinations is None

    def test_single_sweep_parameter(self, create_config_file):
        """Test sweep with a single parameter."""
        config_file = create_config_file(sweeps={"concurrency": [1, 2, 4]})

        runner = AIPerfRunner(config_file)
        combinations = runner._get_sweep_combinations()

        assert len(combinations) == 3
        assert combinations == [
            {"concurrency": 1},
            {"concurrency": 2},
            {"concurrency": 4},
        ]

    def test_multiple_sweep_parameters(self, create_config_file):
        """Test sweep with multiple parameters (Cartesian product)."""
        config_file = create_config_file(
            sweeps={
                "concurrency": [1, 2],
                "benchmark_duration": [30, 60],
            }
        )

        runner = AIPerfRunner(config_file)
        combinations = runner._get_sweep_combinations()

        assert len(combinations) == 4
        assert {"concurrency": 1, "benchmark_duration": 30} in combinations
        assert {"concurrency": 1, "benchmark_duration": 60} in combinations
        assert {"concurrency": 2, "benchmark_duration": 30} in combinations
        assert {"concurrency": 2, "benchmark_duration": 60} in combinations

    def test_too_many_runs_raises(self, create_config_file):
        """Test sweeps with more than 100 runs to make sure Excpetion is raised"""

        # Create a config with two parameter sweeps of 100 each
        # This has a total of 10,000, greater than 100 limit
        config_file = create_config_file(
            sweeps={
                "concurrency": list(range(100)),
                "benchmark_duration": list(range(100)),
            }
        )

        runner = AIPerfRunner(config_file)
        with pytest.raises(RuntimeError, match="Requested 10000 runs, max is 100"):
            _ = runner._get_sweep_combinations()


class TestSanitizeCommandForLogging:
    """Test the _sanitize_command_for_logging static method."""

    def test_sanitize_command_with_api_key(self):
        """Test sanitizing command with API key showing last 6 chars."""
        cmd = [
            "aiperf",
            "profile",
            "--model",
            "test-model",
            "--api-key",
            "secret-key-123",
            "--url",
            "http://localhost:8000",
        ]
        result = AIPerfRunner._sanitize_command_for_logging(cmd)

        # "secret-key-123" has 14 chars, so 8 asterisks + last 6 chars "ey-123"
        assert result == ("aiperf profile --model test-model --api-key ********ey-123 --url http://localhost:8000")

    def test_sanitize_command_without_api_key(self):
        """Test sanitizing command without API key."""
        cmd = [
            "aiperf",
            "profile",
            "--model",
            "test-model",
            "--url",
            "http://localhost:8000",
        ]
        result = AIPerfRunner._sanitize_command_for_logging(cmd)

        assert result == " ".join(cmd)

    def test_sanitize_command_api_key_at_end_no_value(self):
        """Test sanitizing command where --api-key is at the end with no value."""
        cmd = ["aiperf", "profile", "--model", "test-model", "--api-key"]
        result = AIPerfRunner._sanitize_command_for_logging(cmd)

        # Should just include --api-key without sanitizing since there's no value
        assert result == "aiperf profile --model test-model --api-key"

    def test_sanitize_command_empty_list(self):
        """Test sanitizing an empty command list."""
        cmd = []
        result = AIPerfRunner._sanitize_command_for_logging(cmd)

        assert result == ""

    def test_sanitize_command_single_element(self):
        """Test sanitizing command with a single element."""
        cmd = ["aiperf"]
        result = AIPerfRunner._sanitize_command_for_logging(cmd)

        assert result == "aiperf"

    def test_sanitize_command_multiple_api_keys(self):
        """Test sanitizing command with multiple API key occurrences."""
        cmd = [
            "aiperf",
            "profile",
            "--api-key",
            "first-key",
            "--model",
            "test-model",
            "--api-key",
            "second-key",
        ]
        result = AIPerfRunner._sanitize_command_for_logging(cmd)
        assert result == ("aiperf profile --api-key ***st-key --model test-model --api-key ****nd-key")

    def test_sanitize_command_preserves_other_values(self):
        """Test that other command values are preserved exactly."""
        cmd = [
            "aiperf",
            "profile",
            "--api-key",
            "my-secret-key",
            "--concurrency",
            "10",
            "--benchmark-duration",
            "60",
            "--streaming",
        ]
        result = AIPerfRunner._sanitize_command_for_logging(cmd)

        # "my-secret-key" has 13 chars, so 7 asterisks + "et-key" (last 6 chars)
        assert result == ("aiperf profile --api-key *******et-key --concurrency 10 --benchmark-duration 60 --streaming")

    def test_sanitize_command_short_api_key(self):
        """Test sanitizing command with API key shorter than or equal to 6 chars."""
        cmd = ["aiperf", "profile", "--api-key", "abc123"]
        result = AIPerfRunner._sanitize_command_for_logging(cmd)

        # "abc123" has exactly 6 chars, so 0 asterisks + all 6 chars
        assert result == "aiperf profile --api-key abc123"

    def test_sanitize_command_very_short_api_key(self):
        """Test sanitizing command with API key shorter than 6 chars."""
        cmd = ["aiperf", "profile", "--api-key", "abc"]
        result = AIPerfRunner._sanitize_command_for_logging(cmd)

        # "abc" has 3 chars, so shows all of them (no asterisks due to negative masking)
        assert result == "aiperf profile --api-key abc"

    def test_sanitize_command_long_api_key(self):
        """Test sanitizing command with a long API key."""
        cmd = [
            "aiperf",
            "profile",
            "--api-key",
            "sk-proj-1234567890abcdefghijklmnopqrstuvwxyz",
        ]
        result = AIPerfRunner._sanitize_command_for_logging(cmd)

        # API key has 44 chars, so 38 asterisks + last 6 chars "uvwxyz"
        expected_masked = "*" * 38 + "uvwxyz"
        assert result == f"aiperf profile --api-key {expected_masked}"


class TestBuildCommand:
    """Test the _build_command method."""

    def test_build_command_basic(self, create_config_file, tmp_path):
        """Test building a basic command."""
        config_file = create_config_file()

        runner = AIPerfRunner(config_file)
        output_dir = tmp_path / "output"
        cmd = runner._build_command(None, output_dir)

        assert cmd[0] == "aiperf"
        assert cmd[1] == "profile"
        assert "--model" in cmd
        assert "test-model" in cmd
        assert "--url" in cmd
        assert "http://localhost:8000" in cmd
        assert "--output-artifact-dir" in cmd
        assert str(output_dir) in cmd

    def test_build_command_with_sweep_params(self, create_config_file, tmp_path):
        """Test building command with sweep parameters that override base config."""
        config_file = create_config_file()

        runner = AIPerfRunner(config_file)
        output_dir = tmp_path / "output"
        sweep_params = {"concurrency": 10, "benchmark_duration": 30}
        cmd = runner._build_command(sweep_params, output_dir)

        assert "--concurrency" in cmd
        concurrency_idx = cmd.index("--concurrency")
        assert cmd[concurrency_idx + 1] == "10"

        assert "--benchmark-duration" in cmd
        duration_idx = cmd.index("--benchmark-duration")
        assert cmd[duration_idx + 1] == "30"

    def test_build_command_with_api_key_env_var(self, create_config_file, tmp_path, monkeypatch):
        """Test building command with API key from environment variable."""
        config_file = create_config_file(extra_base_config={"api_key_env_var": "TEST_API_KEY"})

        # Set the environment variable
        monkeypatch.setenv("TEST_API_KEY", "secret-key-123")

        runner = AIPerfRunner(config_file)
        output_dir = tmp_path / "output"
        cmd = runner._build_command(None, output_dir)

        assert "--api-key" in cmd
        api_key_idx = cmd.index("--api-key")
        assert cmd[api_key_idx + 1] == "secret-key-123"

    def test_build_command_with_missing_api_key_env_var(self, create_config_file, tmp_path):
        """Test building command when API key environment variable is not set."""
        config_file = create_config_file(extra_base_config={"api_key_env_var": "MISSING_API_KEY"})

        runner = AIPerfRunner(config_file)
        output_dir = tmp_path / "output"

        with pytest.raises(
            RuntimeError,
            match="Environment variable 'MISSING_API_KEY' is not set. Please set it: export MISSING_API_KEY='your-api-key'",
        ):
            runner._build_command(None, output_dir)

    def test_build_command_with_streaming_true(self, create_config_file, tmp_path):
        """Test building command with streaming enabled"""
        config_file = create_config_file(extra_base_config={"streaming": True})

        runner = AIPerfRunner(config_file)
        output_dir = tmp_path / "output"
        cmd = runner._build_command(None, output_dir)

        assert "--streaming" in cmd

    def test_build_command_with_streaming_false(self, create_config_file, tmp_path):
        """Test building command with boolean False value (should not be in command)."""
        config_file = create_config_file(extra_base_config={"streaming": False})

        runner = AIPerfRunner(config_file)
        output_dir = tmp_path / "output"
        cmd = runner._build_command(None, output_dir)

        assert "--streaming" not in cmd

    def test_build_command_default_streaming(self, create_config_file, tmp_path):
        """Test building command with streaming default of False"""
        config_file = create_config_file()

        runner = AIPerfRunner(config_file)
        output_dir = tmp_path / "output"
        cmd = runner._build_command(None, output_dir)

        assert "--streaming" not in cmd

    def test_build_command_default_api_key(self, create_config_file, tmp_path):
        """Test building command with None values (should be skipped)."""
        config_file = create_config_file()

        runner = AIPerfRunner(config_file)
        output_dir = tmp_path / "output"
        cmd = runner._build_command(None, output_dir)

        # Optional fields with None should not appear
        assert "--api-key-env-var" not in cmd

    def test_build_command_ui_type_debug(self, create_config_file, tmp_path):
        """Test that ui_type is 'simple' when log level is DEBUG."""
        config_file = create_config_file()

        runner = AIPerfRunner(config_file)
        output_dir = tmp_path / "output"

        # Patch log.level to be DEBUG
        with patch("benchmark.aiperf.run_aiperf.log.level", logging.DEBUG):
            cmd = runner._build_command(None, output_dir)

            assert "--ui-type" in cmd
            ui_type_idx = cmd.index("--ui-type")
            assert cmd[ui_type_idx + 1] == "simple"

    def test_build_command_ui_type_non_debug(self, create_config_file, tmp_path):
        """Test that ui_type is 'none' when log level is not DEBUG."""
        config_file = create_config_file()

        runner = AIPerfRunner(config_file)
        output_dir = tmp_path / "output"

        # Patch log.level to be INFO
        with patch("benchmark.aiperf.run_aiperf.log.level", logging.INFO):
            cmd = runner._build_command(None, output_dir)

            assert "--ui-type" in cmd
            ui_type_idx = cmd.index("--ui-type")
            assert cmd[ui_type_idx + 1] == "none"

    def test_build_command_with_list_in_sweep_params(self, create_config_file, tmp_path):
        """Test building command when sweep params contain list values."""
        config_file = create_config_file()

        runner = AIPerfRunner(config_file)
        output_dir = tmp_path / "output"

        # Patch model_dump method at the class level to return a list value
        original_model_dump = BaseConfig.model_dump

        def mock_model_dump(self):
            result = original_model_dump(self)
            result["extra_param"] = ["value1", "value2"]
            return result

        with patch.object(BaseConfig, "model_dump", mock_model_dump):
            cmd = runner._build_command(None, output_dir)

            # List values should appear multiple times in the command
            assert "--extra-param" in cmd
            assert cmd.count("--extra-param") == 2
            value1_idx = cmd.index("value1")
            value2_idx = cmd.index("value2")
            assert value1_idx > 0
            assert value2_idx > 0


class TestCreateOutputDir:
    """Test the _create_output_dir static method."""

    def test_create_output_dir_no_sweep(self, tmp_path):
        """Test creating output directory without sweep parameters."""
        base_dir = tmp_path / "output"
        result = AIPerfRunner._create_output_dir(base_dir, None)

        assert result == base_dir
        assert result.exists()
        assert result.is_dir()

    def test_create_output_dir_with_sweep(self, tmp_path):
        """Test creating output directory with sweep parameters."""
        base_dir = tmp_path / "output"
        sweep_params = {"concurrency": 10, "benchmark_duration": 30}
        result = AIPerfRunner._create_output_dir(base_dir, sweep_params)

        # Directory should contain sweep parameter values
        assert result == base_dir / "benchmark_duration30_concurrency10"
        assert result.exists()
        assert result.is_dir()

    def test_create_output_dir_creates_parent(self, tmp_path):
        """Test that parent directories are created if they don't exist."""
        base_dir = tmp_path / "parent" / "child" / "output"
        result = AIPerfRunner._create_output_dir(base_dir, None)

        assert result.exists()
        assert result.is_dir()


class TestSaveRunMetadata:
    """Test the _save_run_metadata method."""

    def test_save_run_metadata_without_sweep(self, create_config_file, tmp_path):
        """Test saving run metadata without sweep parameters."""
        config_file = create_config_file()

        runner = AIPerfRunner(config_file)
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        command = ["aiperf", "profile", "--model", "test-model"]
        runner._save_run_metadata(output_dir, None, command, 0)

        metadata_file = output_dir / "run_metadata.json"
        assert metadata_file.exists()

        with open(metadata_file) as f:
            metadata = json.load(f)

        assert metadata["run_index"] == 0
        assert metadata["config_file"] == str(config_file)
        assert metadata["sweep_params"] is None
        assert metadata["command"] == " ".join(command)
        assert "timestamp" in metadata
        assert "base_config" in metadata

    def test_save_run_metadata_with_sweep(self, create_config_file, tmp_path):
        """Test saving run metadata with sweep parameters."""
        config_file = create_config_file()

        runner = AIPerfRunner(config_file)
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        sweep_params = {"concurrency": 10}
        command = ["aiperf", "profile", "--concurrency", "10"]
        runner._save_run_metadata(output_dir, sweep_params, command, 1)

        metadata_file = output_dir / "run_metadata.json"
        assert metadata_file.exists()

        with open(metadata_file) as f:
            metadata = json.load(f)

        assert metadata["run_index"] == 1
        assert metadata["sweep_params"] == sweep_params


class TestSaveSubprocessResultJson:
    """Test the _save_subprocess_result_json static method."""

    def test_save_subprocess_result_success(self, tmp_path):
        """Test saving successful subprocess result."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Create a mock CompletedProcess
        result = subprocess.CompletedProcess(
            args=["aiperf", "profile"],
            returncode=0,
            stdout="Success output",
            stderr="",
        )

        AIPerfRunner._save_subprocess_result_json(output_dir, result)

        process_result_file = output_dir / "process_result.json"
        assert process_result_file.exists()

        with open(process_result_file) as f:
            saved_data = json.load(f)

        assert saved_data["returncode"] == 0
        assert saved_data["stdout"] == "Success output"

    def test_save_subprocess_result_failure(self, tmp_path):
        """Test saving failed subprocess result."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        result = subprocess.CompletedProcess(
            args=["aiperf", "profile"],
            returncode=1,
            stdout="",
            stderr="Error message",
        )

        AIPerfRunner._save_subprocess_result_json(output_dir, result)

        process_result_file = output_dir / "process_result.json"
        assert process_result_file.exists()

        with open(process_result_file) as f:
            saved_data = json.load(f)

        assert saved_data["returncode"] == 1
        assert saved_data["stderr"] == "Error message"

    def test_save_subprocess_result_io_error(self, tmp_path):
        """Test saving subprocess result when IOError occurs."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        result = subprocess.CompletedProcess(
            args=["aiperf", "profile"],
            returncode=0,
            stdout="Success",
            stderr="",
        )

        # Mock open to raise IOError
        with patch("builtins.open", side_effect=IOError("Disk full")):
            with pytest.raises(IOError):
                AIPerfRunner._save_subprocess_result_json(output_dir, result)

    def test_save_subprocess_result_type_error(self, tmp_path):
        """Test saving subprocess result when TypeError occurs during serialization."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        result = subprocess.CompletedProcess(
            args=["aiperf", "profile"],
            returncode=0,
            stdout="Success",
            stderr="",
        )

        # Mock json.dump to raise TypeError
        with patch("json.dump", side_effect=TypeError("Cannot serialize")):
            with pytest.raises(TypeError):
                AIPerfRunner._save_subprocess_result_json(output_dir, result)


class TestCheckService:
    """Test the _check_service method."""

    def test_check_service_success(self, create_config_file):
        """Test checking service when it's available."""
        config_file = create_config_file()

        runner = AIPerfRunner(config_file)

        # Mock httpx.get to return success
        with patch("httpx.get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_get.return_value = mock_response

            # Should not raise any exception
            runner._check_service()

    def test_check_service_connect_error(self, create_config_file):
        """Test checking service when connection fails."""
        config_file = create_config_file()

        runner = AIPerfRunner(config_file)

        # Mock httpx.get to raise ConnectError
        with patch("httpx.get", side_effect=httpx.ConnectError("Connection refused")):
            with pytest.raises(RuntimeError, match="Can't connect to"):
                runner._check_service()

    def test_check_service_non_200_response(self, create_config_file):
        """Test checking service when it returns non-200 status."""
        config_file = create_config_file()

        runner = AIPerfRunner(config_file)

        # Mock httpx.get to return 404
        with patch("httpx.get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 404
            mock_get.return_value = mock_response

            with pytest.raises(RuntimeError, match="Can't access"):
                runner._check_service()

    def test_check_service_custom_endpoint(self, create_config_file):
        """Test checking service with custom endpoint."""
        config_file = create_config_file()

        runner = AIPerfRunner(config_file)

        # Mock httpx.get
        with patch("httpx.get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_get.return_value = mock_response

            runner._check_service("/custom/endpoint")

            # Verify the URL was constructed correctly
            mock_get.assert_called_once()
            call_args = mock_get.call_args
            assert "/custom/endpoint" in call_args[0][0]

    def test_check_service_no_api_key_env_var(self, create_config_file):
        """Test checking service when api_key_env_var is not configured (None)."""
        config_file = create_config_file()

        runner = AIPerfRunner(config_file)

        # Mock httpx.get
        with patch("httpx.get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_get.return_value = mock_response

            runner._check_service()

            # Verify headers=None was passed
            mock_get.assert_called_once()
            call_args = mock_get.call_args
            assert call_args[1]["headers"] is None

    def test_check_service_api_key_env_var_not_set(self, create_config_file):
        """Test checking service when api_key_env_var is configured but env var doesn't exist."""
        config_file = create_config_file(extra_base_config={"api_key_env_var": "NONEXISTENT_API_KEY"})

        runner = AIPerfRunner(config_file)

        # Mock httpx.get
        with patch("httpx.get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_get.return_value = mock_response

            runner._check_service()

            # Verify headers=None was passed (since env var doesn't exist)
            mock_get.assert_called_once()
            call_args = mock_get.call_args
            assert call_args[1]["headers"] is None

    def test_check_service_api_key_env_var_set(self, create_config_file, monkeypatch):
        """Test checking service when api_key_env_var is configured and env var exists."""
        config_file = create_config_file(extra_base_config={"api_key_env_var": "TEST_API_KEY"})

        # Set the environment variable
        monkeypatch.setenv("TEST_API_KEY", "test-secret-key-123")

        runner = AIPerfRunner(config_file)

        # Mock httpx.get
        with patch("httpx.get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_get.return_value = mock_response

            runner._check_service()

            # Verify headers with Authorization Bearer token was passed
            mock_get.assert_called_once()
            call_args = mock_get.call_args
            assert call_args[1]["headers"] is not None
            assert call_args[1]["headers"]["Authorization"] == "Bearer test-secret-key-123"


class TestGetBatchDir:
    """Test the _get_batch_dir method."""

    def test_get_batch_dir(self, create_config_file, tmp_path):
        """Test getting the batch directory with timestamp."""
        config_file = create_config_file(
            extra_base_config={
                "batch_name": "test_batch",
                "output_base_dir": str(tmp_path / "output"),
            }
        )

        runner = AIPerfRunner(config_file)
        batch_dir = runner._get_batch_dir()

        # Check that the path contains the expected components
        assert "test_batch" in str(batch_dir)
        assert str(tmp_path / "output") in str(batch_dir)
        # Check that there's a timestamp-like pattern (YYYYMMDD_HHMMSS)
        assert len(batch_dir.name) == 15  # Timestamp format


class TestRunSingleBenchmark:
    """Test the run_single_benchmark method."""

    def test_run_single_benchmark_success(self, create_config_file, tmp_path):
        """Test running a single benchmark successfully."""
        config_file = create_config_file()

        runner = AIPerfRunner(config_file)
        run_directory = tmp_path / "runs"

        # Mock subprocess.run to return success
        with patch("subprocess.run") as mock_run:
            mock_result = subprocess.CompletedProcess(
                args=["aiperf", "profile"],
                returncode=0,
                stdout="Success",
                stderr="",
            )
            mock_run.return_value = mock_result

            summary = runner.run_single_benchmark(run_directory, dry_run=False)

            assert summary.total == 1
            assert summary.completed == 1
            assert summary.failed == 0

    def test_run_single_benchmark_dry_run(self, create_config_file, tmp_path):
        """Test running a single benchmark in dry-run mode."""
        config_file = create_config_file()

        runner = AIPerfRunner(config_file)
        run_directory = tmp_path / "runs"

        summary = runner.run_single_benchmark(run_directory, dry_run=True)

        assert summary.total == 0
        assert summary.completed == 0
        assert summary.failed == 0

    def test_run_single_benchmark_failure(self, create_config_file, tmp_path):
        """Test running a single benchmark that fails."""
        config_file = create_config_file()

        runner = AIPerfRunner(config_file)
        run_directory = tmp_path / "runs"

        # Mock subprocess.run to raise CalledProcessError
        with patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "aiperf")):
            summary = runner.run_single_benchmark(run_directory, dry_run=False)

            assert summary.total == 1
            assert summary.completed == 0
            assert summary.failed == 1

    def test_run_single_benchmark_keyboard_interrupt(self, create_config_file, tmp_path):
        """Test that KeyboardInterrupt is re-raised."""
        config_file = create_config_file()

        runner = AIPerfRunner(config_file)
        run_directory = tmp_path / "runs"

        # Mock subprocess.run to raise KeyboardInterrupt
        with patch("subprocess.run", side_effect=KeyboardInterrupt):
            with pytest.raises(KeyboardInterrupt):
                runner.run_single_benchmark(run_directory, dry_run=False)


class TestRunBatchBenchmarks:
    """Test the run_batch_benchmarks method."""

    def test_run_batch_benchmarks_success(self, create_config_file, tmp_path):
        """Test running batch benchmarks successfully."""
        config_file = create_config_file(sweeps={"concurrency": [1, 2]})

        runner = AIPerfRunner(config_file)
        run_directory = tmp_path / "runs"

        # Mock subprocess.run to return success first, then failure
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = [
                subprocess.CompletedProcess(
                    args=["aiperf", "profile"],
                    returncode=0,
                    stdout="Success",
                    stderr="",
                ),
                subprocess.CompletedProcess(
                    args=["aiperf", "profile"],
                    returncode=1,
                    stdout="",
                    stderr="Error",
                ),
            ]

            summary = runner.run_batch_benchmarks(run_directory, dry_run=False)

            assert summary.total == 2
            assert summary.completed == 1
            assert summary.failed == 1
            assert mock_run.call_count == 2

    def test_run_batch_benchmarks_dry_run(self, create_config_file, tmp_path):
        """Test running batch benchmarks in dry-run mode."""
        config_file = create_config_file(sweeps={"concurrency": [1, 2]})

        runner = AIPerfRunner(config_file)
        run_directory = tmp_path / "runs"

        summary = runner.run_batch_benchmarks(run_directory, dry_run=True)

        assert summary.total == 0
        assert summary.completed == 0
        assert summary.failed == 0

    def test_run_batch_benchmarks_partial_failure(self, create_config_file, tmp_path):
        """Test running batch benchmarks with some failures."""
        config_file = create_config_file(sweeps={"concurrency": [1, 2, 4]})

        runner = AIPerfRunner(config_file)
        run_directory = tmp_path / "runs"

        # Mock subprocess.run to fail on second call
        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise subprocess.CalledProcessError(1, "aiperf")
            return subprocess.CompletedProcess(
                args=["aiperf", "profile"],
                returncode=0,
                stdout="Success",
                stderr="",
            )

        with patch("subprocess.run", side_effect=side_effect):
            summary = runner.run_batch_benchmarks(run_directory, dry_run=False)

            assert summary.total == 3
            assert summary.completed == 2
            assert summary.failed == 1

    def test_run_batch_benchmarks_no_combinations(self, create_config_file, tmp_path):
        """Test running batch benchmarks with no sweep combinations raises error."""
        config_file = create_config_file()

        runner = AIPerfRunner(config_file)
        # Manually set sweeps to empty dict to trigger error
        runner.config.sweeps = {}

        run_directory = tmp_path / "runs"

        with pytest.raises(RuntimeError, match="Can't generate sweep combinations"):
            runner.run_batch_benchmarks(run_directory, dry_run=False)

    def test_run_batch_benchmarks_keyboard_interrupt(self, create_config_file, tmp_path):
        """Test that KeyboardInterrupt is re-raised in batch benchmarks."""
        config_file = create_config_file(sweeps={"concurrency": [1, 2]})

        runner = AIPerfRunner(config_file)
        run_directory = tmp_path / "runs"

        # Mock subprocess.run to raise KeyboardInterrupt on first call
        with patch("subprocess.run", side_effect=KeyboardInterrupt):
            with pytest.raises(KeyboardInterrupt):
                runner.run_batch_benchmarks(run_directory, dry_run=False)

    def test_run_batch_benchmarks_non_zero_returncode(self, create_config_file, tmp_path):
        """Test running batch benchmarks when subprocess returns non-zero but doesn't raise."""
        config_file = create_config_file(sweeps={"concurrency": [1, 2]})

        runner = AIPerfRunner(config_file)
        run_directory = tmp_path / "runs"

        # Mock subprocess.run to return non-zero returncode without raising
        with patch("subprocess.run") as mock_run:
            mock_result = subprocess.CompletedProcess(
                args=["aiperf", "profile"],
                returncode=1,  # Non-zero return code
                stdout="",
                stderr="Error",
            )
            mock_run.return_value = mock_result

            summary = runner.run_batch_benchmarks(run_directory, dry_run=False)

            assert summary.total == 2
            assert summary.completed == 0
            assert summary.failed == 2


class TestRun:
    """Test the main run method."""

    def test_run_single_benchmark(self, create_config_file):
        """Test main run method with single benchmark (no sweeps)."""
        config_file = create_config_file()
        runner = AIPerfRunner(config_file)

        # Mock _check_service and subprocess.run
        with patch.object(runner, "_check_service"):
            with patch("subprocess.run") as mock_run:
                mock_result = subprocess.CompletedProcess(
                    args=["aiperf", "profile"],
                    returncode=0,
                    stdout="Success",
                    stderr="",
                )
                mock_run.return_value = mock_result

                exit_code = runner.run(dry_run=False)

                assert exit_code == 0

    def test_run_batch_benchmarks(self, create_config_file):
        """Test main run method with batch benchmarks (with sweeps)."""
        config_file = create_config_file(sweeps={"concurrency": [1, 2]})
        runner = AIPerfRunner(config_file)

        # Mock _check_service and subprocess.run
        with patch.object(runner, "_check_service"):
            with patch("subprocess.run") as mock_run:
                mock_result = subprocess.CompletedProcess(
                    args=["aiperf", "profile"],
                    returncode=0,
                    stdout="Success",
                    stderr="",
                )
                mock_run.return_value = mock_result

                exit_code = runner.run(dry_run=False)

                assert exit_code == 0
                assert mock_run.call_count == 2

    def test_run_with_failures(self, create_config_file):
        """Test main run method returns non-zero exit code on failures."""
        config_file = create_config_file()

        runner = AIPerfRunner(config_file)

        # Mock _check_service and subprocess.run to fail
        with patch.object(runner, "_check_service"):
            with patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "aiperf")):
                exit_code = runner.run(dry_run=False)
                assert exit_code == 1

    def test_run_service_check_failure(self, create_config_file):
        """Test that service check failure raises error."""
        config_file = create_config_file()
        runner = AIPerfRunner(config_file)

        # Mock _check_service to raise error
        with patch.object(runner, "_check_service", side_effect=RuntimeError("Service unavailable")):
            with pytest.raises(RuntimeError, match="Service unavailable"):
                runner.run(dry_run=False)


class TestCLICommand:
    """Test the CLI command function."""

    def test_cli_run_command_basic(self, create_config_file):
        """Test CLI run command with basic options."""
        config_file = create_config_file()
        runner = CliRunner()

        from benchmark.aiperf.run_aiperf import app

        # Mock the runner and service check
        with patch("benchmark.aiperf.run_aiperf.AIPerfRunner") as mock_runner_class:
            mock_runner = Mock()
            mock_runner.run.return_value = 0
            mock_runner_class.return_value = mock_runner

            result = runner.invoke(app, ["--config-file", str(config_file)])

            assert result.exit_code == 0
            mock_runner.run.assert_called_once_with(dry_run=False)

    def test_cli_run_command_with_verbose(self, create_config_file):
        """Test CLI run command with verbose flag."""
        config_file = create_config_file()
        runner = CliRunner()

        from benchmark.aiperf.run_aiperf import app

        # Mock the runner and service check
        with patch("benchmark.aiperf.run_aiperf.AIPerfRunner") as mock_runner_class:
            mock_runner = Mock()
            mock_runner.run.return_value = 0
            mock_runner_class.return_value = mock_runner

            result = runner.invoke(app, ["--config-file", str(config_file), "--verbose"])

            assert result.exit_code == 0
            mock_runner.run.assert_called_once_with(dry_run=False)

    def test_cli_run_command_with_dry_run(self, create_config_file):
        """Test CLI run command with dry-run flag."""
        config_file = create_config_file()
        runner = CliRunner()

        from benchmark.aiperf.run_aiperf import app

        # Mock the runner and service check
        with patch("benchmark.aiperf.run_aiperf.AIPerfRunner") as mock_runner_class:
            mock_runner = Mock()
            mock_runner.run.return_value = 0
            mock_runner_class.return_value = mock_runner

            result = runner.invoke(app, ["--config-file", str(config_file), "--dry-run"])

            assert result.exit_code == 0
            mock_runner.run.assert_called_once_with(dry_run=True)

    def test_cli_run_command_with_failure(self, create_config_file):
        """Test CLI run command when benchmark fails."""
        config_file = create_config_file()
        runner = CliRunner()

        from benchmark.aiperf.run_aiperf import app

        # Mock the runner to return failure
        with patch("benchmark.aiperf.run_aiperf.AIPerfRunner") as mock_runner_class:
            mock_runner = Mock()
            mock_runner.run.return_value = 1  # Failure
            mock_runner_class.return_value = mock_runner

            result = runner.invoke(app, ["--config-file", str(config_file)])

            assert result.exit_code == 1
            mock_runner.run.assert_called_once_with(dry_run=False)
