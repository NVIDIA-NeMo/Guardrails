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
Tests for Locust load test CLI runner.
"""

import json
from datetime import datetime
from json.decoder import JSONDecodeError
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import Mock, patch

import httpx
import pytest
import yaml
from typer.testing import CliRunner

from benchmark.locust.locust_models import LocustConfig
from benchmark.locust.run_locust import LocustRunner, _load_config_from_yaml, app


@pytest.fixture
def create_config_data(tmp_path):
    """Returns a function with sample basic config, and allows mutation of fields to cover
    more cases or add extra fields"""

    def _create_config(
        config_id="test-config",
        model="test-model",
        host="http://localhost:8000",
        users=256,
        spawn_rate=10,
        run_time=60,
        message="Hello, what can you do?",
        headless=False,
        output_base_dir=str(tmp_path),
        **extra_config,
    ):
        config_data = {
            "host": host,
            "config_id": config_id,
            "model": model,
            "users": users,
            "spawn_rate": spawn_rate,
            "run_time": run_time,
            "message": message,
            "headless": headless,
            "output_base_dir": output_base_dir,
        }

        # Merge any extra config parameters
        if extra_config:
            config_data.update(extra_config)

        return config_data

    return _create_config


@pytest.fixture
def create_config_file(tmp_path, create_config_data):
    """Fixture to write config data to a file and return the path."""

    def _write_config_file(
        extra_base_config: Optional[Dict[str, Any]] = None,
        filename: Optional[str] = "config.yml",
    ) -> Path:
        """Apply extra base config to config data, write to file and return the path."""

        # Unpack extra_base_config as kwargs if provided
        if extra_base_config:
            config_data = create_config_data(**extra_base_config)
        else:
            config_data = create_config_data()

        config_file = tmp_path / filename
        config_file.write_text(yaml.dump(config_data))
        return config_file

    return _write_config_file


class TestLocustRunner:
    """Test LocustRunner class."""

    @pytest.fixture
    def valid_config(self, tmp_path):
        """Get a valid LocustConfig for testing."""
        return LocustConfig(
            host="http://localhost:8000",
            config_id="test-config",
            model="test-model",
            users=10,
            spawn_rate=2,
            run_time=30,
            headless=True,
            output_base_dir=str(tmp_path / "locust_results"),
        )

    @pytest.fixture
    def runner(self, valid_config):
        """Get a LocustRunner instance for testing."""
        return LocustRunner(valid_config)

    def _service_health_endpoint(self, runner: LocustRunner):
        """The endpoint used to check if the service is healthy"""
        return f"{runner.config.host}/health"

    def test_runner_init(self, valid_config):
        """Test LocustRunner initialization."""
        runner = LocustRunner(valid_config)
        assert runner.config == valid_config
        assert runner.locustfile_path.exists()
        assert runner.locustfile_path.name == "locustfile.py"

    def test_check_service_success(self, runner):
        """Test _check_service with successful connection."""
        with patch("httpx.get") as mock_get:
            mock_response = Mock()
            mock_response.is_error = False
            mock_response.json.return_value = {"status": "healthy", "timestamp": 1770675471}
            mock_get.return_value = mock_response

            # Should not raise
            runner._check_service()
            mock_get.assert_called_once_with(self._service_health_endpoint(runner), timeout=5)

    def test_check_service_connection_error(self, runner):
        """Test _check_service with httpx.ConnectError"""
        with patch("httpx.get") as mock_get:
            mock_get.side_effect = httpx.ConnectError("Connection refused")

            with pytest.raises(RuntimeError) as exc_info:
                runner._check_service()

            mock_get.assert_called_once_with(self._service_health_endpoint(runner), timeout=5)
            assert (
                exc_info.value.args[0]
                == f"ConnectError accessing {self._service_health_endpoint(runner)}: Connection refused"
            )

    def test_check_service_timeout_error(self, runner):
        """Test _check_service when httpx.get times out"""
        with patch("httpx.get") as mock_get:
            mock_get.side_effect = httpx.TimeoutException("httpx.ConnectTimeout: The connection operation timed out")

            with pytest.raises(RuntimeError) as exc_info:
                runner._check_service()

            mock_get.assert_called_once_with(self._service_health_endpoint(runner), timeout=5)
            assert (
                exc_info.value.args[0]
                == f"HTTP Timeout accessing {self._service_health_endpoint(runner)}: httpx.ConnectTimeout: The connection operation timed out"
            )

    def test_check_service_error_response(self, runner):
        """Test _check_service with non-200 response code"""
        with patch("httpx.get") as mock_get:
            mock_response = Mock()
            mock_response.is_error = True
            mock_response.status_code = 404
            mock_response.text = '{"detail":"Not Found"}'
            mock_response.json.return_value = json.loads(mock_response.text)
            mock_get.return_value = mock_response

            with pytest.raises(RuntimeError) as exc_info:
                runner._check_service()

            mock_get.assert_called_once_with(self._service_health_endpoint(runner), timeout=5)
            assert (
                exc_info.value.args[0]
                == f"Error {mock_response.status_code} connecting to {self._service_health_endpoint(runner)}: {mock_response.text}"
            )

    def test_check_service_unhealthy_response(self, runner):
        """Test _check_service with 200 response from an unhealthy service"""
        with patch("httpx.get") as mock_get:
            mock_response = Mock()
            mock_response.is_error = False
            mock_response.status_code = 200  # Successful HTTP request ..
            mock_response.text = (
                '{"status":"unhealthy","timestamp":1770677847}'  # .. but the application itself is unhealthy
            )
            mock_response.json.return_value = json.loads(mock_response.text)
            mock_get.return_value = mock_response

            with pytest.raises(RuntimeError) as exc_info:
                runner._check_service()

            mock_get.assert_called_once_with(self._service_health_endpoint(runner), timeout=5)
            assert (
                exc_info.value.args[0]
                == f"Service at {self._service_health_endpoint(runner)} is unhealthy: {mock_response.text}"
            )

    def test_check_service_invalid_json(self, runner):
        """Test _check_service with an invalid JSON response"""
        with patch("httpx.get") as mock_get:
            mock_response = Mock()
            mock_response.is_error = False
            mock_response.status_code = 200
            mock_response.text = "{'key': 'value'}"
            json_error = JSONDecodeError("Expecting property name enclosed in double quotes", "{'key': 'value'}", 1)
            mock_response.json.side_effect = json_error
            mock_get.return_value = mock_response

            with pytest.raises(RuntimeError) as exc_info:
                runner._check_service()

            mock_get.assert_called_once_with(self._service_health_endpoint(runner), timeout=5)
            assert (
                exc_info.value.args[0]
                == f"Error: response {mock_response.text} couldn't be parsed as JSON: {json_error}"
            )

    def test_build_locust_command_basic(self, runner):
        """Test building basic Locust command."""
        cmd = runner._build_locust_command()
        assert cmd[0] == "locust"

        cmd_string = " ".join(cmd)
        assert "--host http://localhost:8000 --users 10 --spawn-rate 2.0 --run-time 30s --headless" in cmd_string

    def test_build_locust_command_headless(self, runner, tmp_path):
        """Test building Locust command in headless mode."""
        runner.config.headless = True
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        cmd = runner._build_locust_command(output_dir)

        assert "--headless" in cmd
        assert "--only-summary" in cmd
        assert "--html" in cmd
        assert "--csv" in cmd

    def test_build_locust_command_non_headless(self, runner):
        """Test building Locust command in web UI mode (non-headless)."""
        runner.config.headless = False

        cmd = runner._build_locust_command()

        assert "--headless" not in cmd
        assert "--only-summary" not in cmd
        assert "--html" not in cmd
        assert "--csv" not in cmd

    def test_save_run_metadata(self, runner, tmp_path):
        """Test saving run metadata to file."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        start_time = datetime.now()
        command = ["locust", "-f", "locustfile.py"]

        runner._save_run_metadata(output_dir, command, start_time)

        metadata_file = output_dir / "run_metadata.json"
        assert metadata_file.exists()

        with open(metadata_file) as f:
            metadata = json.load(f)

        assert "start_time" in metadata
        assert "config" in metadata
        assert "command" in metadata
        assert metadata["config"]["config_id"] == "test-config"
        assert metadata["config"]["model"] == "test-model"

    def test_create_output_dir(self, runner, tmp_path):
        """Test creating timestamped output directory."""
        base_dir = str(tmp_path) + "results"

        output_dir = runner._create_output_path(base_dir)

        assert output_dir.exists()
        assert output_dir.is_dir()
        assert output_dir.parent == Path(base_dir)
        # Check that directory name looks like a timestamp
        assert len(output_dir.name) == len("20250101_120000")

    def test_run_success_headless(self, runner, tmp_path):
        """Test successful run in headless mode."""
        runner.config.headless = True

        with patch.object(runner, "_check_service"), patch("subprocess.run") as mock_run:
            mock_result = Mock()
            mock_result.returncode = 0
            mock_run.return_value = mock_result

            exit_code = runner.run(dry_run=False)

            assert exit_code == 0
            mock_run.assert_called_once()

            # Check that command was built correctly
            call_args = mock_run.call_args
            assert call_args[0][0][0] == "locust"
            assert "--headless" in call_args[0][0]

            # Check that env variables were set
            env = call_args[1]["env"]
            assert env["LOCUST_CONFIG_ID"] == "test-config"
            assert env["LOCUST_MODEL"] == "test-model"
            assert env["LOCUST_MESSAGE"] == "Hello, what can you do?"

    def test_run_success_web_ui(self, runner):
        """Test successful run in web UI mode."""
        runner.config.headless = False

        with patch.object(runner, "_check_service"), patch("subprocess.run") as mock_run:
            mock_result = Mock()
            mock_result.returncode = 0
            mock_run.return_value = mock_result

            exit_code = runner.run(dry_run=False)

            assert exit_code == 0
            mock_run.assert_called_once()

            # Check that command was built correctly
            call_args = mock_run.call_args
            assert call_args[0][0][0] == "locust"
            assert "--headless" not in call_args[0][0]

    def test_run_failure(self, runner):
        """Test run with command failure."""
        with patch.object(runner, "_check_service"), patch("subprocess.run") as mock_run:
            mock_result = Mock()
            mock_result.returncode = 1
            mock_run.return_value = mock_result

            exit_code = runner.run(dry_run=False)

            assert exit_code == 1

    def test_run_keyboard_interrupt(self, runner):
        """Test run interrupted by user."""
        with patch.object(runner, "_check_service"), patch("subprocess.run") as mock_run:
            mock_run.side_effect = KeyboardInterrupt()

            exit_code = runner.run(dry_run=False)

            assert exit_code == 130

    def test_run_service_check_failure(self, runner):
        """Test run when service check fails."""
        with patch.object(runner, "_check_service") as mock_check:
            mock_check.side_effect = RuntimeError("Service unavailable")

            exit_code = runner.run(dry_run=False)

            assert exit_code == 1

    def test_run_exception(self, runner):
        """Test run with unexpected exception."""
        with patch.object(runner, "_check_service"), patch("subprocess.run") as mock_run:
            mock_run.side_effect = Exception("Unexpected error")

            exit_code = runner.run(dry_run=False)

            assert exit_code == 1

    def test_run_dry_run(self, runner):
        """Test dry-run mode prints command without executing subprocess or checking service."""
        with patch.object(runner, "_check_service") as mock_check, patch("subprocess.run") as mock_run:
            exit_code = runner.run(dry_run=True)

            assert exit_code == 0
            mock_check.assert_not_called()
            mock_run.assert_not_called()


class TestLoadConfigFromYaml:
    """Test _load_config_from_yaml function."""

    def test_load_valid_config(self, create_config_file):
        """Test loading a valid config file."""
        config_file = create_config_file()

        config = _load_config_from_yaml(config_file)

        assert isinstance(config, LocustConfig)
        assert config.config_id == "test-config"
        assert config.model == "test-model"

    def test_load_config_file_not_found(self, tmp_path):
        """Test loading non-existent config file."""
        config_file = tmp_path / "nonexistent.yml"

        with pytest.raises(SystemExit) as exc_info:
            _load_config_from_yaml(config_file)

        assert exc_info.value.code == 1

    def test_load_config_invalid_yaml(self, tmp_path):
        """Test loading file with invalid YAML."""
        config_file = tmp_path / "invalid.yml"
        config_file.write_text("invalid: yaml: content: [")

        with pytest.raises(SystemExit) as exc_info:
            _load_config_from_yaml(config_file)

        assert exc_info.value.code == 1

    def test_load_config_validation_error(self, tmp_path):
        """Test loading file with validation errors."""
        config_file = tmp_path / "invalid.yml"
        config_data = {
            "batch_name": "test",
            "base_config": {
                "config_id": "test-config",
                # Missing required model field
            },
        }
        config_file.write_text(yaml.dump(config_data))

        with pytest.raises(SystemExit) as exc_info:
            _load_config_from_yaml(config_file)

        assert exc_info.value.code == 1

    def test_load_config_unexpected_error(self, tmp_path):
        """Test that unexpected errors propagate instead of being silently swallowed."""
        config_file = tmp_path / "config.yml"
        config_file.write_text("valid_yaml: true")

        with patch("yaml.safe_load") as mock_load:
            mock_load.side_effect = Exception("Unexpected error")

            with pytest.raises(Exception, match="Unexpected error"):
                _load_config_from_yaml(config_file)


class TestCLI:
    """Test CLI commands."""

    @pytest.fixture
    def cli_runner(self):
        """Get a Typer CLI test runner."""
        return CliRunner()

    def test_run_command_missing_config_file(self, cli_runner):
        """Test run command without required config file."""
        result = cli_runner.invoke(app, [])

        assert result.exit_code != 0  # Should fail
        # Check that error message mentions missing argument
        assert "missing" in result.stdout.lower() or result.exit_code == 2

    def test_run_command_config_file_not_found(self, cli_runner, tmp_path):
        """Test run command with non-existent config file."""
        nonexistent_file = tmp_path / "nonexistent.yaml"
        result = cli_runner.invoke(app, [str(nonexistent_file)])

        assert result.exit_code != 0  # Should fail

    def test_run_command_with_config_file(self, cli_runner, create_config_file):
        """Test run command with YAML config file."""
        config_file = create_config_file()

        with patch("benchmark.locust.run_locust.LocustRunner") as mock_runner_class:
            mock_runner = Mock()
            mock_runner.run.return_value = 0
            mock_runner_class.return_value = mock_runner

            result = cli_runner.invoke(
                app,
                [str(config_file)],
                catch_exceptions=False,
            )

            assert result.exit_code == 0, f"Output: {result.stdout}"
            mock_runner_class.assert_called_once()
            config = mock_runner_class.call_args[0][0]
            assert config.config_id == "test-config"
            assert config.model == "test-model"
            # Verify run was called with dry_run parameter
            mock_runner.run.assert_called_once_with(False)

    def test_run_command_with_dry_run(self, cli_runner, create_config_file):
        """Test run command with --dry-run CLI option."""
        config_file = create_config_file()

        with patch("benchmark.locust.run_locust.LocustRunner") as mock_runner_class:
            mock_runner = Mock()
            mock_runner.run.return_value = 0
            mock_runner_class.return_value = mock_runner

            result = cli_runner.invoke(
                app,
                [str(config_file), "--dry-run"],
                catch_exceptions=False,
            )

            assert result.exit_code == 0, f"Output: {result.stdout}"
            mock_runner.run.assert_called_once_with(True)
