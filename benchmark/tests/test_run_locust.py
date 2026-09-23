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

from benchmark.locust.locust_models import LocustConfig, LocustSweepConfig
from benchmark.locust.run_locust import (
    LocustRunner,
    LocustSweepRunner,
    _load_config_from_yaml,
    app,
)


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
            "target_users": users,
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
        """Test _check_service with non-200 response code

        Uses 503 rather than 404: a 404 means the host serves no `/health` and
        falls through to the `/v1/health` probe, covered separately below.
        """
        with patch("httpx.get") as mock_get:
            mock_response = Mock()
            mock_response.is_error = True
            mock_response.status_code = 503
            mock_response.text = '{"detail":"Service Unavailable"}'
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

    def test_check_service_other_transport_error(self, runner):
        """A transport failure other than connect or timeout is still a controlled error."""
        with patch("httpx.get") as mock_get:
            mock_get.side_effect = httpx.ReadError("connection reset")

            with pytest.raises(RuntimeError, match="HTTP error accessing"):
                runner._check_service()

    def test_check_service_invalid_url(self, runner):
        """InvalidURL does not derive from HTTPError, so it needs naming explicitly."""
        with patch("httpx.get") as mock_get:
            mock_get.side_effect = httpx.InvalidURL("Invalid port: 'b:c'")

            with pytest.raises(RuntimeError, match="HTTP error accessing"):
                runner._check_service()

    def _v1_health_endpoint(self, runner: LocustRunner):
        """The Guardrails server's health endpoint"""
        return f"{runner.config.host}/v1/health"

    def _health_404_response(self):
        """A response from a host that serves no /health endpoint"""
        response = Mock()
        response.is_error = True
        response.status_code = 404
        response.text = '{"error":{"message":"Not Found"}}'
        return response

    def test_check_service_falls_back_to_v1_health(self, runner):
        """The Guardrails server serves /v1/health rather than /health"""
        with patch("httpx.get") as mock_get:
            v1_response = Mock()
            v1_response.is_error = False
            v1_response.status_code = 200
            v1_response.json.return_value = {"status": "pass"}
            mock_get.side_effect = [self._health_404_response(), v1_response]

            # Should not raise
            runner._check_service()

            assert [call.args[0] for call in mock_get.call_args_list] == [
                self._service_health_endpoint(runner),
                self._v1_health_endpoint(runner),
            ]

    def test_check_service_no_health_endpoint(self, runner):
        """A host serving neither health path is an error"""
        with patch("httpx.get") as mock_get:
            mock_get.side_effect = [self._health_404_response(), self._health_404_response()]

            with pytest.raises(RuntimeError) as exc_info:
                runner._check_service()

            assert exc_info.value.args[0] == (
                f"No health endpoint found at {runner.config.host}: tried /health, /v1/health"
            )

    def test_check_service_fallback_error_response(self, runner):
        """A non-404 error from the fallback health path is reported"""
        with patch("httpx.get") as mock_get:
            v1_response = Mock()
            v1_response.is_error = True
            v1_response.status_code = 503
            v1_response.text = '{"detail":"Service Unavailable"}'
            mock_get.side_effect = [self._health_404_response(), v1_response]

            with pytest.raises(RuntimeError) as exc_info:
                runner._check_service()

            assert (
                exc_info.value.args[0]
                == f"Error 503 connecting to {self._v1_health_endpoint(runner)}: {v1_response.text}"
            )

    def test_check_service_fallback_unhealthy(self, runner):
        """An unhealthy status from the fallback health path is reported"""
        with patch("httpx.get") as mock_get:
            v1_response = Mock()
            v1_response.is_error = False
            v1_response.status_code = 200
            v1_response.text = '{"status":"fail"}'
            v1_response.json.return_value = {"status": "fail"}
            mock_get.side_effect = [self._health_404_response(), v1_response]

            with pytest.raises(RuntimeError) as exc_info:
                runner._check_service()

            assert (
                exc_info.value.args[0]
                == f"Service at {self._v1_health_endpoint(runner)} is unhealthy: {v1_response.text}"
            )

    def test_check_service_fallback_invalid_json(self, runner):
        """An unparseable fallback health response is an error"""
        with patch("httpx.get") as mock_get:
            v1_response = Mock()
            v1_response.is_error = False
            v1_response.status_code = 200
            v1_response.text = "not json"
            json_error = JSONDecodeError("Expecting value", "not json", 0)
            v1_response.json.side_effect = json_error
            mock_get.side_effect = [self._health_404_response(), v1_response]

            with pytest.raises(RuntimeError) as exc_info:
                runner._check_service()

            assert exc_info.value.args[0] == f"Error: response not json couldn't be parsed as JSON: {json_error}"

    def test_build_locust_command_basic(self, runner):
        """Test building basic Locust command."""
        cmd = runner._build_locust_command()
        assert cmd[0] == "locust"

        cmd_string = " ".join(cmd)
        # 10 users at 2/s ramps for 5s, added on top of the 30s measured window.
        assert (
            "--host http://localhost:8000 --users 10 --spawn-rate 2.0 --run-time 35s --reset-stats --headless"
            in cmd_string
        )

    def test_ramp_is_added_to_the_measured_window(self, runner):
        """run_time is the measured duration; the ramp does not eat into it."""
        assert runner.ramp_seconds == 5
        assert runner.config.run_time == 30
        assert runner.total_run_seconds == 35

    def test_ramp_seconds_rounds_up(self, runner):
        """A partial ramp second still has to elapse before the plateau starts."""
        runner.config.target_users = 10
        runner.config.spawn_rate = 3

        assert runner.ramp_seconds == 4

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
        assert "--csv-full-history" in cmd

    def test_build_locust_command_non_headless(self, runner):
        """Test building Locust command in web UI mode (non-headless)."""
        runner.config.headless = False

        cmd = runner._build_locust_command()

        assert "--headless" not in cmd
        assert "--only-summary" not in cmd
        assert "--html" not in cmd
        assert "--csv" not in cmd
        assert "--csv-full-history" not in cmd

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


class TestLocustSweepRunner:
    """Test running a sweep of concurrency levels."""

    @pytest.fixture
    def sweep_config(self, tmp_path):
        """A three-level concurrency sweep writing under tmp_path."""
        return LocustSweepConfig(
            batch_name="sweep",
            output_base_dir=str(tmp_path / "results"),
            base_config={
                "host": "http://localhost:9000",
                "config_id": "test-config",
                "model": "test-model",
                "spawn_rate": 4,
                "run_time": 10,
                "headless": True,
            },
            sweeps={"target_users": [1, 2, 4]},
        )

    @pytest.fixture
    def sweep_runner(self, sweep_config):
        return LocustSweepRunner(sweep_config)

    def test_runs_every_level_in_order(self, sweep_runner):
        """Each swept value becomes its own Locust invocation."""
        with patch.object(LocustRunner, "_check_service"), patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=0)

            assert sweep_runner.run(dry_run=False) == 0
            assert mock_run.call_count == 3

            users = [command[command.index("--users") + 1] for (command,) in (c[0] for c in mock_run.call_args_list)]
            assert users == ["1", "2", "4"]

    def test_level_directories_are_named_by_swept_value(self, sweep_runner):
        """Level directories are deterministic, which is what makes resume possible."""
        with patch.object(LocustRunner, "_check_service"), patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=0)
            sweep_runner.run(dry_run=False)

        names = sorted(path.name for path in sweep_runner.batch_path.iterdir())
        assert names == ["target_users-1", "target_users-2", "target_users-4"]

    def test_request_failures_do_not_stop_the_sweep(self, sweep_runner):
        """A level past saturation exits non-zero; later levels still run."""
        with patch.object(LocustRunner, "_check_service"), patch("subprocess.run") as mock_run:
            # The middle level fails, as an overloaded server would.
            mock_run.side_effect = [Mock(returncode=0), Mock(returncode=1), Mock(returncode=0)]

            exit_code = sweep_runner.run(dry_run=False)

            assert exit_code == 0, "request failures are a result, not a runner error"
            assert mock_run.call_count == 3

    def test_exit_code_is_recorded_per_level(self, sweep_runner):
        """Each level's Locust exit code survives in its metadata."""
        with patch.object(LocustRunner, "_check_service"), patch("subprocess.run") as mock_run:
            mock_run.side_effect = [Mock(returncode=0), Mock(returncode=1), Mock(returncode=0)]
            sweep_runner.run(dry_run=False)

        recorded = {
            path.name: json.loads((path / "run_metadata.json").read_text())["exit_code"]
            for path in sweep_runner.batch_path.iterdir()
        }
        assert recorded == {"target_users-1": 0, "target_users-2": 1, "target_users-4": 0}

    def test_health_check_runs_once_for_the_batch(self, sweep_runner):
        """The server is checked once, not once per level."""
        with patch.object(LocustRunner, "_check_service") as mock_check, patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=0)
            sweep_runner.run(dry_run=False)

            mock_check.assert_called_once()

    def test_failed_health_check_stops_the_sweep(self, sweep_runner):
        """A sweep that cannot reach the server did not run, so it exits non-zero."""
        with patch.object(LocustRunner, "_check_service") as mock_check, patch("subprocess.run") as mock_run:
            mock_check.side_effect = RuntimeError("Service unavailable")

            assert sweep_runner.run(dry_run=False) == 1
            mock_run.assert_not_called()

    def test_resume_skips_completed_levels(self, sweep_runner):
        """A completed level is kept rather than re-run."""
        with patch.object(LocustRunner, "_check_service"), patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=0)
            sweep_runner.run(dry_run=False)
            assert mock_run.call_count == 3

            mock_run.reset_mock()
            sweep_runner.run(dry_run=False, resume=True)

            assert mock_run.call_count == 0

    def test_resume_reruns_an_interrupted_level(self, sweep_runner):
        """Metadata without an exit code means the level never finished."""
        interrupted = sweep_runner.batch_path / "target_users-2"
        interrupted.mkdir(parents=True)
        (interrupted / "run_metadata.json").write_text(json.dumps({"exit_code": None}))

        with patch.object(LocustRunner, "_check_service"), patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=0)
            sweep_runner.run(dry_run=False, resume=True)

            assert mock_run.call_count == 3

    def test_is_complete_rejects_unreadable_metadata(self, sweep_runner, tmp_path):
        """Corrupt metadata is treated as incomplete rather than crashing the sweep."""
        level = tmp_path / "level"
        level.mkdir()
        (level / "run_metadata.json").write_text("{not json")

        assert sweep_runner.is_complete(level) is False

    def test_every_swept_host_is_health_checked(self, tmp_path):
        """Sweeping host targets several servers, and each must be reachable."""
        config = LocustSweepConfig(
            batch_name="hosts",
            output_base_dir=str(tmp_path / "results"),
            base_config={"config_id": "test-config", "model": "test-model"},
            sweeps={"host": ["http://localhost:9000", "http://localhost:9001"]},
        )

        with patch.object(LocustRunner, "_check_service") as mock_check, patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=0)
            LocustSweepRunner(config).run(dry_run=False)

            assert mock_check.call_count == 2

    def test_repeated_host_is_checked_once(self, sweep_runner):
        """Levels sharing a host do not re-check it."""
        with patch.object(LocustRunner, "_check_service") as mock_check, patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=0)
            sweep_runner.run(dry_run=False)

            mock_check.assert_called_once()

    def test_resume_reruns_a_level_whose_config_changed(self, sweep_runner):
        """A level is identified by its swept values, so unswept settings can drift."""
        with patch.object(LocustRunner, "_check_service"), patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=0)
            sweep_runner.run(dry_run=False)

            # Change a setting that is not part of the level's identity.
            sweep_runner.config.base_config.message = "a different prompt"

            mock_run.reset_mock()
            sweep_runner.run(dry_run=False, resume=True)

            assert mock_run.call_count == 3, "stale results must not be kept under --resume"

    def test_resume_keeps_levels_whose_config_is_unchanged(self, sweep_runner):
        """The config check must not defeat resume for an unmodified batch."""
        with patch.object(LocustRunner, "_check_service"), patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=0)
            sweep_runner.run(dry_run=False)

            mock_run.reset_mock()
            sweep_runner.run(dry_run=False, resume=True)

            assert mock_run.call_count == 0

    def test_estimate_sums_ramp_and_measured_for_every_level(self, sweep_runner):
        """The up-front estimate is what tells someone whether to start the run."""
        levels = sweep_runner.config.expand()

        # 1, 2 and 4 users at spawn_rate 4 each ramp 1s, then 10s measured.
        assert sweep_runner.estimated_seconds(levels) == 3 * (1 + 10)

    def test_estimate_switches_to_hours_when_long(self, sweep_runner):
        """A multi-hour ladder should not be reported as "240 minutes"."""
        assert sweep_runner._format_duration(60) == "1 minute"
        assert sweep_runner._format_duration(90) == "2 minutes"
        assert sweep_runner._format_duration(30 * 60) == "30 minutes"
        assert sweep_runner._format_duration(89 * 60) == "89 minutes"
        assert sweep_runner._format_duration(240 * 60) == "4.0 hours"

    def test_resume_does_not_check_hosts_whose_levels_are_all_complete(self, tmp_path):
        """A finished host going away must not block incomplete levels elsewhere."""
        config = LocustSweepConfig(
            batch_name="hosts",
            output_base_dir=str(tmp_path / "results"),
            base_config={"config_id": "test-config", "model": "test-model", "run_time": 5},
            sweeps={"host": ["http://localhost:9000", "http://localhost:9001"]},
        )
        sweep = LocustSweepRunner(config)

        # Complete only the first host's level.
        done, pending = sorted(label for label, _ in config.expand())
        done_path = sweep.batch_path / done
        done_path.mkdir(parents=True)
        completed = next(c for label, c in config.expand() if label == done)
        (done_path / "run_metadata.json").write_text(
            json.dumps({"exit_code": 0, "config": json.loads(completed.model_dump_json())})
        )

        checked = []
        with (
            patch.object(LocustRunner, "_check_service", autospec=True) as mock_check,
            patch("subprocess.run") as mock_run,
        ):
            mock_check.side_effect = lambda self: checked.append(self.config.host)
            mock_run.return_value = Mock(returncode=0)

            assert sweep.run(dry_run=False, resume=True) == 0

        assert len(checked) == 1, f"only the pending host should be checked, got {checked}"
        assert mock_run.call_count == 1

    def test_resume_with_everything_complete_runs_nothing(self, sweep_runner):
        """A fully complete batch exits cleanly without probing the server."""
        with patch.object(LocustRunner, "_check_service"), patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=0)
            sweep_runner.run(dry_run=False)

        with patch.object(LocustRunner, "_check_service") as mock_check, patch("subprocess.run") as mock_run:
            assert sweep_runner.run(dry_run=False, resume=True) == 0
            mock_check.assert_not_called()
            mock_run.assert_not_called()

    def test_dry_run_executes_nothing(self, sweep_runner):
        """Dry-run prints each level's command without running or checking the service."""
        with patch.object(LocustRunner, "_check_service") as mock_check, patch("subprocess.run") as mock_run:
            assert sweep_runner.run(dry_run=True) == 0

            mock_check.assert_not_called()
            mock_run.assert_not_called()


class TestLoadConfigFromYaml:
    """Test _load_config_from_yaml function."""

    def test_load_valid_config(self, create_config_file):
        """Test loading a valid config file."""
        config_file = create_config_file()

        config = _load_config_from_yaml(config_file)

        assert isinstance(config, LocustSweepConfig)
        assert config.sweeps is None
        assert config.base_config.config_id == "test-config"
        assert config.base_config.model == "test-model"

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
