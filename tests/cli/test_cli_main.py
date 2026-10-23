# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import logging
import os
import sys
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from nemoguardrails import __version__, telemetry
from nemoguardrails.cli import app

runner = CliRunner()

HOOKS_CONFIG_YML = """
models: []
rails:
  tool_call:
    per_tool:
      Bash: [regex check tool call]
"""

METRICS_VARIABLES = (
    "NEMO_GUARDRAILS_SERVER_METRICS_EXPORTER",
    "NEMO_GUARDRAILS_SERVER_METRICS_HOST",
    "NEMO_GUARDRAILS_SERVER_METRICS_PORT",
)


def test_app():
    result = runner.invoke(
        app,
        [
            "chat",
            "--config=examples/rails/benefits_co/config.yml",
            "--config=examples/rails/benefits_co/general.co",
        ],
    )
    assert result.exit_code == 1
    assert "not supported" in result.stdout
    assert "Please provide a single" in result.stdout


class TestCLIVersion:
    def test_version_flag(self):
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert __version__ in result.stdout

    def test_version_flag_short(self):
        result = runner.invoke(app, ["-v"])
        assert result.exit_code == 0
        assert __version__ in result.stdout


class TestChatCommand:
    def test_chat_with_multiple_configs_fails(self):
        result = runner.invoke(
            app,
            [
                "chat",
                "--config=config1.yml",
                "--config=config2.yml",
            ],
        )
        assert result.exit_code == 1
        assert "Multiple configurations are not supported" in result.stdout

    @patch("nemoguardrails.cli.run_chat")
    @patch("os.path.exists")
    def test_chat_with_single_config(self, mock_exists, mock_run_chat):
        mock_exists.return_value = True
        result = runner.invoke(app, ["chat", "--config=test_config"])
        assert result.exit_code == 0
        mock_run_chat.assert_called_once()

    @patch("nemoguardrails.cli.run_chat")
    @patch("os.path.exists")
    def test_chat_with_verbose(self, mock_exists, mock_run_chat):
        mock_exists.return_value = True
        result = runner.invoke(app, ["chat", "--config=test_config", "--verbose"])
        assert result.exit_code == 0
        mock_run_chat.assert_called_once_with(
            config_path="test_config",
            verbose=True,
            verbose_llm_calls=True,
            streaming=False,
            server_url=None,
            config_id=None,
        )

    @patch("nemoguardrails.cli.run_chat")
    @patch("os.path.exists")
    def test_chat_with_verbose_no_llm(self, mock_exists, mock_run_chat):
        mock_exists.return_value = True
        result = runner.invoke(app, ["chat", "--config=test_config", "--verbose-no-llm"])
        assert result.exit_code == 0
        mock_run_chat.assert_called_once_with(
            config_path="test_config",
            verbose=True,
            verbose_llm_calls=False,
            streaming=False,
            server_url=None,
            config_id=None,
        )

    @patch("nemoguardrails.cli.run_chat")
    @patch("os.path.exists")
    def test_chat_with_streaming(self, mock_exists, mock_run_chat):
        mock_exists.return_value = True
        result = runner.invoke(app, ["chat", "--config=test_config", "--streaming"])
        assert result.exit_code == 0
        mock_run_chat.assert_called_once_with(
            config_path="test_config",
            verbose=False,
            verbose_llm_calls=True,
            streaming=True,
            server_url=None,
            config_id=None,
        )

    @patch("nemoguardrails.cli.run_chat")
    def test_chat_with_server_url(self, mock_run_chat):
        result = runner.invoke(
            app,
            [
                "chat",
                "--server-url=http://localhost:8000",
                "--config-id=test_id",
            ],
        )
        assert result.exit_code == 0
        mock_run_chat.assert_called_once_with(
            config_path="config",
            verbose=False,
            verbose_llm_calls=True,
            streaming=False,
            server_url="http://localhost:8000",
            config_id="test_id",
        )

    @patch("nemoguardrails.cli.run_chat")
    @patch("nemoguardrails.cli.init_random_seed")
    @patch("os.path.exists")
    def test_chat_with_debug_level(self, mock_exists, mock_init_seed, mock_run_chat):
        mock_exists.return_value = True
        result = runner.invoke(app, ["chat", "--config=test_config", "--debug-level=DEBUG"])
        assert result.exit_code == 0
        mock_init_seed.assert_called_once_with(0)
        mock_run_chat.assert_called_once()


class TestServerCommand:
    @patch("uvicorn.run")
    @patch("nemoguardrails.server.api.app")
    def test_server_default_port(self, mock_app, mock_uvicorn):
        result = runner.invoke(app, ["server"])
        assert result.exit_code == 0
        mock_uvicorn.assert_called_once()
        call_args = mock_uvicorn.call_args
        assert call_args[1]["port"] == 8000
        assert call_args[1]["host"] == "0.0.0.0"

    @patch("uvicorn.run")
    @patch("nemoguardrails.server.api.app")
    def test_server_custom_port(self, mock_app, mock_uvicorn):
        result = runner.invoke(app, ["server", "--port=9000"])
        assert result.exit_code == 0
        mock_uvicorn.assert_called_once()
        call_args = mock_uvicorn.call_args
        assert call_args[1]["port"] == 9000

    @patch("uvicorn.run")
    @patch("nemoguardrails.server.api.app")
    @patch("os.path.exists")
    @patch("os.path.expanduser")
    def test_server_with_config(self, mock_expanduser, mock_exists, mock_app, mock_uvicorn):
        mock_expanduser.return_value = "/path/to/config"
        mock_exists.return_value = True
        result = runner.invoke(app, ["server", "--config=/path/to/config"])
        assert result.exit_code == 0
        assert mock_app.rails_config_path == "/path/to/config"

    @patch("uvicorn.run")
    @patch("nemoguardrails.server.api.app")
    @patch("os.path.exists")
    @patch("os.getcwd")
    def test_server_with_local_config(self, mock_getcwd, mock_exists, mock_app, mock_uvicorn):
        mock_getcwd.return_value = "/current/dir"
        mock_exists.return_value = True
        result = runner.invoke(app, ["server"])
        assert result.exit_code == 0
        expected_path = os.path.join("/current/dir", "config")
        assert mock_app.rails_config_path == expected_path

    @patch("uvicorn.run")
    @patch("nemoguardrails.server.api.app")
    def test_server_with_disable_chat_ui(self, mock_app, mock_uvicorn):
        result = runner.invoke(app, ["server", "--disable-chat-ui"])
        assert result.exit_code == 0
        assert mock_app.disable_chat_ui is True

    @patch("nemoguardrails.server.metrics.shutdown_metrics_exporter")
    @patch("nemoguardrails.server.metrics.start_metrics_exporter")
    @patch("uvicorn.run")
    @patch("nemoguardrails.server.api.app")
    def test_server_metrics_flags_forward_to_env_and_start_exporter(
        self, mock_app, mock_uvicorn, mock_start, mock_shutdown
    ):
        """The metrics flags land in the env vars the lifespan reads, and the exporter starts and stops around uvicorn."""
        with patch.dict(os.environ, {}, clear=True):
            result = runner.invoke(
                app,
                ["server", "--metrics-exporter=prometheus", "--metrics-host=127.0.0.1", "--metrics-port=9500"],
            )
            assert result.exit_code == 0, result.output
            assert os.environ["NEMO_GUARDRAILS_SERVER_METRICS_EXPORTER"] == "prometheus"
            assert os.environ["NEMO_GUARDRAILS_SERVER_METRICS_HOST"] == "127.0.0.1"
            assert os.environ["NEMO_GUARDRAILS_SERVER_METRICS_PORT"] == "9500"
        mock_start.assert_called_once_with()
        mock_uvicorn.assert_called_once()
        mock_shutdown.assert_called_once_with()

    @patch("nemoguardrails.server.metrics.shutdown_metrics_exporter")
    @patch("nemoguardrails.server.metrics.start_metrics_exporter")
    @patch("uvicorn.run")
    @patch("nemoguardrails.server.api.app")
    def test_server_without_metrics_flags_leaves_env_alone(self, mock_app, mock_uvicorn, mock_start, mock_shutdown):
        """Omitting the metrics flags leaves the environment untouched so env-var configuration still applies."""
        with patch.dict(os.environ, {}, clear=True):
            result = runner.invoke(app, ["server"])
            assert result.exit_code == 0
            assert "NEMO_GUARDRAILS_SERVER_METRICS_EXPORTER" not in os.environ
        mock_start.assert_called_once_with()

    @patch("nemoguardrails.server.metrics.start_metrics_exporter")
    @patch("uvicorn.run")
    @patch("nemoguardrails.server.api.app")
    def test_server_exits_before_binding_when_metrics_config_is_invalid(self, mock_app, mock_uvicorn, mock_start):
        """An exporter configuration error is printed and exits 1 before uvicorn binds the API port."""
        from nemoguardrails.server.metrics import MetricsExporterConfigError

        mock_start.side_effect = MetricsExporterConfigError("Unsupported exporter 'otlp'")
        with patch.dict(os.environ, {}, clear=True):
            result = runner.invoke(app, ["server", "--metrics-exporter=otlp"])
        assert result.exit_code == 1
        assert "Unsupported exporter 'otlp'" in result.output
        mock_uvicorn.assert_not_called()

    @patch("uvicorn.run")
    @patch("nemoguardrails.server.api.app")
    def test_server_with_auto_reload(self, mock_app, mock_uvicorn):
        result = runner.invoke(app, ["server", "--auto-reload"])
        assert result.exit_code == 0
        assert mock_app.auto_reload is True

    @patch("uvicorn.run")
    @patch("nemoguardrails.server.api.app")
    @patch("nemoguardrails.server.api.set_default_config_id")
    def test_server_with_default_config_id(self, mock_set_default, mock_app, mock_uvicorn):
        result = runner.invoke(app, ["server", "--default-config-id=test_config"])
        assert result.exit_code == 0
        mock_set_default.assert_called_once_with("test_config")

    @patch("uvicorn.run")
    @patch("nemoguardrails.server.api.app")
    def test_server_with_prefix(self, mock_app, mock_uvicorn):
        from fastapi import FastAPI

        with patch.object(FastAPI, "mount") as mock_mount:
            result = runner.invoke(app, ["server", "--prefix=/api/v1"])
            assert result.exit_code == 0
            mock_mount.assert_called_once()

    @patch("uvicorn.run")
    @patch("nemoguardrails.server.api.app")
    @patch("nemoguardrails.telemetry.set_deployment_type")
    def test_server_sets_api_deployment_type_before_prefixed_mount(
        self, mock_set_deployment_type, mock_app, mock_uvicorn
    ):
        from fastapi import FastAPI

        order = []
        mock_set_deployment_type.side_effect = lambda deployment_type: order.append(("telemetry", deployment_type))

        with patch.object(FastAPI, "mount", side_effect=lambda *args, **kwargs: order.append(("mount", None))):
            result = runner.invoke(app, ["server", "--prefix=/api/v1"])

        assert result.exit_code == 0
        mock_set_deployment_type.assert_called_once_with("api")
        assert order == [("telemetry", "api"), ("mount", None)]

    @patch("uvicorn.run")
    @patch("nemoguardrails.server.api.app")
    def test_server_with_prefix_reports_api_deployment_type_during_mount(self, mock_app, mock_uvicorn):
        from fastapi import FastAPI

        from nemoguardrails import telemetry

        telemetry._session_uuid = None
        telemetry._heartbeat_started = False
        telemetry._deployment_type_override = None
        telemetry._lock = telemetry.threading.Lock()

        def report_during_mount(*args, **kwargs):
            telemetry.report_usage(None, deployment_type="library")

        try:
            with (
                patch.object(telemetry, "_is_usage_stats_enabled", return_value=True),
                patch("nemoguardrails.telemetry.threading.Thread") as mock_thread,
                patch.object(FastAPI, "mount", side_effect=report_during_mount),
            ):
                mock_thread.return_value = MagicMock()
                result = runner.invoke(app, ["server", "--prefix=/api/v1"])

            assert result.exit_code == 0
            send_calls = [
                call
                for call in mock_thread.call_args_list
                if getattr(call.kwargs.get("target"), "__name__", "") == "_send_one_event"
            ]
            assert len(send_calls) == 1
            usage_data = send_calls[0].kwargs["args"][0]
            assert usage_data.deployment_type == "api"
        finally:
            telemetry._session_uuid = None
            telemetry._heartbeat_started = False
            telemetry._deployment_type_override = None
            telemetry._lock = telemetry.threading.Lock()


class TestConvertCommand:
    def test_convert_missing_path(self):
        result = runner.invoke(app, ["convert"])
        assert result.exit_code != 0

    @patch("os.path.abspath")
    @patch("nemoguardrails.cli.migrate")
    def test_convert_with_defaults(self, mock_migrate, mock_abspath):
        mock_abspath.return_value = "/abs/path/to/file"
        result = runner.invoke(app, ["convert", "test_file.co"])
        assert result.exit_code == 0
        mock_migrate.assert_called_once_with(
            path="/abs/path/to/file",
            include_main_flow=True,
            use_active_decorator=True,
            from_version="1.0",
            validate=False,
        )

    @patch("os.path.abspath")
    @patch("nemoguardrails.cli.migrate")
    def test_convert_with_options(self, mock_migrate, mock_abspath):
        mock_abspath.return_value = "/abs/path/to/file"
        result = runner.invoke(
            app,
            [
                "convert",
                "test_file.co",
                "--from-version=2.0-alpha",
                "--validate",
                "--no-use-active-decorator",
                "--no-include-main-flow",
            ],
        )
        assert result.exit_code == 0
        mock_migrate.assert_called_once_with(
            path="/abs/path/to/file",
            include_main_flow=False,
            use_active_decorator=False,
            from_version="2.0-alpha",
            validate=True,
        )

    @patch("nemoguardrails.cli.migrate")
    @patch("os.path.abspath")
    @patch("logging.getLogger")
    def test_convert_with_verbose(self, mock_logger, mock_abspath, mock_migrate):
        mock_abspath.return_value = "/abs/path/to/file"
        mock_logger_instance = MagicMock()
        mock_logger.return_value = mock_logger_instance
        result = runner.invoke(app, ["convert", "test_file.co", "--verbose"])
        assert result.exit_code == 0
        mock_logger_instance.setLevel.assert_called()


class TestActionsServerCommand:
    @patch("uvicorn.run")
    def test_actions_server_default_port(self, mock_uvicorn):
        result = runner.invoke(app, ["actions-server"])
        assert result.exit_code == 0
        mock_uvicorn.assert_called_once()
        call_args = mock_uvicorn.call_args
        assert call_args[1]["port"] == 8001
        assert call_args[1]["host"] == "0.0.0.0"

    @patch("uvicorn.run")
    def test_actions_server_custom_port(self, mock_uvicorn):
        result = runner.invoke(app, ["actions-server", "--port=9001"])
        assert result.exit_code == 0
        mock_uvicorn.assert_called_once()
        call_args = mock_uvicorn.call_args
        assert call_args[1]["port"] == 9001


class TestHooksServerCommand:
    @pytest.fixture(autouse=True)
    def _hooks_server_env(self, monkeypatch):
        """Provider credentials, no metrics variables, and the process's deployment type restored after the test."""
        monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
        for name in (*METRICS_VARIABLES, "NEMO_GUARDRAILS_SERVER_METRICS_SERVICE_NAME", "OTEL_SERVICE_NAME"):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setattr(telemetry, "_deployment_type_override", None)

    @staticmethod
    def _config_dir(tmp_path, config_yml: str) -> str:
        """A config directory holding *config_yml* as its `config.yml`."""
        (tmp_path / "config.yml").write_text(config_yml)
        return str(tmp_path)

    @patch("uvicorn.run")
    def test_hooks_server_serves_the_config_on_loopback_port_8100_by_default(self, mock_uvicorn, tmp_path):
        """With only `--config`, the hooks app is served on `127.0.0.1:8100`."""
        from fastapi import FastAPI

        result = runner.invoke(app, ["hooks-server", f"--config={self._config_dir(tmp_path, HOOKS_CONFIG_YML)}"])

        assert result.exit_code == 0, result.output
        (server_app,), kwargs = mock_uvicorn.call_args
        assert isinstance(server_app, FastAPI)
        assert kwargs == {"host": "127.0.0.1", "port": 8100, "log_level": "info"}

    @patch("uvicorn.run")
    def test_hooks_server_logs_at_info_even_when_the_root_logger_has_a_handler(self, mock_uvicorn, tmp_path):
        """The root logger is at INFO when the server starts, so each check's log line shows, whatever handlers exist."""
        root = logging.getLogger()
        saved_level = root.level
        root.setLevel(logging.WARNING)
        root.addHandler(handler := logging.NullHandler())
        try:
            result = runner.invoke(app, ["hooks-server", f"--config={self._config_dir(tmp_path, HOOKS_CONFIG_YML)}"])

            assert result.exit_code == 0, result.output
            assert root.level == logging.INFO
        finally:
            root.removeHandler(handler)
            root.setLevel(saved_level)

    @patch("uvicorn.run")
    def test_hooks_server_honors_host_and_port(self, mock_uvicorn, tmp_path):
        """`--host` and `--port` set where the server listens."""
        config = self._config_dir(tmp_path, HOOKS_CONFIG_YML)

        result = runner.invoke(app, ["hooks-server", f"--config={config}", "--host=0.0.0.0", "--port=9100"])

        assert result.exit_code == 0, result.output
        assert (mock_uvicorn.call_args.kwargs["host"], mock_uvicorn.call_args.kwargs["port"]) == ("0.0.0.0", 9100)

    @patch("uvicorn.run")
    def test_hooks_server_exits_on_an_invalid_config_without_quoting_it(self, mock_uvicorn, tmp_path):
        """A config that fails validation exits 1 with each failure, before binding, and without its values."""
        config_yml = (
            "models:\n"
            "  - type: main\n"
            "    engine: nim\n"
            "    model: meta/llama-3.3-70b-instruct\n"
            "    parameters:\n"
            "      api_key: sk-secret-7f3a\n"
            "rails:\n"
            "  input:\n"
            "    flows: [content safety check input $model=content_safety]\n"
        )

        result = runner.invoke(app, ["hooks-server", f"--config={self._config_dir(tmp_path, config_yml)}"])

        assert result.exit_code == 1
        assert "Invalid configuration: config: Value error, Missing a `content_safety_check_input" in result.output
        assert "sk-secret-7f3a" not in result.output
        mock_uvicorn.assert_not_called()

    @patch("uvicorn.run")
    def test_hooks_server_exits_on_an_unset_api_key_variable(self, mock_uvicorn, tmp_path, monkeypatch):
        """A model whose `api_key_env_var` isn't set exits 1 with the variable's name, before binding."""
        monkeypatch.delenv("HOOKS_SERVER_TEST_API_KEY", raising=False)
        config_yml = (
            "models:\n"
            "  - type: main\n"
            "    engine: nim\n"
            "    model: meta/llama-3.3-70b-instruct\n"
            "    api_key_env_var: HOOKS_SERVER_TEST_API_KEY\n"
            "rails:\n"
            "  tool_call:\n"
            "    per_tool:\n"
            "      Bash: [regex check tool call]\n"
        )

        result = runner.invoke(app, ["hooks-server", f"--config={self._config_dir(tmp_path, config_yml)}"])

        assert result.exit_code == 1
        assert (
            "Invalid configuration: models: Value error, "
            "Model API Key environment variable 'HOOKS_SERVER_TEST_API_KEY' not set." in result.output
        )
        mock_uvicorn.assert_not_called()

    @patch("uvicorn.run")
    def test_hooks_server_exits_on_a_config_it_cannot_serve(self, mock_uvicorn, tmp_path):
        """A config that can deny no tool call exits 1 with the reason, before binding."""
        config_yml = "models:\n  - type: main\n    engine: nim\n    model: meta/llama-3.3-70b-instruct\n"

        result = runner.invoke(app, ["hooks-server", f"--config={self._config_dir(tmp_path, config_yml)}"])

        assert result.exit_code == 1
        assert "The hooks server needs a tool policy in the hooks_server section, or per-tool rails" in result.output
        mock_uvicorn.assert_not_called()

    @patch("uvicorn.run")
    def test_hooks_server_exits_on_a_config_that_is_not_yaml(self, mock_uvicorn, tmp_path):
        """A `config.yml` that isn't YAML exits 1 with the parser's message, before binding."""
        config = self._config_dir(tmp_path, "models: [\n")

        result = runner.invoke(app, ["hooks-server", f"--config={config}"])

        assert result.exit_code == 1
        assert "while parsing a flow node" in result.output
        mock_uvicorn.assert_not_called()

    @patch("uvicorn.run")
    def test_hooks_server_refuses_a_second_config(self, mock_uvicorn, tmp_path):
        """A second `--config` exits 1, rather than one of the two policies being dropped."""
        config = self._config_dir(tmp_path, HOOKS_CONFIG_YML)

        result = runner.invoke(app, ["hooks-server", f"--config={config}", f"--config={config}"])

        assert result.exit_code == 1
        assert "exactly one --config" in result.output
        mock_uvicorn.assert_not_called()

    @patch("uvicorn.run")
    def test_hooks_server_exits_on_a_config_that_sets_hooks_server_twice(self, mock_uvicorn, tmp_path):
        """A config with two files that set hooks_server exits 1 with the loader's message, before binding."""
        config = self._config_dir(tmp_path, HOOKS_CONFIG_YML + "hooks_server:\n  blocked_tools: [WebFetch]\n")
        (tmp_path / "extra.yml").write_text("hooks_server:\n  allowed_tools: [Read]\n")

        result = runner.invoke(app, ["hooks-server", f"--config={config}"])

        assert result.exit_code == 1
        assert "The hooks_server section is set in more than one file of this config" in result.output
        mock_uvicorn.assert_not_called()

    def test_hooks_server_without_the_server_extra_prints_the_install_hint(self, tmp_path, monkeypatch):
        """Without the `server` extra's packages, the command exits 1 with the install hint."""
        monkeypatch.setitem(sys.modules, "uvicorn", None)

        result = runner.invoke(app, ["hooks-server", f"--config={self._config_dir(tmp_path, HOOKS_CONFIG_YML)}"])

        assert result.exit_code == 1
        assert "Install them with: pip install nemoguardrails[server]" in result.output

    @patch("uvicorn.run")
    def test_hooks_server_requires_an_existing_config(self, mock_uvicorn, tmp_path):
        """A `--config` path that doesn't exist is a usage error."""
        result = runner.invoke(app, ["hooks-server", f"--config={tmp_path / 'missing'}"])

        assert result.exit_code == 2
        mock_uvicorn.assert_not_called()

    @patch("nemoguardrails.server.metrics.shutdown_metrics_exporter")
    @patch("nemoguardrails.server.metrics.start_metrics_exporter")
    @patch("uvicorn.run")
    def test_hooks_server_metrics_flags_start_the_exporter_around_the_server(
        self, mock_uvicorn, mock_start, mock_shutdown, tmp_path
    ):
        """The flags reach the variables; the exporter starts with them once the app is built, and stops after the server."""
        from nemoguardrails.server.hooks import app as hooks_app
        from nemoguardrails.server.metrics import MetricsExporter, MetricsExporterSettings

        events = []
        build_app = hooks_app.create_hooks_app

        def create_hooks_app(config):
            """Record that the app is built, and build it."""
            events.append("build")
            return build_app(config)

        mock_start.side_effect = lambda settings: events.append(("start", settings))
        mock_uvicorn.side_effect = lambda *args, **kwargs: events.append("serve")
        mock_shutdown.side_effect = lambda: events.append("stop")
        flags = ["--metrics-exporter=prometheus", "--metrics-host=0.0.0.0", "--metrics-port=9500"]

        with patch.dict(os.environ), patch.object(hooks_app, "create_hooks_app", create_hooks_app):
            result = runner.invoke(
                app, ["hooks-server", f"--config={self._config_dir(tmp_path, HOOKS_CONFIG_YML)}", *flags]
            )
            variables = {name: os.environ.get(name) for name in METRICS_VARIABLES}

        assert result.exit_code == 0, result.output
        assert variables == dict(zip(METRICS_VARIABLES, ["prometheus", "0.0.0.0", "9500"]))
        settings = MetricsExporterSettings(MetricsExporter.PROMETHEUS, "0.0.0.0", 9500, "nemoguardrails-hooks-server")
        assert events == ["build", ("start", settings), "serve", "stop"]

    @patch("nemoguardrails.server.metrics.shutdown_metrics_exporter")
    @patch("nemoguardrails.server.metrics.start_metrics_exporter")
    @patch("uvicorn.run")
    def test_hooks_server_without_metrics_flags_uses_the_hooks_defaults(
        self, mock_uvicorn, mock_start, mock_shutdown, tmp_path
    ):
        """Without the metrics flags, the variables stay unset, and the exporter gets the hooks server's defaults."""
        from nemoguardrails.server.metrics import MetricsExporterSettings

        with patch.dict(os.environ):
            result = runner.invoke(app, ["hooks-server", f"--config={self._config_dir(tmp_path, HOOKS_CONFIG_YML)}"])
            variables = {name: os.environ.get(name) for name in METRICS_VARIABLES}

        assert result.exit_code == 0, result.output
        assert variables == dict.fromkeys(METRICS_VARIABLES)
        mock_start.assert_called_once_with(
            MetricsExporterSettings(host="127.0.0.1", port=9465, service_name="nemoguardrails-hooks-server")
        )
        mock_shutdown.assert_called_once_with()

    @patch("uvicorn.run")
    def test_hooks_server_exits_on_an_invalid_metrics_setting_before_binding(self, mock_uvicorn, tmp_path):
        """An exporter setting the server can't use exits 1 with the reason, before the server starts."""
        config = self._config_dir(tmp_path, HOOKS_CONFIG_YML)

        with patch.dict(os.environ):
            result = runner.invoke(app, ["hooks-server", f"--config={config}", "--metrics-exporter=otlp"])

        assert result.exit_code == 1
        assert "Unsupported NEMO_GUARDRAILS_SERVER_METRICS_EXPORTER='otlp'" in result.output
        mock_uvicorn.assert_not_called()

    @patch("nemoguardrails.server.metrics.start_metrics_exporter")
    @patch("uvicorn.run")
    def test_hooks_server_refuses_a_config_before_starting_the_exporter(self, mock_uvicorn, mock_start, tmp_path):
        """A config the server can't serve exits 1 before the exporter starts, so no metrics port is bound."""
        config_yml = "models:\n  - type: main\n    engine: nim\n    model: meta/llama-3.3-70b-instruct\n"

        result = runner.invoke(app, ["hooks-server", f"--config={self._config_dir(tmp_path, config_yml)}"])

        assert result.exit_code == 1
        assert "The hooks server needs a tool policy in the hooks_server section, or per-tool rails" in result.output
        mock_start.assert_not_called()
        mock_uvicorn.assert_not_called()


class TestFindProvidersCommand:
    @patch("nemoguardrails.cli._list_providers")
    def test_find_providers_list_only(self, mock_list_providers):
        result = runner.invoke(app, ["find-providers", "--list"])
        assert result.exit_code == 0
        mock_list_providers.assert_called_once()

    @patch("nemoguardrails.cli.select_provider_with_type")
    def test_find_providers_interactive(self, mock_select):
        mock_select.return_value = ("chat completion", "openai")
        result = runner.invoke(app, ["find-providers"])
        assert result.exit_code == 0
        assert "Selected chat completion provider: openai" in result.stdout

    @patch("nemoguardrails.cli.select_provider_with_type")
    def test_find_providers_no_selection(self, mock_select):
        mock_select.return_value = None
        result = runner.invoke(app, ["find-providers"])
        assert result.exit_code == 0
        assert "No provider selected" in result.stdout


class TestEvalCommand:
    def test_eval_command_exists(self):
        result = runner.invoke(app, ["eval", "--help"])
        assert result.exit_code == 0
        assert "run" in result.stdout
        assert "check-compliance" in result.stdout
