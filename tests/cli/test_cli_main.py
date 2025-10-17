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

import os
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from nemoguardrails import __version__
from nemoguardrails.cli import app

runner = CliRunner()


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
        result = runner.invoke(
            app, ["chat", "--config=test_config", "--verbose-no-llm"]
        )
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
        result = runner.invoke(
            app, ["chat", "--config=test_config", "--debug-level=DEBUG"]
        )
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
    def test_server_with_config(
        self, mock_expanduser, mock_exists, mock_app, mock_uvicorn
    ):
        mock_expanduser.return_value = "/path/to/config"
        mock_exists.return_value = True
        result = runner.invoke(app, ["server", "--config=/path/to/config"])
        assert result.exit_code == 0
        assert mock_app.rails_config_path == "/path/to/config"

    @patch("uvicorn.run")
    @patch("nemoguardrails.server.api.app")
    @patch("os.path.exists")
    @patch("os.getcwd")
    def test_server_with_local_config(
        self, mock_getcwd, mock_exists, mock_app, mock_uvicorn
    ):
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

    @patch("uvicorn.run")
    @patch("nemoguardrails.server.api.app")
    def test_server_with_auto_reload(self, mock_app, mock_uvicorn):
        result = runner.invoke(app, ["server", "--auto-reload"])
        assert result.exit_code == 0
        assert mock_app.auto_reload is True

    @patch("uvicorn.run")
    @patch("nemoguardrails.server.api.app")
    @patch("nemoguardrails.server.api.set_default_config_id")
    def test_server_with_default_config_id(
        self, mock_set_default, mock_app, mock_uvicorn
    ):
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
