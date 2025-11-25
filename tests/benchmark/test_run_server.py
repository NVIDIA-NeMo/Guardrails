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
from unittest.mock import patch

import pytest

from nemoguardrails.benchmark.mock_llm_server.run_server import (
    parse_arguments,
    validate_config_file,
)


class TestParseArguments:
    def test_parse_arguments_defaults(self):
        with patch("sys.argv", ["run_server.py", "--config-file", "test.yaml"]):
            args = parse_arguments()
            assert args.host == "0.0.0.0"
            assert args.port == 8000
            assert args.reload is False
            assert args.log_level == "info"
            assert args.config_file == "test.yaml"

    def test_parse_arguments_custom_host(self):
        with patch(
            "sys.argv",
            ["run_server.py", "--config-file", "test.yaml", "--host", "127.0.0.1"],
        ):
            args = parse_arguments()
            assert args.host == "127.0.0.1"

    def test_parse_arguments_custom_port(self):
        with patch(
            "sys.argv",
            ["run_server.py", "--config-file", "test.yaml", "--port", "9000"],
        ):
            args = parse_arguments()
            assert args.port == 9000

    def test_parse_arguments_reload_flag(self):
        with patch("sys.argv", ["run_server.py", "--config-file", "test.yaml", "--reload"]):
            args = parse_arguments()
            assert args.reload is True

    def test_parse_arguments_custom_log_level(self):
        with patch(
            "sys.argv",
            ["run_server.py", "--config-file", "test.yaml", "--log-level", "debug"],
        ):
            args = parse_arguments()
            assert args.log_level == "debug"

    def test_parse_arguments_all_custom(self):
        with patch(
            "sys.argv",
            [
                "run_server.py",
                "--config-file",
                "custom.yaml",
                "--host",
                "localhost",
                "--port",
                "5000",
                "--reload",
                "--log-level",
                "trace",
            ],
        ):
            args = parse_arguments()
            assert args.host == "localhost"
            assert args.port == 5000
            assert args.reload is True
            assert args.log_level == "trace"
            assert args.config_file == "custom.yaml"

    def test_parse_arguments_missing_config_file(self):
        with patch("sys.argv", ["run_server.py"]):
            with pytest.raises(SystemExit):
                parse_arguments()


class TestValidateConfigFile:
    def test_validate_config_file_none(self):
        with pytest.raises(RuntimeError, match="No CONFIG_FILE"):
            validate_config_file(None)

    def test_validate_config_file_empty_string(self):
        with pytest.raises(RuntimeError, match="No CONFIG_FILE"):
            validate_config_file("")

    def test_validate_config_file_not_exists(self):
        with pytest.raises(RuntimeError, match="Can't open"):
            validate_config_file("/nonexistent/path/config.yaml")

    def test_validate_config_file_is_directory(self, tmp_path):
        with pytest.raises(RuntimeError, match="Can't open"):
            validate_config_file(str(tmp_path))

    def test_validate_config_file_valid(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("test: value")

        result = validate_config_file(str(config_file))
        assert result == str(config_file)

    def test_validate_config_file_valid_with_content(self, tmp_path):
        config_file = tmp_path / "model_config.yaml"
        config_file.write_text(
            """
models:
  - type: main
    engine: openai
    model: gpt-4
"""
        )

        result = validate_config_file(str(config_file))
        assert result == str(config_file)
        assert os.path.exists(result)
        assert os.path.isfile(result)
