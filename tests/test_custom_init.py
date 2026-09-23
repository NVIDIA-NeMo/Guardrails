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

import os
import re
from pathlib import Path

import pytest
import yaml

from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.testing.fake_model import FakeLLMModel
from tests.utils import TestChat

CONFIGS_FOLDER = os.path.join(os.path.dirname(__file__), ".", "test_configs")


def test_custom_init():
    config = RailsConfig.from_path(os.path.join(CONFIGS_FOLDER, "with_custom_init"))
    chat = TestChat(
        config,
        llm_completions=[
            "  express greeting",
        ],
    )

    chat >> "hi"
    chat << "John"


def _write_config(config_path, config_module, config_content="models: []\n"):
    config_path.mkdir()
    (config_path / "config.yml").write_text(config_content, encoding="utf-8")
    return (config_path / "config.py").write_text(config_module, encoding="utf-8")


def test_custom_init_runs_for_each_combined_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    first_config_path = Path("first")
    second_config_path = Path("second")
    _write_config(
        first_config_path,
        """
def parse_policy_output(_response):
    return [False]

def init(app):
    app.register_output_parser(parse_policy_output, "policy_parser")
    app.register_action_param("first_config_initialized", True)
""",
    )
    _write_config(
        second_config_path,
        """
def init(app):
    app.register_action_param("second_config_initialized", True)
""",
    )
    Path("first,second").mkdir()

    config = RailsConfig.from_path(str(first_config_path)) + RailsConfig.from_path(str(second_config_path))
    rails = LLMRails(config, llm=FakeLLMModel(responses=[]))

    assert rails.runtime.registered_action_params["first_config_initialized"] is True
    assert rails.runtime.registered_action_params["second_config_initialized"] is True
    assert rails.runtime.llm_task_manager.output_parsers["policy_parser"]("raw output") == [False]


def test_custom_init_deduplicates_imported_and_combined_config(tmp_path):
    imported_config_path = tmp_path / "imported"
    _write_config(
        imported_config_path,
        """
def init(app):
    params = app.runtime.registered_action_params
    app.register_action_param("init_count", params.get("init_count", 0) + 1)
""",
    )
    importing_config_path = tmp_path / "importing"
    _write_config(
        importing_config_path,
        "",
        yaml.safe_dump(
            {"models": [], "import_paths": [str(imported_config_path)]},
            sort_keys=False,
        ),
    )

    config = RailsConfig.from_path(str(importing_config_path)) + RailsConfig.from_path(str(imported_config_path))
    rails = LLMRails(config, llm=FakeLLMModel(responses=[]))

    assert rails.runtime.registered_action_params["init_count"] == 1


@pytest.mark.parametrize(
    ("config_module", "error_message"),
    [
        ('raise ValueError("load failed")\n', "Failed to load configuration module"),
        ('def init(_app):\n    raise ValueError("init failed")\n', "Failed to initialize configuration module"),
    ],
)
def test_custom_init_failure_names_config_file(tmp_path, config_module, error_message):
    config_path = tmp_path / "broken"
    _write_config(config_path, config_module)
    config_file = config_path / "config.py"
    config = RailsConfig.from_path(str(config_path))

    with pytest.raises(RuntimeError, match=rf"{error_message} at {re.escape(str(config_file))}") as exc_info:
        LLMRails(config, llm=FakeLLMModel(responses=[]))

    assert isinstance(exc_info.value.__cause__, ValueError)


def test_custom_init_failure_when_module_loader_is_unavailable(tmp_path, monkeypatch):
    config_path = tmp_path / "broken"
    _write_config(config_path, "")
    config_file = config_path / "config.py"
    config = RailsConfig.from_path(str(config_path))
    monkeypatch.setattr("nemoguardrails.rails.llm.llmrails.importlib.util.spec_from_file_location", lambda *_args: None)

    with pytest.raises(
        RuntimeError,
        match=rf"Failed to load configuration module at {re.escape(str(config_file))}",
    ) as exc_info:
        LLMRails(config, llm=FakeLLMModel(responses=[]))

    assert isinstance(exc_info.value.__cause__, ImportError)
