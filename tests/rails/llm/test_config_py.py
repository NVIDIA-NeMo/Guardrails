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

from types import ModuleType

from nemoguardrails.rails.llm.config import RailsConfig
from nemoguardrails.rails.llm.llmrails import LLMRails
from nemoguardrails.rails.llm.startup.config_py import (
    load_config_py_modules,
    run_config_py_init_hooks,
)


def test_load_config_py_modules_preserves_imported_then_main_order(tmp_path):
    imported_path = tmp_path / "imported"
    main_path = tmp_path / "main"
    imported_path.mkdir()
    main_path.mkdir()
    (imported_path / "config.py").write_text("SOURCE = 'imported'\n")
    (main_path / "config.py").write_text("SOURCE = 'main'\n")

    config = RailsConfig(
        config_path=str(main_path),
        imported_paths={"imported": str(imported_path)},
        models=[],
    )

    config_modules = load_config_py_modules(config)

    assert [config_module.SOURCE for config_module in config_modules] == [
        "imported",
        "main",
    ]


def test_load_config_py_modules_skips_missing_paths(tmp_path):
    main_path = tmp_path / "main"
    imported_path = tmp_path / "imported"
    main_path.mkdir()
    imported_path.mkdir()
    (main_path / "config.py").write_text("SOURCE = 'main'\n")

    config = RailsConfig(
        config_path=str(main_path),
        imported_paths={"imported": str(imported_path)},
        models=[],
    )

    config_modules = load_config_py_modules(config)

    assert [config_module.SOURCE for config_module in config_modules] == ["main"]


def test_llmrails_runs_imported_config_py_init_before_main(tmp_path):
    imported_path = tmp_path / "imported"
    main_path = tmp_path / "main"
    imported_path.mkdir()
    main_path.mkdir()
    (imported_path / "config.py").write_text("def init(app):\n    app.hook_order = ['imported']\n")
    (main_path / "config.py").write_text("def init(app):\n    app.hook_order.append('main')\n")

    config = RailsConfig(
        config_path=str(main_path),
        imported_paths={"imported": str(imported_path)},
        models=[],
    )

    rails = LLMRails(config)

    assert getattr(rails, "hook_order") == ["imported", "main"]


def test_run_config_py_init_hooks_passes_rails_instance():
    rails = object()
    config_module = ModuleType("config")

    def init(received_rails):
        setattr(config_module, "received_rails", received_rails)

    setattr(config_module, "init", init)

    run_config_py_init_hooks(rails, [config_module])

    assert getattr(config_module, "received_rails") is rails


def test_run_config_py_init_hooks_ignores_modules_without_init():
    rails = object()
    config_module = ModuleType("config")

    run_config_py_init_hooks(rails, [config_module])
