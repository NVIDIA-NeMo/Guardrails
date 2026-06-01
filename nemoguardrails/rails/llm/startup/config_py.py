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

"""Config.py module hooks."""

import importlib.util
import os
from types import ModuleType
from typing import Any, List

from nemoguardrails.rails.llm.config import RailsConfig

__all__ = ["load_config_py_modules", "run_config_py_init_hooks"]


def load_config_py_modules(config: RailsConfig) -> List[ModuleType]:
    """Load config.py modules from imported config paths and the main config path."""
    config_modules = []
    config_paths = list(config.imported_paths.values() if config.imported_paths else []) + [config.config_path]

    for config_path in config_paths:
        if not config_path:
            continue

        filepath = os.path.join(config_path, "config.py")
        if not os.path.exists(filepath):
            continue

        filename = os.path.basename(filepath)
        spec = importlib.util.spec_from_file_location(filename, filepath)
        if spec and spec.loader:
            config_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(config_module)
            config_modules.append(config_module)

    return config_modules


def run_config_py_init_hooks(rails: Any, config_modules: List[ModuleType]) -> None:
    """Run config.py init hooks with the LLMRails instance."""
    for config_module in config_modules:
        if hasattr(config_module, "init"):
            config_module.init(rails)
