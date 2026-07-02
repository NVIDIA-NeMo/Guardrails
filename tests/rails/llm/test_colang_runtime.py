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

import pytest

from nemoguardrails.colang.v1_0.runtime.runtime import RuntimeV1_0
from nemoguardrails.colang.v2_x.runtime.runtime import RuntimeV2_x
from nemoguardrails.exceptions import InvalidRailsConfigurationError
from nemoguardrails.rails.llm import runtime as runtime_package
from nemoguardrails.rails.llm.config import RailsConfig
from nemoguardrails.rails.llm.llmrails import LLMRails
from nemoguardrails.rails.llm.runtime import colang_runtime
from nemoguardrails.rails.llm.runtime.colang_runtime import runtime_for_colang_version


def test_runtime_package_has_no_star_exports():
    assert runtime_package.__all__ == []


def test_runtime_for_colang_version_uses_v1_runtime():
    config = RailsConfig(models=[], colang_version="1.0")

    runtime = runtime_for_colang_version(config=config, verbose=False)

    assert isinstance(runtime, RuntimeV1_0)


def test_runtime_for_colang_version_uses_v2_runtime():
    config = RailsConfig(models=[], colang_version="2.x")

    runtime = runtime_for_colang_version(config=config, verbose=True)

    assert isinstance(runtime, RuntimeV2_x)


def test_runtime_for_colang_version_rejects_unsupported_version():
    config = RailsConfig(models=[], colang_version="3.x")

    with pytest.raises(InvalidRailsConfigurationError) as exc_info:
        runtime_for_colang_version(config=config, verbose=False)

    assert str(exc_info.value) == "Unsupported colang version: 3.x. Supported versions: ['1.0', '2.x']"


def test_colang_runtime_module_exports_public_helpers():
    assert colang_runtime.__all__ == ["runtime_for_colang_version"]


def test_llmrails_initializes_v1_runtime():
    rails = LLMRails(RailsConfig(models=[], colang_version="1.0"))

    assert isinstance(rails.runtime, RuntimeV1_0)


def test_llmrails_initializes_v2_runtime():
    rails = LLMRails(RailsConfig(models=[], colang_version="2.x"))

    assert isinstance(rails.runtime, RuntimeV2_x)


def test_llmrails_rejects_unsupported_runtime_version():
    config = RailsConfig(models=[], colang_version="3.x")

    with pytest.raises(InvalidRailsConfigurationError) as exc_info:
        LLMRails(config)

    assert str(exc_info.value) == "Unsupported colang version: 3.x. Supported versions: ['1.0', '2.x']"


def test_config_py_init_hook_sees_initialized_runtime(tmp_path):
    config_path = tmp_path / "config"
    config_path.mkdir()
    (config_path / "config.py").write_text(
        "def init(app):\n    app.runtime_seen_by_init = type(app.runtime).__name__\n"
    )
    config = RailsConfig(
        config_path=str(config_path),
        models=[],
    )

    rails = LLMRails(config)

    assert getattr(rails, "runtime_seen_by_init") == "RuntimeV1_0"
