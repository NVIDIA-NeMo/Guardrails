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

import pytest

from nemoguardrails import LLMRails, RailsConfig

CONFIGS_FOLDER = os.path.join(os.path.dirname(__file__), ".", "test_configs")


def test_combine_configs_engine_mismatch():
    general_config = RailsConfig.from_path(os.path.join(CONFIGS_FOLDER, "general"))
    factcheck_config = RailsConfig.from_path(
        os.path.join(CONFIGS_FOLDER, "fact_checking")
    )

    with pytest.raises(ValueError) as exc_info:
        full_llm_config = general_config + factcheck_config
        assert (
            "Both config files should have the same engine for the same model type"
            in str(exc_info.value)
        )


def test_combine_configs_model_mismatch():
    general_config = RailsConfig.from_path(os.path.join(CONFIGS_FOLDER, "general"))
    prompt_override_config = RailsConfig.from_path(
        os.path.join(CONFIGS_FOLDER, "with_prompt_override")
    )

    with pytest.raises(ValueError) as exc_info:
        full_llm_config = general_config + prompt_override_config
        assert "Both config files should have the same model for the same model" in str(
            exc_info.value
        )


def test_combine_two_configs():
    general_config = RailsConfig.from_path(os.path.join(CONFIGS_FOLDER, "general"))
    input_rails_config = RailsConfig.from_path(
        os.path.join(CONFIGS_FOLDER, "input_rails")
    )

    full_llm_config = general_config + input_rails_config

    assert full_llm_config.models[0].model == "gpt-3.5-turbo-instruct"
    assert (
        full_llm_config.instructions[0].content
        == input_rails_config.instructions[0].content
    )
    assert full_llm_config.rails.input.flows == ["self check input"]


def test_combine_three_configs():
    general_config = RailsConfig.from_path(os.path.join(CONFIGS_FOLDER, "general"))

    input_rails_config = RailsConfig.from_path(
        os.path.join(CONFIGS_FOLDER, "input_rails")
    )
    output_rails_config = RailsConfig.from_path(
        os.path.join(CONFIGS_FOLDER, "output_rails")
    )

    full_llm_config = general_config + input_rails_config + output_rails_config
    assert full_llm_config.rails.input.flows == ["dummy input rail", "self check input"]
    assert full_llm_config.rails.output.flows == [
        "self check output",
        "check blocked terms",
    ]
    assert (
        full_llm_config.instructions[0].content
        == output_rails_config.instructions[0].content
    )
    assert (
        full_llm_config.rails.dialog.single_call
        == output_rails_config.rails.dialog.single_call
    )
