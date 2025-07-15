# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from nemoguardrails.rails.llm.options import GenerationOptions, GenerationRailsOptions


def test_generation_options_initialization():
    """Test GenerationOptions initialization."""

    options = GenerationOptions()
    assert options.rails is not None
    assert isinstance(options.rails, GenerationRailsOptions)
    assert options.rails.input is True
    assert options.rails.output is True
    assert options.rails.retrieval is True
    assert options.rails.dialog is True

    # rails as list
    options = GenerationOptions(rails=["input", "output"])
    assert isinstance(options.rails, GenerationRailsOptions)
    assert options.rails.input is True
    assert options.rails.output is True
    assert options.rails.retrieval is False
    assert options.rails.dialog is False

    # rails as dict
    options = GenerationOptions(rails={"input": True, "output": False})
    assert isinstance(options.rails, GenerationRailsOptions)
    assert options.rails.input is True
    assert options.rails.output is False


def test_generation_options_validation():
    """Test GenerationOptions validation."""

    # valid rails list
    values = {"rails": ["input", "output"]}
    result = GenerationOptions.check_fields(values)
    assert result == values
    assert isinstance(result["rails"], dict)
    assert result["rails"]["input"] is True
    assert result["rails"]["output"] is True
    assert result["rails"]["dialog"] is False
    assert result["rails"]["retrieval"] is False

    # valid rails dict
    values = {"rails": {"input": True, "output": False}}
    result = GenerationOptions.check_fields(values)
    assert result == values
    assert isinstance(result["rails"], dict)
    assert result["rails"]["input"] is True
    assert result["rails"]["output"] is False

    # invalid rails type
    values = {"rails": 123}
    with pytest.raises(ValueError):
        GenerationOptions(**values)


def test_generation_options_log():
    """Test GenerationOptions log handling."""

    options = GenerationOptions(log={"activated_rails": True, "llm_calls": True})
    assert options.log is not None
    assert options.log.activated_rails is True
    assert options.log.llm_calls is True

    with pytest.raises(ValueError):
        GenerationOptions(log="invalid")


def test_generation_options_serialization():
    """Test GenerationOptions serialization."""

    options = GenerationOptions(
        rails={"input": True, "output": False},
        log={"activated_rails": True, "llm_calls": True},
    )

    # serialization to dict
    options_dict = options.dict()
    assert isinstance(options_dict, dict)
    assert "rails" in options_dict
    assert "log" in options_dict
    assert isinstance(options_dict["rails"], dict)
    assert isinstance(options_dict["log"], dict)
    assert options_dict["rails"]["input"] is True
    assert options_dict["rails"]["output"] is False
    assert options_dict["log"]["activated_rails"] is True
    assert options_dict["log"]["llm_calls"] is True

    # serialization to JSON
    options_json = options.json()
    assert isinstance(options_json, str)
    assert '"input":true' in options_json
    assert '"output":false' in options_json
    assert '"activated_rails":true' in options_json
    assert '"llm_calls":true' in options_json
