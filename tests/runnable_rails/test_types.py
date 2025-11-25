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

"""Tests for RunnableRails type annotations and schema methods."""

from typing import Any, Dict, Union

from pydantic import BaseModel, ConfigDict

from nemoguardrails import RailsConfig
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails


def test_input_type_property():
    """Test that InputType property is properly defined."""
    config = RailsConfig.from_content(config={"models": []})
    rails = RunnableRails(config)

    assert hasattr(rails, "InputType")
    assert rails.InputType == Any


def test_output_type_property():
    """Test that OutputType property is properly defined."""
    config = RailsConfig.from_content(config={"models": []})
    rails = RunnableRails(config)

    assert hasattr(rails, "OutputType")
    assert rails.OutputType == Any


def test_get_name_method():
    """Test that get_name() returns correct name with optional suffix."""
    config = RailsConfig.from_content(config={"models": []})
    rails = RunnableRails(config)

    assert rails.get_name() == "RunnableRails"
    assert rails.get_name("Input") == "RunnableRailsInput"


class RailsInputSchema(BaseModel):
    """Test input schema model."""

    model_config = ConfigDict(extra="allow")

    input: Union[str, Dict[str, Any]]


class RailsOutputSchema(BaseModel):
    """Test output schema model."""

    model_config = ConfigDict(extra="allow")

    output: Union[str, Dict[str, Any]]


def test_schema_methods_exist():
    """Test that schema methods exist and return valid schemas."""
    config = RailsConfig.from_content(config={"models": []})
    rails = RunnableRails(config)

    # input_schema and output_schema should exist (from base class)
    # and return valid Pydantic models
    input_schema = rails.input_schema
    output_schema = rails.output_schema

    assert hasattr(input_schema, "__fields__") or hasattr(input_schema, "model_fields")
    assert hasattr(output_schema, "__fields__") or hasattr(output_schema, "model_fields")

    config_schema = rails.config_schema()
    assert hasattr(config_schema, "__fields__") or hasattr(config_schema, "model_fields")
