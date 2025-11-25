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

import pytest

from nemoguardrails.rails.llm.config import Model


def test_explicit_model_param():
    """Test model specified directly via the model parameter."""
    model = Model(type="main", engine="test_engine", model="test_model")
    assert model.model == "test_model"
    assert "model" not in model.parameters


def test_model_in_parameters():
    """Test model specified via parameters dictionary."""
    model = Model(type="main", engine="test_engine", parameters={"model": "test_model"})
    assert model.model == "test_model"
    assert "model" not in model.parameters


def test_model_name_in_parameters():
    """Test model specified via model_name in parameters dictionary."""
    model = Model(type="main", engine="test_engine", parameters={"model_name": "test_model"})
    assert model.model == "test_model"
    assert "model_name" not in model.parameters


def test_model_equivalence():
    """Test that models defined in different ways are considered equivalent."""
    model1 = Model(type="main", engine="test_engine", model="test_model")
    model2 = Model(type="main", engine="test_engine", parameters={"model": "test_model"})
    assert model1 == model2


def test_empty_model_and_parameters():
    """Test that an empty model and parameters dict fails validation."""
    with pytest.raises(ValueError, match="Model name must be specified"):
        Model(type="main", engine="openai", parameters={})


def test_none_model_and_parameters():
    """Test that None model and empty parameters dict fails validation."""
    with pytest.raises(ValueError, match="Model name must be specified"):
        Model(type="main", engine="openai", model=None, parameters={})


def test_none_model_and_none_parameters():
    """Test that None model and None parameters fails validation."""
    with pytest.raises(ValueError):
        Model(type="main", engine="openai", model=None, parameters=None)


def test_model_and_model_name_in_parameters():
    """Test that having both model and model_name in parameters raises an error."""
    with pytest.raises(ValueError, match="Model name must be specified in exactly one place"):
        Model(
            type="main",
            engine="openai",
            model="gpt-4",
            parameters={"model_name": "gpt-3.5-turbo"},
        )


def test_model_and_model_in_parameters():
    """Test that having both model field and model in parameters raises an error."""
    with pytest.raises(ValueError, match="Model name must be specified in exactly one place"):
        Model(
            type="main",
            engine="openai",
            model="gpt-4",
            parameters={"model": "gpt-3.5-turbo"},
        )


def test_empty_string_model():
    """Test that an empty string model fails validation."""
    with pytest.raises(ValueError, match="Model name must be specified"):
        Model(type="main", engine="openai", model="", parameters={})


def test_whitespace_only_model():
    """Test that a whitespace-only model fails validation."""
    with pytest.raises(ValueError, match="Model name must be specified"):
        Model(type="main", engine="openai", model="   ", parameters={})


def test_empty_string_model_name_in_parameters():
    """Test that an empty string model_name in parameters fails validation."""
    with pytest.raises(ValueError, match="Model name must be specified"):
        Model(type="main", engine="openai", parameters={"model_name": ""})


def test_whitespace_only_model_in_parameters():
    """Test that a whitespace-only model in parameters fails validation."""
    with pytest.raises(ValueError, match="Model name must be specified"):
        Model(type="main", engine="openai", parameters={"model": "  "})


def test_model_name_none_in_parameters():
    """Test that None model_name in parameters fails validation."""
    with pytest.raises(ValueError, match="Model name must be specified"):
        Model(type="main", engine="openai", parameters={"model_name": None})
