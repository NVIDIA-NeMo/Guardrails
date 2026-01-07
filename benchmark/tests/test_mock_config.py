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
from pydantic import ValidationError

from benchmark.mock_llm_server.config import ModelSettings


class TestAppModelConfig:
    """Test the AppModelConfig Pydantic model."""

    def test_app_model_config_with_defaults(self):
        """Test creating AppModelConfig with default values."""
        config = ModelSettings(
            model="test-model",
            unsafe_text="Unsafe",
            safe_text="Safe",
        )
        # Check defaults
        assert config.unsafe_probability == 0.1
        assert config.latency_min_seconds == 0.1
        assert config.latency_max_seconds == 5
        assert config.latency_mean_seconds == 0.5
        assert config.latency_std_seconds == 0.1

    def test_app_model_config_missing_required_field(self):
        """Test that missing required fields raise validation error."""
        with pytest.raises(ValidationError):
            ModelSettings(  # type: ignore (Test is meant to check missing mandatory field)
                model="test-model",
                unsafe_text="Unsafe",
                # Missing safe_text
            )

    def test_app_model_config_model_serialization(self):
        """Test that AppModelConfig can be serialized to dict."""
        config = ModelSettings(
            model="test-model",
            unsafe_text="Unsafe",
            safe_text="Safe",
        )
        config_dict = config.model_dump()
        assert isinstance(config_dict, dict)
        assert config_dict["model"] == "test-model"
        assert config_dict["safe_text"] == "Safe"
