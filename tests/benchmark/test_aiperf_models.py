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

"""
Tests for AIPerf configuration models.
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from benchmark.aiperf.aiperf_models import AIPerfConfig, BaseConfig


class TestBaseConfig:
    """Test the BaseConfig model."""

    def test_base_config_minimal_valid(self):
        """Test creating BaseConfig with minimal required fields."""
        config = BaseConfig(
            model="test-model",
            tokenizer="test-tokenizer",
            url="http://localhost:8000",
            warmup_request_count=10,
            benchmark_duration=60,
            concurrency=5,
        )
        assert config.model == "test-model"
        assert config.tokenizer == "test-tokenizer"
        assert config.url == "http://localhost:8000"
        assert config.endpoint == "/v1/chat/completions"  # Default
        assert config.endpoint_type == "chat"  # Default
        assert config.warmup_request_count == 10
        assert config.benchmark_duration == 60
        assert config.concurrency == 5
        assert config.request_rate_mode == "constant"  # Default

    def test_base_config_with_all_fields(self):
        """Test creating BaseConfig with all fields specified."""
        config = BaseConfig(
            model="test-model",
            tokenizer="test-tokenizer",
            url="http://localhost:8000",
            endpoint="/v1/completions",
            endpoint_type="completions",
            api_key_env_var="AIPERF_API_KEY",
            warmup_request_count=10,
            benchmark_duration=60,
            concurrency=5,
            request_rate=2.5,
            request_rate_mode="poisson",
            random_seed=42,
            prompt_input_tokens_mean=100,
            prompt_input_tokens_stddev=10,
            prompt_output_tokens_mean=50,
            prompt_output_tokens_stddev=5,
        )
        assert config.model == "test-model"
        assert config.tokenizer == "test-tokenizer"
        assert config.endpoint == "/v1/completions"
        assert config.endpoint_type == "completions"
        assert config.api_key_env_var == "AIPERF_API_KEY"
        assert config.request_rate == 2.5
        assert config.request_rate_mode == "poisson"
        assert config.random_seed == 42
        assert config.prompt_input_tokens_mean == 100
        assert config.prompt_input_tokens_stddev == 10
        assert config.prompt_output_tokens_mean == 50
        assert config.prompt_output_tokens_stddev == 5

    def test_base_config_missing_required_fields(self):
        """Test that missing required fields raise validation error."""
        with pytest.raises(ValidationError) as exc_info:
            BaseConfig(
                model="test-model",
                url="http://localhost:8000",
                # Missing warmup_request_count, benchmark_duration, concurrency
            )
        errors = exc_info.value.errors()
        error_fields = {err["loc"][0] for err in errors}
        assert "warmup_request_count" in error_fields
        assert "benchmark_duration" in error_fields
        assert "concurrency" in error_fields

    def test_base_config_invalid_endpoint_type(self):
        """Test that invalid endpoint_type raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            BaseConfig(
                model="test-model",
                url="http://localhost:8000",
                endpoint_type="invalid",  # Must be "chat" or "completions"
                warmup_request_count=10,
                benchmark_duration=60,
                concurrency=5,
            )
        errors = exc_info.value.errors()
        assert any("endpoint_type" in str(err["loc"]) for err in errors)

    def test_base_config_invalid_request_rate_mode(self):
        """Test that invalid request_rate_mode raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            BaseConfig(
                model="test-model",
                url="http://localhost:8000",
                request_rate_mode="invalid",  # Must be "constant" or "poisson"
                warmup_request_count=10,
                benchmark_duration=60,
                concurrency=5,
            )
        errors = exc_info.value.errors()
        assert any("request_rate_mode" in str(err["loc"]) for err in errors)


class TestAIPerfConfig:
    """Test the AIPerfConfig model."""

    @pytest.fixture(autouse=True)
    def valid_base_config(self) -> BaseConfig:
        """Helper to get a valid base config dictionary."""
        return BaseConfig(
            model="test-model",
            url="http://localhost:8000",
            warmup_request_count=10,
            benchmark_duration=60,
            concurrency=5,
        )

    def test_aiperf_config_minimal_valid(self, valid_base_config):
        """Test creating AIPerfConfig with minimal required fields."""
        config = AIPerfConfig(base_config=valid_base_config)

        assert config.batch_name == "benchmark"  # Default
        assert config.output_base_dir == "aiperf_results"  # Default
        assert config.base_config.model == "test-model"
        assert config.sweeps is None

    def test_aiperf_config_with_custom_fields(self, valid_base_config):
        """Test creating AIPerfConfig with custom batch_name and output_dir."""
        config = AIPerfConfig(
            batch_name="my_benchmark",
            output_base_dir="custom_results",
            base_config=valid_base_config,
        )
        assert config.batch_name == "my_benchmark"
        assert config.output_base_dir == "custom_results"
        assert config.base_config.model == "test-model"

    def test_aiperf_config_with_valid_sweeps_int(self, valid_base_config):
        """Test creating AIPerfConfig with valid integer sweeps."""

        sweeps: dict[str, list[int]] = {
            "concurrency": [10, 20, 30],
            "warmup_request_count": [5, 10, 15],
        }
        config = AIPerfConfig(
            base_config=valid_base_config,
            sweeps=sweeps,
        )
        assert config.sweeps == sweeps

    def test_aiperf_config_with_valid_sweeps_str(self, valid_base_config):
        """Test creating AIPerfConfig with valid string sweeps."""
        config = AIPerfConfig(
            base_config=valid_base_config,
            sweeps={
                "model": ["model-a", "model-b", "model-c"],
                "endpoint": ["/v1/chat", "/v1/completions"],
            },
        )
        assert config.sweeps == {
            "model": ["model-a", "model-b", "model-c"],
            "endpoint": ["/v1/chat", "/v1/completions"],
        }

    def test_aiperf_config_with_valid_sweeps_mixed(self, valid_base_config):
        """Test creating AIPerfConfig with mixed int and string sweeps."""
        sweeps = {
            "concurrency": [10, 20],
            "model": ["model-a", "model-b"],
        }
        config = AIPerfConfig(
            base_config=valid_base_config,
            sweeps=sweeps,
        )
        assert config.sweeps == sweeps

    def test_aiperf_config_sweep_invalid_key(self, valid_base_config):
        """Test that invalid sweep key raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            AIPerfConfig(
                base_config=valid_base_config,
                sweeps={
                    "invalid_field": [1, 2, 3],
                },
            )
        error_msg = str(exc_info.value)
        assert "invalid_field" in error_msg
        assert "not a valid BaseConfig field" in error_msg

    def test_aiperf_config_sweep_invalid_value_type_float(self, valid_base_config):
        """Test that float values in sweeps raise validation error."""
        with pytest.raises(ValidationError) as exc_info:
            AIPerfConfig(
                base_config=valid_base_config,
                sweeps={
                    "concurrency": [10, 20.5, 30],  # Float not allowed
                },
            )
        error_msg = str(exc_info.value)
        # Pydantic catches this during type validation
        assert "sweeps.concurrency" in error_msg
        assert "must be int or str" in error_msg or "int_from_float" in error_msg

    def test_aiperf_config_sweep_invalid_value_type_dict(self, valid_base_config):
        """Test that dict values in sweeps raise validation error."""
        with pytest.raises(ValidationError) as exc_info:
            AIPerfConfig(
                base_config=valid_base_config,
                sweeps={
                    "concurrency": [10, {"value": 20}, 30],  # Dict not allowed
                },
            )
        error_msg = str(exc_info.value)
        # Pydantic catches this during type validation
        assert "sweeps.concurrency" in error_msg
        assert "must be int or str" in error_msg or "int_type" in error_msg or "string_type" in error_msg

    def test_aiperf_config_sweep_invalid_value_type_list(self, valid_base_config):
        """Test that list values in sweeps raise validation error."""
        with pytest.raises(ValidationError) as exc_info:
            AIPerfConfig(
                base_config=valid_base_config,
                sweeps={
                    "concurrency": [10, [20, 30]],  # List not allowed
                },
            )
        error_msg = str(exc_info.value)
        # Pydantic catches this during type validation
        assert "sweeps.concurrency" in error_msg
        assert "must be int or str" in error_msg or "int_type" in error_msg or "string_type" in error_msg

    def test_aiperf_config_sweep_empty_list(self, valid_base_config):
        """Test that empty sweep list raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            AIPerfConfig(
                base_config=valid_base_config,
                sweeps={
                    "concurrency": [],
                },
            )
        error_msg = str(exc_info.value)
        assert "cannot be empty" in error_msg

    def test_aiperf_config_sweep_not_list(self, valid_base_config):
        """Test that non-list sweep value raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            AIPerfConfig(
                base_config=valid_base_config,
                sweeps={
                    "concurrency": 10,  # Should be a list
                },
            )
        error_msg = str(exc_info.value)
        # Pydantic catches this during type validation
        assert "sweeps.concurrency" in error_msg
        assert "must be a list" in error_msg or "list_type" in error_msg

    def test_aiperf_config_multiple_invalid_sweep_keys(self, valid_base_config):
        """Test that multiple invalid sweep keys are all reported."""
        with pytest.raises(ValidationError) as exc_info:
            AIPerfConfig(
                base_config=valid_base_config,
                sweeps={
                    "invalid_field_1": [1, 2],
                    "invalid_field_2": [3, 4],
                },
            )
        error_msg = str(exc_info.value)
        # At least one of the invalid fields should be mentioned
        assert "invalid_field" in error_msg

    def test_aiperf_config_get_output_base_path(self, valid_base_config):
        """Test get_output_base_path method."""
        config = AIPerfConfig(output_base_dir="custom_results", base_config=valid_base_config)
        path = config.get_output_base_path()
        assert isinstance(path, Path)
        assert str(path) == "custom_results"

    def test_aiperf_config_get_output_base_path_default(self, valid_base_config):
        """Test get_output_base_path method with default output_base_dir."""
        config = AIPerfConfig(base_config=valid_base_config)
        path = config.get_output_base_path()
        assert isinstance(path, Path)
        assert str(path) == "aiperf_results"

    def test_aiperf_config_missing_base_config(self):
        """Test that missing base_config raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            AIPerfConfig()  # Missing required base_config
        errors = exc_info.value.errors()
        assert any("base_config" in str(err["loc"]) for err in errors)

    def test_aiperf_config_invalid_base_config(self):
        """Test that invalid base_config raises validation error."""
        with pytest.raises(ValidationError):
            AIPerfConfig(
                base_config={
                    "model": "test-model",
                    # Missing required fields
                },
            )

    def test_aiperf_config_all_valid_sweep_keys(self, valid_base_config):
        """Test sweeps with all valid BaseConfig field names."""
        config = AIPerfConfig(
            base_config=valid_base_config,
            sweeps={
                "model": ["model-a", "model-b"],
                "url": ["http://localhost:8000", "http://localhost:8001"],
                "endpoint": ["/v1/chat", "/v1/completions"],
                "endpoint_type": ["chat", "completions"],
                "warmup_request_count": [5, 10],
                "benchmark_duration": [30, 60],
                "concurrency": [5, 10],
                "request_rate_mode": ["constant", "poisson"],
                "random_seed": [42, 123],
                "prompt_input_tokens_mean": [100, 200],
                "prompt_input_tokens_stddev": [10, 20],
                "prompt_output_tokens_mean": [50, 100],
                "prompt_output_tokens_stddev": [5, 10],
            },
        )
        # All sweeps should be accepted
        assert len(config.sweeps) == 13

    def test_sweeps_none(self, valid_base_config):
        """Test sweeps set to None don't raise Exception."""
        config = AIPerfConfig(
            base_config=valid_base_config,
            sweeps=None,
        )

    def test_sweeps_not_list_raises(self, valid_base_config):
        """Test sweeps set to None don't raise Exception."""
        with pytest.raises(ValueError, match="Input should be a valid list"):
            config = AIPerfConfig(
                base_config=valid_base_config,
                sweeps={"benchmark_duration": 1},
            )

    def test_sweeps_empty_list_raises(self, valid_base_config):
        """Test sweeps set to None don't raise Exception."""
        with pytest.raises(
            ValueError,
            match="Sweep parameter 'concurrency' cannot be empty",
        ):
            config = AIPerfConfig(
                base_config=valid_base_config,
                sweeps={"concurrency": []},
            )
