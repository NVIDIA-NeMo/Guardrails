#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
Tests for Locust load test configuration models.
"""

import pytest
from pydantic import ValidationError

from benchmark.locust.locust_models import LocustConfig, LocustSweepConfig


class TestLocustConfig:
    """Test the LocustConfig model."""

    def test_config_minimal_valid_with_defaults(self):
        """Test creating LocustConfig with minimal required fields and verify all defaults."""
        config = LocustConfig(
            config_id="test-config",
            model="test-model",
        )

        # Verify required fields
        assert config.config_id == "test-config"
        assert config.model == "test-model"

        # Verify all defaults
        assert config.host == "http://localhost:8000"
        assert config.users == 256
        assert config.spawn_rate == 10
        assert config.run_time == 60
        assert config.message == "Hello, what can you do?"
        assert config.headless is True
        assert config.output_base_dir == "locust_results"

    def test_config_with_all_fields(self):
        """Test creating LocustBaseConfig with all fields specified."""
        config = LocustConfig(
            host="http://example.com:9000",
            config_id="my-config",
            model="my-model",
            users=100,
            spawn_rate=5.5,
            run_time=120,
            message="Custom message",
            headless=True,
            output_base_dir="/tmp/locust",
        )
        assert config.host == "http://example.com:9000"
        assert config.config_id == "my-config"
        assert config.model == "my-model"
        assert config.users == 100
        assert config.spawn_rate == 5.5
        assert config.run_time == 120
        assert config.message == "Custom message"
        assert config.headless is True
        assert config.output_base_dir == "/tmp/locust"

    def test_config_extra_fields_forbidden(self):
        """Test that extra/unknown fields raise validation error."""
        with pytest.raises(ValidationError) as exc_info:
            LocustConfig(
                config_id="test-config",
                model="test-model",
                spawn_rats=5,  # typo of spawn_rate
            )
        error_msg = str(exc_info.value)
        assert "spawn_rats" in error_msg

    def test_config_missing_required_fields(self):
        """Test that missing required fields raise validation error."""
        with pytest.raises(ValidationError) as exc_info:
            LocustConfig(
                host="http://localhost:8000",
                # Missing config_id and model
            )
        errors = exc_info.value.errors()
        error_fields = {err["loc"][0] for err in errors}
        assert "config_id" in error_fields
        assert "model" in error_fields

    def test_config_host_without_protocol(self):
        """Test that host without http:// or https:// raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            LocustConfig(
                host="localhost:8000",  # Missing http://
                config_id="test-config",
                model="test-model",
            )
        error_msg = str(exc_info.value)
        assert "Host must start with http:// or https://" in error_msg

    def test_config_host_with_https(self):
        """Test that host with https:// is valid."""
        config = LocustConfig(
            host="https://secure.example.com",
            config_id="test-config",
            model="test-model",
        )
        assert config.host == "https://secure.example.com"

    def test_config_host_trailing_slash_removed(self):
        """Test that trailing slash in host is removed."""
        config = LocustConfig(
            host="http://localhost:8000/",
            config_id="test-config",
            model="test-model",
        )
        assert config.host == "http://localhost:8000"

    def test_config_host_multiple_trailing_slashes(self):
        """Test that multiple trailing slashes are removed."""
        config = LocustConfig(
            host="http://localhost:8000///",
            config_id="test-config",
            model="test-model",
        )
        assert config.host == "http://localhost:8000"


class TestLocustConfigHelpers:
    """Test helper methods on LocustConfig model."""

    def test_locust_config_with_dict(self):
        """Test creating LocustConfig with dict base_config."""
        config = LocustConfig(
            **{
                "config_id": "test-config",
                "model": "test-model",
                "users": 100,
            }
        )
        assert config.config_id == "test-config"
        assert config.model == "test-model"
        assert config.users == 100


class TestLocustSweepConfig:
    """Test the LocustSweepConfig model."""

    BASE = {"config_id": "test-config", "model": "test-model"}

    def test_flat_config_is_read_as_a_single_run(self):
        """Config files written before sweeps existed keep working."""
        config = LocustSweepConfig(**{**self.BASE, "host": "http://localhost:9000", "users": 64})

        assert config.sweeps is None
        assert config.base_config.config_id == "test-config"
        assert config.base_config.users == 64
        assert config.expand() == [("", config.base_config)]

    def test_nested_config_with_sweep(self):
        """The nested form mirrors the AIPerf batch layout."""
        config = LocustSweepConfig(
            batch_name="sweep_concurrency",
            base_config=self.BASE,
            sweeps={"users": [1, 2, 4]},
        )

        assert config.batch_name == "sweep_concurrency"
        assert [label for label, _ in config.expand()] == ["users-1", "users-2", "users-4"]
        assert [run.users for _, run in config.expand()] == [1, 2, 4]

    def test_flat_config_may_carry_a_sweep(self):
        """A sweep can be added to an existing flat config without restructuring it."""
        config = LocustSweepConfig(**{**self.BASE, "sweeps": {"users": [8, 16]}})

        assert [run.users for _, run in config.expand()] == [8, 16]

    def test_base_config_supplies_values_not_swept(self):
        """Sweeping one field leaves the rest of the base configuration intact."""
        config = LocustSweepConfig(
            base_config={**self.BASE, "message": "hello", "run_time": 120},
            sweeps={"users": [1, 2]},
        )

        for _, run in config.expand():
            assert run.message == "hello"
            assert run.run_time == 120

    def test_multiple_sweeps_run_the_cartesian_product(self):
        """Several swept parameters expand to every combination, as AIPerf does."""
        config = LocustSweepConfig(
            base_config=self.BASE,
            sweeps={"users": [1, 2], "spawn_rate": [4, 8]},
        )

        assert [label for label, _ in config.expand()] == [
            "spawn_rate-4_users-1",
            "spawn_rate-4_users-2",
            "spawn_rate-8_users-1",
            "spawn_rate-8_users-2",
        ]

    def test_output_base_dir_defaults_to_the_base_config(self):
        """A flat config's output directory is not lost when it is wrapped."""
        config = LocustSweepConfig(**{**self.BASE, "output_base_dir": "somewhere"})

        assert config.output_base_dir == "somewhere"

    def test_sweep_over_unknown_field_rejected(self):
        """A swept parameter that is not a LocustConfig field cannot be applied."""
        with pytest.raises(ValidationError, match="not LocustConfig fields"):
            LocustSweepConfig(base_config=self.BASE, sweeps={"nonsense": [1]})

    def test_sweep_with_no_values_rejected(self):
        """An empty sweep list would silently run nothing."""
        with pytest.raises(ValidationError, match="no values"):
            LocustSweepConfig(base_config=self.BASE, sweeps={"users": []})

    def test_swept_values_are_validated(self):
        """Field constraints still apply to values coming from a sweep."""
        config = LocustSweepConfig(base_config=self.BASE, sweeps={"users": [0]})

        with pytest.raises(ValidationError):
            config.expand()

    def test_extra_top_level_fields_forbidden(self):
        """A misspelled batch key should not be silently ignored."""
        with pytest.raises(ValidationError):
            LocustSweepConfig(base_config=self.BASE, bogus=1)
