# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for domain hallucination guard configuration."""

import json
import tempfile
from pathlib import Path

from nemoguardrails.library.domain_hallucination import config


class TestLayer1Config:
    """Test Layer1Config dataclass."""

    def test_layer1_config_defaults(self):
        """Test Layer1Config default values."""
        cfg = config.Layer1Config()
        assert cfg.enabled is True
        assert cfg.temperature == 0.1
        assert cfg.max_tokens == 1024

    def test_layer1_config_custom_values(self):
        """Test Layer1Config with custom values."""
        cfg = config.Layer1Config(
            enabled=False,
            temperature=0.5,
            max_tokens=2048,
        )
        assert cfg.enabled is False
        assert cfg.temperature == 0.5
        assert cfg.max_tokens == 2048


class TestDomainHallucinationGuardConfig:
    """Test main configuration class."""

    def test_config_defaults(self):
        """Test default configuration values."""
        cfg = config.DomainHallucinationGuardConfig()
        assert cfg.layer1 is not None
        assert cfg.layer1.enabled is True
        assert cfg.debug is False
        assert cfg.log_level == "INFO"

    def test_config_post_init(self):
        """Test __post_init__ creates default Layer1Config."""
        cfg = config.DomainHallucinationGuardConfig(layer1=None)
        # __post_init__ should create default Layer1Config
        assert cfg.layer1 is not None
        assert cfg.layer1.enabled is True

    def test_config_to_dict(self):
        """Test conversion to dictionary."""
        cfg = config.DomainHallucinationGuardConfig(
            debug=True,
            log_level="DEBUG",
        )
        d = cfg.to_dict()
        assert d["debug"] is True
        assert d["log_level"] == "DEBUG"
        assert "layer1" in d
        assert isinstance(d["layer1"], dict)
        assert d["layer1"]["enabled"] is True

    def test_config_to_json(self):
        """Test conversion to JSON."""
        cfg = config.DomainHallucinationGuardConfig(debug=True)
        json_str = cfg.to_json()
        # Parse and verify
        parsed = json.loads(json_str)
        assert parsed["debug"] is True
        assert "layer1" in parsed

    def test_config_custom_values(self):
        """Test configuration with custom values."""
        layer1 = config.Layer1Config(temperature=0.3)
        cfg = config.DomainHallucinationGuardConfig(
            layer1=layer1,
            debug=True,
            log_level="WARNING",
        )
        assert cfg.layer1.temperature == 0.3
        assert cfg.debug is True
        assert cfg.log_level == "WARNING"


class TestConfigIO:
    """Test configuration loading and saving."""

    def test_load_config_from_file(self):
        """Test loading configuration from JSON file."""
        config_dict = {
            "layer1": {
                "enabled": True,
                "temperature": 0.2,
                "max_tokens": 512,
            },
            "debug": True,
            "log_level": "DEBUG",
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_dict, f)
            temp_path = f.name

        try:
            loaded_cfg = config.load_config(temp_path)
            assert loaded_cfg.debug is True
            assert loaded_cfg.log_level == "DEBUG"
            assert loaded_cfg.layer1.temperature == 0.2
            assert loaded_cfg.layer1.max_tokens == 512
        finally:
            Path(temp_path).unlink()

    def test_load_config_minimal(self):
        """Test loading minimal configuration."""
        config_dict = {}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_dict, f)
            temp_path = f.name

        try:
            loaded_cfg = config.load_config(temp_path)
            # Should use defaults
            assert loaded_cfg.layer1.enabled is True
            assert loaded_cfg.debug is False
        finally:
            Path(temp_path).unlink()

    def test_load_config_partial(self):
        """Test loading partial configuration."""
        config_dict = {
            "debug": True,
            # No layer1 specified, should use defaults
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_dict, f)
            temp_path = f.name

        try:
            loaded_cfg = config.load_config(temp_path)
            assert loaded_cfg.debug is True
            assert loaded_cfg.layer1.enabled is True
        finally:
            Path(temp_path).unlink()


class TestConfigGlobalState:
    """Test global configuration management."""

    def test_get_default_config(self):
        """Test getting default configuration."""
        cfg = config.get_config()
        assert cfg is not None
        assert cfg.layer1 is not None

    def test_set_and_get_config(self):
        """Test setting and getting configuration."""
        original = config.get_config()

        new_cfg = config.DomainHallucinationGuardConfig(
            debug=True,
            log_level="CUSTOM",
        )
        config.set_config(new_cfg)

        retrieved = config.get_config()
        assert retrieved.debug is True
        assert retrieved.log_level == "CUSTOM"

        # Restore original
        config.set_config(original)

    def test_config_singleton_behavior(self):
        """Test that global config acts as singleton."""
        cfg1 = config.get_config()
        cfg2 = config.get_config()
        # Both should reference the same object (default)
        assert cfg1 is cfg2


class TestConfigValidation:
    """Test configuration validation."""

    def test_layer1_config_type_conversion(self):
        """Test type handling in Layer1Config."""
        cfg = config.Layer1Config(
            enabled=True,
            temperature=0.1,
            max_tokens=1024,
        )
        assert isinstance(cfg.enabled, bool)
        assert isinstance(cfg.temperature, float)
        assert isinstance(cfg.max_tokens, int)

    def test_config_json_roundtrip(self):
        """Test JSON serialization and deserialization."""
        original = config.DomainHallucinationGuardConfig(
            debug=True,
            log_level="DEBUG",
        )
        json_str = original.to_json()

        # Parse back
        parsed = json.loads(json_str)
        assert parsed["debug"] is True
        assert parsed["log_level"] == "DEBUG"
