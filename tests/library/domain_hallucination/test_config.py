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


class TestLayer2Config:
    """Test Layer2Config dataclass."""

    def test_layer2_config_defaults(self):
        """Test Layer2Config default values."""
        cfg = config.Layer2Config()
        assert cfg.enabled is False
        assert cfg.temperature == 0.1
        assert cfg.max_tokens == 1024
        assert cfg.dns_timeout == 5.0

    def test_layer2_config_custom_values(self):
        """Test Layer2Config with custom values."""
        cfg = config.Layer2Config(
            enabled=True,
            temperature=0.2,
            max_tokens=2048,
            dns_timeout=10.0,
        )
        assert cfg.enabled is True
        assert cfg.temperature == 0.2
        assert cfg.max_tokens == 2048
        assert cfg.dns_timeout == 10.0


class TestConfigOptionalTypes:
    """Test Optional type annotations in configuration."""

    def test_config_layer1_optional_none(self):
        """Test that layer1 can be None initially."""
        # layer1=None should be accepted
        cfg = config.DomainHallucinationGuardConfig(layer1=None)
        # After post_init, should be initialized
        assert cfg.layer1 is not None
        assert isinstance(cfg.layer1, config.Layer1Config)

    def test_config_layer2_optional_none(self):
        """Test that layer2 can be None initially."""
        # layer2=None should be accepted
        cfg = config.DomainHallucinationGuardConfig(layer2=None)
        # After post_init, should be initialized
        assert cfg.layer2 is not None
        assert isinstance(cfg.layer2, config.Layer2Config)

    def test_config_both_layers_none(self):
        """Test that both layers can be None and will be initialized."""
        cfg = config.DomainHallucinationGuardConfig(layer1=None, layer2=None)
        assert cfg.layer1 is not None
        assert cfg.layer2 is not None
        assert isinstance(cfg.layer1, config.Layer1Config)
        assert isinstance(cfg.layer2, config.Layer2Config)

    def test_config_preserves_provided_layers(self):
        """Test that provided layer configs are preserved."""
        layer1 = config.Layer1Config(temperature=0.5)
        layer2 = config.Layer2Config(dns_timeout=15.0)

        cfg = config.DomainHallucinationGuardConfig(
            layer1=layer1,
            layer2=layer2,
        )

        # Should preserve the provided instances
        assert cfg.layer1 is layer1
        assert cfg.layer2 is layer2
        assert cfg.layer1.temperature == 0.5
        assert cfg.layer2.dns_timeout == 15.0

    def test_config_initialization_with_custom_layer1_none_layer2(self):
        """Test config with custom layer1 and None layer2."""
        layer1 = config.Layer1Config(temperature=0.3)
        cfg = config.DomainHallucinationGuardConfig(layer1=layer1, layer2=None)

        assert cfg.layer1 is layer1
        assert cfg.layer2 is not None
        assert cfg.layer1.temperature == 0.3

    def test_config_type_hints_are_optional(self):
        """Test that type hints are correctly marked as Optional."""
        # This is a runtime test to verify the dataclass fields accept None
        import inspect

        # Check if the __init__ signature accepts layer1=None and layer2=None
        sig = inspect.signature(config.DomainHallucinationGuardConfig.__init__)
        assert sig.parameters["layer1"].default is None
        assert sig.parameters["layer2"].default is None


class TestConfigLayer2Integration:
    """Test Layer2Config integration with main config."""

    def test_config_to_dict_includes_layer2(self):
        """Test that to_dict includes layer2 configuration."""
        cfg = config.DomainHallucinationGuardConfig()
        d = cfg.to_dict()

        assert "layer2" in d
        assert isinstance(d["layer2"], dict)
        assert "enabled" in d["layer2"]
        assert "dns_timeout" in d["layer2"]

    def test_config_to_json_includes_layer2(self):
        """Test that to_json includes layer2 configuration."""
        cfg = config.DomainHallucinationGuardConfig()
        json_str = cfg.to_json()
        parsed = json.loads(json_str)

        assert "layer2" in parsed
        assert "dns_timeout" in parsed["layer2"]

    def test_load_config_with_layer2(self):
        """Test loading config with layer2 settings."""
        config_dict = {
            "layer1": {"enabled": True, "temperature": 0.1},
            "layer2": {"enabled": True, "dns_timeout": 8.0},
            "debug": False,
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_dict, f)
            temp_path = f.name

        try:
            loaded_cfg = config.load_config(temp_path)
            assert loaded_cfg.layer2.enabled is True
            assert loaded_cfg.layer2.dns_timeout == 8.0
        finally:
            Path(temp_path).unlink()
