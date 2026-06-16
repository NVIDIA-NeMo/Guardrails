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
import logging
import os
from unittest.mock import patch

import pytest

from nemoguardrails.rails.llm.config import JailbreakDetectionConfig, RailsConfig


class TestJailbreakDetectionConfig:
    def test_new_configuration_fields(self):
        config = JailbreakDetectionConfig(
            nim_base_url="http://localhost:8000/v1",
            nim_server_endpoint="classify",
            api_key_env_var="MY_API_KEY",
        )

        assert config.nim_base_url == "http://localhost:8000/v1"
        assert config.nim_server_endpoint == "classify"
        assert config.api_key_env_var == "MY_API_KEY"

    def test_default_values(self):
        config = JailbreakDetectionConfig()

        assert config.nim_base_url is None
        assert config.nim_server_endpoint == "classify"  # Default value
        assert config.api_key_env_var is None

    def test_deprecated_field_migration(self):
        """Test that deprecated nim_url and nim_port fields are migrated to nim_base_url."""
        config = JailbreakDetectionConfig(nim_url="localhost", nim_port=8000)

        # The model validator should migrate these to nim_base_url
        assert config.nim_base_url == "http://localhost:8000/v1"
        assert config.nim_url == "localhost"  # Original value preserved
        assert config.nim_port == 8000  # Original value preserved

    def test_deprecated_field_migration_with_string_port(self):
        """Test migration when port is provided as string."""
        config = JailbreakDetectionConfig(nim_url="localhost", nim_port="9000")

        # The model validator should migrate these to nim_base_url
        assert config.nim_base_url == "http://localhost:9000/v1"

    def test_deprecated_field_migration_no_port(self):
        """Test migration when only nim_url is provided (default port should be used)."""
        config = JailbreakDetectionConfig(nim_url="localhost")

        # Should use default port 8000
        assert config.nim_base_url == "http://localhost:8000/v1"

    def test_no_migration_when_nim_base_url_already_set(self):
        """Test that migration doesn't occur when nim_base_url is already set."""
        config = JailbreakDetectionConfig(nim_base_url="http://existing:9999/v1", nim_url="localhost", nim_port=8000)

        # Should not override existing nim_base_url
        assert config.nim_base_url == "http://existing:9999/v1"

    def test_embedding_field_deprecated(self):
        """Test that embedding field defaults to None (deprecated)."""
        config = JailbreakDetectionConfig()
        assert config.embedding is None

    def test_server_endpoint_description_updated(self):
        """Test that server_endpoint description includes model container."""
        config = JailbreakDetectionConfig(server_endpoint="http://localhost:1337/model")
        assert config.server_endpoint == "http://localhost:1337/model"

    def test_configuration_with_all_new_fields(self):
        config = JailbreakDetectionConfig(
            server_endpoint="http://legacy:1337/heuristics",
            nim_base_url="http://nim:8000/v1",
            nim_server_endpoint="custom-classify",
            api_key_env_var="CUSTOM_API_KEY",
            length_per_perplexity_threshold=100.0,
            prefix_suffix_perplexity_threshold=2000.0,
        )

        assert config.server_endpoint == "http://legacy:1337/heuristics"
        assert config.nim_base_url == "http://nim:8000/v1"
        assert config.nim_server_endpoint == "custom-classify"
        assert config.api_key_env_var == "CUSTOM_API_KEY"
        assert config.length_per_perplexity_threshold == 100.0
        assert config.prefix_suffix_perplexity_threshold == 2000.0

    def test_backward_compatibility(self):
        """Test that old configuration still works with migration."""
        # simulate old config format
        config = JailbreakDetectionConfig(
            server_endpoint="http://old-server:1337/heuristics",
            nim_url="old-nim-host",
            nim_port=8888,
            length_per_perplexity_threshold=89.79,
            prefix_suffix_perplexity_threshold=1845.65,
        )

        # legacy fields should work
        assert config.server_endpoint == "http://old-server:1337/heuristics"
        assert config.length_per_perplexity_threshold == 89.79
        assert config.prefix_suffix_perplexity_threshold == 1845.65

        # deprecated fields should be migrated
        assert config.nim_base_url == "http://old-nim-host:8888/v1"

    def test_empty_configuration(self):
        """Test that completely empty config works with defaults."""

        config = JailbreakDetectionConfig()

        assert config.server_endpoint is None
        assert config.nim_base_url is None
        assert config.nim_server_endpoint == "classify"
        assert config.api_key_env_var is None
        assert config.length_per_perplexity_threshold == 89.79
        assert config.prefix_suffix_perplexity_threshold == 1845.65
        assert config.nim_url is None
        assert config.nim_port is None
        assert config.embedding is None

    def test_get_api_key_no_key(self):
        """Check when neither `api_key` nor `api_key_env_var` are provided, auth token is None"""

        config = JailbreakDetectionConfig(
            nim_base_url="http://localhost:8000/v1",
            nim_server_endpoint="classify",
        )

        auth_token = config.get_api_key()
        assert auth_token is None

    def test_get_api_key_api_key(self):
        """Check when both `api_key` and `api_key_env_var` are provided, `api_key` takes precedence"""
        api_key_value = "nvapi-abcdef12345"
        api_key_env_var_name = "CUSTOM_API_KEY"
        api_key_env_var_value = "env-var-nvapi-abcdef12345"

        with patch.dict(os.environ, {api_key_env_var_name: api_key_env_var_value}):
            config = JailbreakDetectionConfig(
                nim_base_url="http://localhost:8000/v1",
                nim_server_endpoint="classify",
                api_key=api_key_value,
                api_key_env_var=api_key_env_var_name,
            )

            auth_token = config.get_api_key()
            assert auth_token == api_key_value

    def test_get_api_key_api_key_env_var(self):
        """Check when only `api_key_env_var` is provided, the env-var value is correctly returned"""
        api_key_env_var_name = "CUSTOM_API_KEY"
        api_key_env_var_value = "env-var-nvapi-abcdef12345"

        with patch.dict(os.environ, {api_key_env_var_name: api_key_env_var_value}):
            config = JailbreakDetectionConfig(
                nim_base_url="http://localhost:8000/v1",
                nim_server_endpoint="classify",
                api_key_env_var=api_key_env_var_name,
            )

            auth_token = config.get_api_key()
            assert auth_token == api_key_env_var_value

    def test_get_api_key_api_key_env_var_not_set(self):
        """Check configuring an `api_key_env_var` that isn't set in the shell returns None"""
        api_key_env_var_name = "CUSTOM_API_KEY"

        with patch.dict(os.environ, {}):
            config = JailbreakDetectionConfig(
                nim_base_url="http://localhost:8000/v1",
                nim_server_endpoint="classify",
                api_key_env_var=api_key_env_var_name,
            )

            auth_token = config.get_api_key()
            assert auth_token is None

    def test_negative_length_per_perplexity_threshold_raises(self):
        """Threshold <= 0 should raise ValueError."""
        with pytest.raises(ValueError, match="greater than 0"):
            JailbreakDetectionConfig(length_per_perplexity_threshold=-1.0)

    def test_negative_prefix_suffix_perplexity_threshold_raises(self):
        """Threshold <= 0 should raise ValueError."""
        with pytest.raises(ValueError, match="greater than 0"):
            JailbreakDetectionConfig(prefix_suffix_perplexity_threshold=0)

    def test_invalid_nim_base_url_raises(self):
        """nim_base_url without http(s) scheme should raise ValueError."""
        with pytest.raises(ValueError, match="nim_base_url must start with"):
            JailbreakDetectionConfig(nim_base_url="ftp://localhost:8000/v1")

    def test_invalid_server_endpoint_raises(self):
        """server_endpoint without http(s) scheme should raise ValueError."""
        with pytest.raises(ValueError, match="server_endpoint must start with"):
            JailbreakDetectionConfig(server_endpoint="localhost:1337/model")

    def test_valid_urls_accepted(self):
        """Valid http and https URLs should be accepted."""
        config = JailbreakDetectionConfig(
            nim_base_url="https://nim.example.com/v1",
            server_endpoint="http://localhost:1337/model",
        )
        assert config.nim_base_url == "https://nim.example.com/v1"
        assert config.server_endpoint == "http://localhost:1337/model"


def _make_rails_config(**kwargs):
    """Helper to build a RailsConfig with minimal required fields."""
    defaults = {
        "models": [{"type": "main", "engine": "openai", "model": "gpt-3.5-turbo"}],
    }
    defaults.update(kwargs)
    return RailsConfig(**defaults)


class TestJailbreakDetectionCrossValidation:
    def test_model_flow_with_nim_url_but_no_endpoint_raises(self):
        """nim_base_url set but nim_server_endpoint empty should raise."""
        with pytest.raises(Exception, match="nim_server_endpoint is empty"):
            _make_rails_config(
                rails={
                    "input": {"flows": ["jailbreak detection model"]},
                    "config": {
                        "jailbreak_detection": {
                            "nim_base_url": "http://localhost:8000/v1",
                            "nim_server_endpoint": "",
                        }
                    },
                },
            )

    def test_model_flow_with_no_endpoints_warns(self, caplog):
        """No nim_base_url or server_endpoint should warn about local fallback."""
        with caplog.at_level(logging.WARNING):
            _make_rails_config(
                rails={
                    "input": {"flows": ["jailbreak detection model"]},
                    "config": {"jailbreak_detection": {}},
                },
            )
        assert "No endpoint configured for jailbreak detection model" in caplog.text

    def test_heuristics_flow_with_no_server_endpoint_warns(self, caplog):
        """No server_endpoint for heuristics flow should warn."""
        with caplog.at_level(logging.WARNING):
            _make_rails_config(
                rails={
                    "input": {"flows": ["jailbreak detection heuristics"]},
                    "config": {"jailbreak_detection": {}},
                },
            )
        assert "No server_endpoint configured for jailbreak detection heuristics" in caplog.text

    def test_jailbreak_config_present_but_no_flow_warns(self, caplog):
        """Orphaned jailbreak_detection config should warn."""
        with caplog.at_level(logging.WARNING):
            _make_rails_config(
                rails={
                    "config": {
                        "jailbreak_detection": {
                            "nim_base_url": "http://localhost:8000/v1",
                        }
                    },
                },
            )
        assert "no jailbreak detection flow is enabled" in caplog.text

    def test_model_flow_with_nim_fully_configured_passes(self, caplog):
        """Fully configured NIM-based model flow should produce no warnings."""
        with caplog.at_level(logging.WARNING):
            config = _make_rails_config(
                rails={
                    "input": {"flows": ["jailbreak detection model"]},
                    "config": {
                        "jailbreak_detection": {
                            "nim_base_url": "http://localhost:8000/v1",
                            "nim_server_endpoint": "classify",
                        }
                    },
                },
            )
        assert "jailbreak" not in caplog.text.lower()
        assert config.rails.config.jailbreak_detection.nim_base_url == "http://localhost:8000/v1"

    def test_model_flow_with_deprecated_nim_url_no_spurious_warning(self, caplog):
        """Deprecated nim_url/nim_port should not trigger 'no endpoint' warning."""
        with caplog.at_level(logging.WARNING):
            config = _make_rails_config(
                rails={
                    "input": {"flows": ["jailbreak detection model"]},
                    "config": {
                        "jailbreak_detection": {
                            "nim_url": "localhost",
                            "nim_port": 8000,
                        }
                    },
                },
            )
        assert "No endpoint configured" not in caplog.text
        # Verify migration happened
        assert config.rails.config.jailbreak_detection.nim_base_url == "http://localhost:8000/v1"

    def test_model_flow_with_server_endpoint_passes(self, caplog):
        """Model flow with server_endpoint (no NIM) should pass without warning."""
        with caplog.at_level(logging.WARNING):
            _make_rails_config(
                rails={
                    "input": {"flows": ["jailbreak detection model"]},
                    "config": {
                        "jailbreak_detection": {
                            "server_endpoint": "http://localhost:1337/model",
                        }
                    },
                },
            )
        assert "jailbreak" not in caplog.text.lower()

    def test_heuristics_flow_with_server_endpoint_passes(self, caplog):
        """Heuristics flow with server_endpoint should pass without warning."""
        with caplog.at_level(logging.WARNING):
            _make_rails_config(
                rails={
                    "input": {"flows": ["jailbreak detection heuristics"]},
                    "config": {
                        "jailbreak_detection": {
                            "server_endpoint": "http://localhost:1337/heuristics",
                        }
                    },
                },
            )
        assert "jailbreak" not in caplog.text.lower()

    def test_model_flow_deprecated_nim_url_empty_server_endpoint_raises(self):
        """Deprecated nim_url with empty nim_server_endpoint should raise."""
        with pytest.raises(Exception, match="nim_server_endpoint is empty"):
            _make_rails_config(
                rails={
                    "input": {"flows": ["jailbreak detection model"]},
                    "config": {
                        "jailbreak_detection": {
                            "nim_url": "localhost",
                            "nim_server_endpoint": "",
                        }
                    },
                },
            )

    def test_model_flow_nim_port_only_warns(self, caplog):
        """nim_port alone (no nim_url or nim_base_url) should warn about local fallback."""
        with caplog.at_level(logging.WARNING):
            _make_rails_config(
                rails={
                    "input": {"flows": ["jailbreak detection model"]},
                    "config": {
                        "jailbreak_detection": {
                            "nim_port": 9000,
                        }
                    },
                },
            )
        assert "No endpoint configured for jailbreak detection model" in caplog.text

    def test_both_flows_nim_only_warns_heuristics(self, caplog):
        """Both flows with only NIM configured should warn for heuristics only."""
        with caplog.at_level(logging.WARNING):
            _make_rails_config(
                rails={
                    "input": {
                        "flows": [
                            "jailbreak detection model",
                            "jailbreak detection heuristics",
                        ]
                    },
                    "config": {
                        "jailbreak_detection": {
                            "nim_base_url": "http://localhost:8000/v1",
                        }
                    },
                },
            )
        assert "No endpoint configured for jailbreak detection model" not in caplog.text
        assert "No server_endpoint configured for jailbreak detection heuristics" in caplog.text

    def test_both_flows_server_endpoint_only_passes(self, caplog):
        """Both flows with server_endpoint should pass without warnings."""
        with caplog.at_level(logging.WARNING):
            _make_rails_config(
                rails={
                    "input": {
                        "flows": [
                            "jailbreak detection model",
                            "jailbreak detection heuristics",
                        ]
                    },
                    "config": {
                        "jailbreak_detection": {
                            "server_endpoint": "http://localhost:1337/model",
                        }
                    },
                },
            )
        assert "jailbreak" not in caplog.text.lower()

    def test_explicit_null_jailbreak_detection_config(self, caplog):
        """Explicit None for jailbreak_detection should not raise AttributeError."""
        with caplog.at_level(logging.WARNING):
            _make_rails_config(
                rails={
                    "input": {"flows": ["jailbreak detection model"]},
                    "config": {"jailbreak_detection": None},
                },
            )
        assert "No endpoint configured for jailbreak detection model" in caplog.text

    def test_no_jailbreak_config_no_flow_no_warnings(self, caplog):
        """Default config with no jailbreak config or flows should produce no warnings."""
        with caplog.at_level(logging.WARNING):
            _make_rails_config()
        assert "jailbreak" not in caplog.text.lower()
