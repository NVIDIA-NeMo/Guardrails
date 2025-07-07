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

# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import yaml

from nemoguardrails.evaluate.utils_translate import (
    PluginConfigurationError,
    _load_langprovider,
)


class TestLangProviderIntegration:
    """Integration tests for LangProvider functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_config_path = os.path.join(self.temp_dir, "test_translation.yaml")

    def teardown_method(self):
        """Clean up test fixtures."""
        if os.path.exists(self.test_config_path):
            os.remove(self.test_config_path)
        if os.path.exists(self.temp_dir):
            import shutil

            shutil.rmtree(self.temp_dir)

    def create_test_config(self, config_data):
        """Helper method to create test configuration file."""
        with open(self.test_config_path, "w") as f:
            yaml.dump(config_data, f)

    @patch("nemoguardrails.evaluate.utils_translate._load_plugin")
    def test_load_deepl_translator_integration(self, mock_load_plugin):
        """Test loading DeeplTranslator through the utility function."""
        config_data = {
            "langproviders": [
                {"language": "en,ja", "model_type": "remote.DeeplTranslator"}
            ]
        }
        self.create_test_config(config_data)

        # Mock the plugin loader to return a mock DeeplTranslator instance
        mock_provider = MagicMock()
        mock_provider.language = "en,ja"
        mock_provider.source_lang = "en"
        mock_provider.target_lang = "ja"
        mock_load_plugin.return_value = mock_provider

        # Call the function
        result = _load_langprovider(self.test_config_path)

        # Verify the result
        assert result == mock_provider
        assert result.language == "en,ja"
        assert result.source_lang == "en"
        assert result.target_lang == "ja"

        # Verify _load_plugin was called with correct arguments
        mock_load_plugin.assert_called_once_with(
            path="nemoguardrails.evaluate.langproviders.remote.DeeplTranslator",
            config_root={
                "langproviders": {
                    "remote.DeeplTranslator": {
                        "language": "en,ja",
                        "model_type": "remote.DeeplTranslator",
                    }
                }
            },
        )

    @patch("nemoguardrails.evaluate.utils_translate._load_plugin")
    def test_load_local_hf_translator_integration(self, mock_load_plugin):
        """Test loading LocalHFTranslator through the utility function."""
        config_data = {
            "langproviders": [
                {
                    "language": "ja,en",
                    "model_type": "local.LocalHFTranslator",
                    "model_name": "Helsinki-NLP/opus-mt-{}",
                    "hf_args": {"device": "cpu"},
                }
            ]
        }
        self.create_test_config(config_data)

        # Mock the plugin loader to return a mock LocalHFTranslator instance
        mock_provider = MagicMock()
        mock_provider.language = "ja,en"
        mock_provider.source_lang = "ja"
        mock_provider.target_lang = "en"
        mock_load_plugin.return_value = mock_provider

        # Call the function
        result = _load_langprovider(self.test_config_path)

        # Verify the result
        assert result == mock_provider
        assert result.language == "ja,en"
        assert result.source_lang == "ja"
        assert result.target_lang == "en"

        # Verify _load_plugin was called with correct arguments
        mock_load_plugin.assert_called_once_with(
            path="nemoguardrails.evaluate.langproviders.local.LocalHFTranslator",
            config_root={
                "langproviders": {
                    "local.LocalHFTranslator": {
                        "language": "ja,en",
                        "model_type": "local.LocalHFTranslator",
                        "model_name": "Helsinki-NLP/opus-mt-{}",
                        "hf_args": {"device": "cpu"},
                    }
                }
            },
        )

    def test_load_langprovider_with_invalid_config_file(self):
        """Test loading with non-existent configuration file."""
        invalid_path = "/path/to/nonexistent/config.yaml"

        with pytest.raises(FileNotFoundError):
            _load_langprovider(invalid_path)

    def test_load_langprovider_with_invalid_yaml(self):
        """Test loading with invalid YAML configuration."""
        # Create invalid YAML file
        invalid_config_path = os.path.join(self.temp_dir, "invalid.yaml")
        with open(invalid_config_path, "w") as f:
            f.write("invalid: yaml: content: [")

        with pytest.raises(yaml.YAMLError):
            _load_langprovider(invalid_config_path)

    def test_load_langprovider_with_missing_langproviders_key(self):
        """Test loading with configuration missing 'langproviders' key."""
        config_data = {"other_key": "value"}
        self.create_test_config(config_data)

        with pytest.raises(KeyError):
            _load_langprovider(self.test_config_path)

    def test_load_langprovider_with_empty_langproviders_list(self):
        """Test loading with empty langproviders list."""
        config_data = {"langproviders": []}
        self.create_test_config(config_data)

        with pytest.raises(IndexError):
            _load_langprovider(self.test_config_path)

    @patch("nemoguardrails.evaluate.utils_translate._load_plugin")
    def test_load_langprovider_plugin_load_error(self, mock_load_plugin):
        """Test handling of plugin loading errors."""
        config_data = {
            "langproviders": [
                {"language": "en,ja", "model_type": "remote.DeeplTranslator"}
            ]
        }
        self.create_test_config(config_data)

        # Mock _load_plugin to raise an exception
        mock_load_plugin.side_effect = ImportError("Module not found")

        with pytest.raises(PluginConfigurationError) as exc_info:
            _load_langprovider(self.test_config_path)

        assert (
            "Failed to load 'en,ja' langprovider of type 'remote.DeeplTranslator'"
            in str(exc_info.value)
        )

    def test_load_langprovider_with_default_config(self):
        """Test loading with the default configuration file."""
        # Call without specifying config path should raise an error
        with pytest.raises(PluginConfigurationError) as exc_info:
            _load_langprovider()
        assert "No configuration file provided" in str(exc_info.value)

    @patch("nemoguardrails.evaluate.utils_translate._load_plugin")
    def test_load_langprovider_multiple_configurations(self, mock_load_plugin):
        """Test loading with multiple language provider configurations."""
        config_data = {
            "langproviders": [
                {"language": "en,ja", "model_type": "remote.DeeplTranslator"},
                {"language": "ja,en", "model_type": "local.LocalHFTranslator"},
            ]
        }
        self.create_test_config(config_data)

        mock_provider = MagicMock()
        mock_load_plugin.return_value = mock_provider

        # Should use the first configuration
        result = _load_langprovider(self.test_config_path)

        assert result == mock_provider
        mock_load_plugin.assert_called_once_with(
            path="nemoguardrails.evaluate.langproviders.remote.DeeplTranslator",
            config_root={
                "langproviders": {
                    "remote.DeeplTranslator": {
                        "language": "en,ja",
                        "model_type": "remote.DeeplTranslator",
                    }
                }
            },
        )

    @patch("nemoguardrails.evaluate.utils_translate._load_plugin")
    def test_load_langprovider_with_additional_config(self, mock_load_plugin):
        """Test loading with additional configuration parameters."""
        config_data = {
            "langproviders": [
                {
                    "language": "en,ja",
                    "model_type": "remote.DeeplTranslator",
                    "custom_param": "custom_value",
                    "another_param": 123,
                }
            ]
        }
        self.create_test_config(config_data)

        mock_provider = MagicMock()
        mock_load_plugin.return_value = mock_provider

        result = _load_langprovider(self.test_config_path)

        assert result == mock_provider

        # Verify all config parameters are passed through
        call_args = mock_load_plugin.call_args
        config_root = call_args[1]["config_root"]
        provider_config = config_root["langproviders"]["remote.DeeplTranslator"]

        assert provider_config["language"] == "en,ja"
        assert provider_config["model_type"] == "remote.DeeplTranslator"
        assert provider_config["custom_param"] == "custom_value"
        assert provider_config["another_param"] == 123

    def test_config_file_structure_validation(self):
        """Test validation of configuration file structure."""
        # Test with minimal valid config
        config_data = {
            "langproviders": [
                {"language": "en,ja", "model_type": "remote.DeeplTranslator"}
            ]
        }
        self.create_test_config(config_data)

        with patch(
            "nemoguardrails.evaluate.utils_translate._load_plugin"
        ) as mock_load_plugin:
            mock_provider = MagicMock()
            mock_load_plugin.return_value = mock_provider

            result = _load_langprovider(self.test_config_path)
            assert result == mock_provider

    def test_language_pair_validation_in_config(self):
        """Test validation of language pairs in configuration."""
        # Test with invalid language pair (same source and target)
        config_data = {
            "langproviders": [
                {
                    "language": "en,en",  # Invalid: same language
                    "model_type": "remote.DeeplTranslator",
                }
            ]
        }
        self.create_test_config(config_data)

        with patch(
            "nemoguardrails.evaluate.utils_translate._load_plugin"
        ) as mock_load_plugin:
            # The validation should happen in the LangProvider class, not in the utility function
            mock_provider = MagicMock()
            mock_load_plugin.return_value = mock_provider

            # This should not raise an exception at the utility level
            result = _load_langprovider(self.test_config_path)
            assert result == mock_provider

    @patch("nemoguardrails.evaluate.utils_translate._load_plugin")
    def test_load_langprovider_error_handling(self, mock_load_plugin):
        """Test comprehensive error handling."""
        config_data = {
            "langproviders": [
                {"language": "en,ja", "model_type": "remote.DeeplTranslator"}
            ]
        }
        self.create_test_config(config_data)

        # Test various types of exceptions
        exceptions_to_test = [
            ImportError("Module not found"),
            AttributeError("Missing attribute"),
            ValueError("Invalid value"),
            RuntimeError("Runtime error"),
        ]

        for exception in exceptions_to_test:
            mock_load_plugin.side_effect = exception

            with pytest.raises(PluginConfigurationError) as exc_info:
                _load_langprovider(self.test_config_path)

            assert (
                "Failed to load 'en,ja' langprovider of type 'remote.DeeplTranslator'"
                in str(exc_info.value)
            )
            assert str(exception) in str(exc_info.value.__cause__)
