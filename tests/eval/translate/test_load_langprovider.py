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

import os
import tempfile
import yaml
import pytest
from unittest.mock import patch, MagicMock
import shutil

from nemoguardrails.evaluate.utils_translate import _load_langprovider, PluginConfigurationError, load_dataset


class TestLoadLangProvider:
    """Test cases for _load_langprovider function."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create a temporary config file for testing
        self.temp_dir = tempfile.mkdtemp()
        self.test_config_path = os.path.join(self.temp_dir, "test_translation.yaml")

        # Create test configuration
        test_config = {
            "langproviders": [
                {
                    "language": "en,ja",
                    "model_type": "remote.DeeplTranslator"
                }
            ]
        }

        with open(self.test_config_path, "w") as f:
            yaml.dump(test_config, f)

    def teardown_method(self):
        """Clean up test fixtures."""
        if os.path.exists(self.test_config_path):
            os.remove(self.test_config_path)
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch('nemoguardrails.evaluate.utils_translate._load_plugin')
    def test_load_langprovider_success(self, mock_load_plugin):
        """Test successful loading of language provider."""
        # Mock the plugin loader to return a mock LangProvider instance
        mock_provider = MagicMock()
        mock_load_plugin.return_value = mock_provider

        # Call the function
        result = _load_langprovider(self.test_config_path)

        # Verify the result
        assert result == mock_provider

        # Verify _load_plugin was called with correct arguments
        mock_load_plugin.assert_called_once_with(
            path="nemoguardrails.evaluate.langproviders.remote.DeeplTranslator",
            config_root={
                "langproviders": {
                    "remote.DeeplTranslator": {
                        "language": "en,ja",
                        "model_type": "remote.DeeplTranslator"
                    }
                }
            }
        )

    def test_load_langprovider_default_config(self):
        """Test loading with default configuration file."""
        # Call without specifying config path should raise an error
        with pytest.raises(PluginConfigurationError) as exc_info:
            _load_langprovider()

        assert "No configuration file provided" in str(exc_info.value)

    def test_load_langprovider_invalid_config_file(self):
        """Test loading with non-existent configuration file."""
        invalid_path = "/path/to/nonexistent/config.yaml"

        with pytest.raises(FileNotFoundError):
            _load_langprovider(invalid_path)

    def test_load_langprovider_invalid_yaml(self):
        """Test loading with invalid YAML configuration."""
        # Create invalid YAML file
        invalid_config_path = os.path.join(self.temp_dir, "invalid.yaml")
        with open(invalid_config_path, "w") as f:
            f.write("invalid: yaml: content: [")

        with pytest.raises(yaml.YAMLError):
            _load_langprovider(invalid_config_path)

    def test_load_langprovider_missing_langproviders_key(self):
        """Test loading with configuration missing 'langproviders' key."""
        # Create config without langproviders key
        invalid_config = {"other_key": "value"}
        invalid_config_path = os.path.join(self.temp_dir, "invalid_config.yaml")

        with open(invalid_config_path, "w") as f:
            yaml.dump(invalid_config, f)

        with pytest.raises(KeyError):
            _load_langprovider(invalid_config_path)

    def test_load_langprovider_empty_langproviders_list(self):
        """Test loading with empty langproviders list."""
        # Create config with empty langproviders list
        empty_config = {"langproviders": []}
        empty_config_path = os.path.join(self.temp_dir, "empty_config.yaml")

        with open(empty_config_path, "w") as f:
            yaml.dump(empty_config, f)

        with pytest.raises(IndexError):
            _load_langprovider(empty_config_path)

    @patch('nemoguardrails.evaluate.utils_translate._load_plugin')
    def test_load_langprovider_plugin_load_error(self, mock_load_plugin):
        """Test handling of plugin loading errors."""
        # Mock _load_plugin to raise an exception
        mock_load_plugin.side_effect = ImportError("Module not found")

        with pytest.raises(PluginConfigurationError) as exc_info:
            _load_langprovider(self.test_config_path)

        assert "Failed to load 'en,ja' langprovider of type 'remote.DeeplTranslator'" in str(exc_info.value)

    def test_load_langprovider_config_structure(self):
        """Test that the function correctly processes the configuration structure."""
        with patch('nemoguardrails.evaluate.utils_translate._load_plugin') as mock_load_plugin:
            mock_provider = MagicMock()
            mock_load_plugin.return_value = mock_provider

            # Call the function
            _load_langprovider(self.test_config_path)

            # Verify the config structure passed to _load_plugin
            call_args = mock_load_plugin.call_args
            config_root = call_args[1]['config_root']

            assert 'langproviders' in config_root
            assert 'remote.DeeplTranslator' in config_root['langproviders']
            assert config_root['langproviders']['remote.DeeplTranslator']['language'] == 'en,ja'
            assert config_root['langproviders']['remote.DeeplTranslator']['model_type'] == 'remote.DeeplTranslator'

    def test_load_langprovider_different_model_type(self):
        """Test loading with different model type."""
        # Create config with different model type
        different_config = {
            "langproviders": [
                {
                    "language": "ja,en",
                    "model_type": "local.LocalTranslator"
                }
            ]
        }
        different_config_path = os.path.join(self.temp_dir, "different_config.yaml")

        with open(different_config_path, "w") as f:
            yaml.dump(different_config, f)

        with patch('nemoguardrails.evaluate.utils_translate._load_plugin') as mock_load_plugin:
            mock_provider = MagicMock()
            mock_load_plugin.return_value = mock_provider

            result = _load_langprovider(different_config_path)

            assert result == mock_provider
            mock_load_plugin.assert_called_once_with(
                path="nemoguardrails.evaluate.langproviders.local.LocalTranslator",
                config_root={
                    "langproviders": {
                        "local.LocalTranslator": {
                            "language": "ja,en",
                            "model_type": "local.LocalTranslator"
                        }
                    }
                }
            )

    @patch('nemoguardrails.evaluate.utils_translate._load_langprovider')
    def test_load_dataset_with_local_translator_model_name(self, mock_load_langprovider):
        """Test that local translator with model_name creates appropriate cache filename."""
        # Create a mock translator with model_name attribute
        mock_translator = MagicMock()
        mock_translator.__class__.__name__ = "LocalHFTranslator"
        mock_translator.model_name = "facebook/m2m100_1.2B"
        mock_translator.target_lang = "ja"
        mock_translator._translate.return_value = "翻訳されたテキスト"
        mock_load_langprovider.return_value = mock_translator

        # Create test dataset file
        test_dataset_path = os.path.join(self.temp_dir, "test_dataset.txt")
        with open(test_dataset_path, "w") as f:
            f.write("Hello world\n")

        # Create test translation config
        test_translation_config = os.path.join(self.temp_dir, "test_translation_config.yaml")
        test_config = {
            "langproviders": [
                {
                    "language": "en,ja",
                    "model_type": "local.LocalHFTranslator",
                    "model_name": "facebook/m2m100_1.2B"
                }
            ]
        }
        with open(test_translation_config, "w") as f:
            yaml.dump(test_config, f)

        # Call load_dataset
        with patch('nemoguardrails.evaluate.utils_translate.get_translation_cache') as mock_get_cache:
            mock_cache = MagicMock()
            mock_get_cache.return_value = mock_cache
            mock_cache.get.return_value = None  # No cached translation
            mock_cache.get_cache_stats.return_value = {
                'total_entries': 0,
                'cache_size_bytes': 0,
                'cache_size_mb': 0.0,
                'cache_file': 'test_cache.json'
            }

            result = load_dataset(test_dataset_path, test_translation_config)

            # Verify that get_translation_cache was called with the expected service name
            expected_service_name = "LocalHFTranslator_facebook_m2m100_1.2B"
            mock_get_cache.assert_called_once_with(expected_service_name)

    @patch('nemoguardrails.evaluate.utils_translate._load_langprovider')
    def test_load_dataset_with_remote_translator_no_model_name(self, mock_load_langprovider):
        """Test that remote translator without model_name uses class name only."""
        # Create a mock translator without model_name attribute
        mock_translator = MagicMock()
        mock_translator.__class__.__name__ = "DeeplTranslator"
        mock_translator.target_lang = "ja"
        mock_translator._translate.return_value = "翻訳されたテキスト"
        # model_name属性を明示的に削除
        if hasattr(mock_translator, "model_name"):
            del mock_translator.model_name
        mock_load_langprovider.return_value = mock_translator

        # Create test dataset file
        test_dataset_path = os.path.join(self.temp_dir, "test_dataset.txt")
        with open(test_dataset_path, "w") as f:
            f.write("Hello world\n")

        # Create test translation config
        test_translation_config = os.path.join(self.temp_dir, "test_translation_config.yaml")
        test_config = {
            "langproviders": [
                {
                    "language": "en,ja",
                    "model_type": "remote.DeeplTranslator"
                }
            ]
        }
        with open(test_translation_config, "w") as f:
            yaml.dump(test_config, f)

        # Call load_dataset
        with patch('nemoguardrails.evaluate.utils_translate.get_translation_cache') as mock_get_cache:
            mock_cache = MagicMock()
            mock_get_cache.return_value = mock_cache
            mock_cache.get.return_value = None  # No cached translation
            mock_cache.get_cache_stats.return_value = {
                'total_entries': 0,
                'cache_size_bytes': 0,
                'cache_size_mb': 0.0,
                'cache_file': 'test_cache.json'
            }

            result = load_dataset(test_dataset_path, test_translation_config)

            # Verify that get_translation_cache was called with the expected service name
            expected_service_name = "DeeplTranslator"
            mock_get_cache.assert_called_once_with(expected_service_name)