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
from unittest.mock import patch

from nemoguardrails.evaluate.utils_translate import _load_langprovider, PluginConfigurationError
from nemoguardrails.evaluate.langproviders.base import LangProvider


class TestLoadLangProviderIntegration:
    """Integration tests for _load_langprovider function with actual LangProvider classes."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            for file in os.listdir(self.temp_dir):
                os.remove(os.path.join(self.temp_dir, file))
            os.rmdir(self.temp_dir)

    def test_load_local_hf_translator_integration(self):
        """Test loading LocalHFTranslator with actual class."""
        config = {
            "langproviders": [
                {
                    "language": "en,ja",
                    "model_type": "local.LocalHFTranslator"
                }
            ]
        }
        config_path = os.path.join(self.temp_dir, "local_hf_config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(config, f)
        with patch('transformers.M2M100ForConditionalGeneration') as mock_model, \
             patch('transformers.M2M100Tokenizer') as mock_tokenizer, \
             patch('transformers.MarianMTModel') as mock_marian_model, \
             patch('transformers.MarianTokenizer') as mock_marian_tokenizer, \
             patch('torch.multiprocessing.set_start_method'):
            mock_model_instance = mock_model.from_pretrained.return_value
            mock_tokenizer_instance = mock_tokenizer.from_pretrained.return_value
            mock_marian_model_instance = mock_marian_model.from_pretrained.return_value
            mock_marian_tokenizer_instance = mock_marian_tokenizer.from_pretrained.return_value
            result = _load_langprovider(config_path)
            assert isinstance(result, LangProvider)
            assert result.language == "en,ja"
            assert result.source_lang == "en"
            assert result.target_lang == "jap"

    def test_load_langprovider_missing_api_key(self):
        """Test loading with missing API key for remote services."""
        config = {
            "langproviders": [
                {
                    "language": "en,ja",
                    "model_type": "remote.DeeplTranslator"
                }
            ]
        }
        config_path = os.path.join(self.temp_dir, "missing_key_config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(config, f)
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(PluginConfigurationError) as exc_info:
                _load_langprovider(config_path)
            assert "Failed to load" in str(exc_info.value)

    def test_load_langprovider_invalid_language_pair(self):
        """Test loading with invalid language pair."""
        config = {
            "langproviders": [
                {
                    "language": "en,en",
                    "model_type": "local.LocalHFTranslator"
                }
            ]
        }
        config_path = os.path.join(self.temp_dir, "invalid_lang_config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(config, f)
        with patch('transformers.M2M100ForConditionalGeneration'), \
             patch('transformers.M2M100Tokenizer'), \
             patch('transformers.MarianMTModel'), \
             patch('transformers.MarianTokenizer'), \
             patch('torch.multiprocessing.set_start_method'):
            with pytest.raises(Exception) as exc_info:
                _load_langprovider(config_path)
            assert "Source and target languages cannot be the same" in str(exc_info.value) or "Failed to load" in str(exc_info.value)

    def test_load_langprovider_unsupported_language(self):
        """Test loading with unsupported language pair."""
        config = {
            "langproviders": [
                {
                    "language": "xx,yy",
                    "model_type": "local.LocalHFTranslator"
                }
            ]
        }
        config_path = os.path.join(self.temp_dir, "unsupported_lang_config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(config, f)
        with patch('transformers.M2M100ForConditionalGeneration') as mock_model, \
             patch('transformers.M2M100Tokenizer'), \
             patch('transformers.MarianMTModel') as mock_marian_model, \
             patch('transformers.MarianTokenizer'), \
             patch('torch.multiprocessing.set_start_method'):
            mock_marian_model.from_pretrained.side_effect = Exception("is not supported")
            with pytest.raises(Exception) as exc_info:
                _load_langprovider(config_path)
            assert "Failed to load" in str(exc_info.value)

    def test_load_langprovider_nonexistent_module(self):
        """Test loading with non-existent module path."""
        config = {
            "langproviders": [
                {
                    "language": "en,ja",
                    "model_type": "nonexistent.NonexistentTranslator"
                }
            ]
        }
        config_path = os.path.join(self.temp_dir, "nonexistent_config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(config, f)
        with pytest.raises(PluginConfigurationError) as exc_info:
            _load_langprovider(config_path)
        assert "Failed to load" in str(exc_info.value)

    def test_load_langprovider_translation_functionality(self):
        """Test that the loaded provider can perform translation."""
        config = {
            "langproviders": [
                {
                    "language": "en,ja",
                    "model_type": "local.LocalHFTranslator"
                }
            ]
        }
        config_path = os.path.join(self.temp_dir, "translation_test_config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(config, f)
        with patch('transformers.M2M100ForConditionalGeneration') as mock_model, \
             patch('transformers.M2M100Tokenizer') as mock_tokenizer, \
             patch('transformers.MarianMTModel') as mock_marian_model, \
             patch('transformers.MarianTokenizer') as mock_marian_tokenizer, \
             patch('torch.multiprocessing.set_start_method'):
            mock_model_instance = mock_model.from_pretrained.return_value
            mock_tokenizer_instance = mock_tokenizer.from_pretrained.return_value
            mock_marian_model_instance = mock_marian_model.from_pretrained.return_value
            mock_marian_tokenizer_instance = mock_marian_tokenizer.from_pretrained.return_value
            # Mock the translation process
            mock_tokenizer_instance.src_lang = "en"
            mock_tokenizer_instance.get_lang_id.return_value = 123
            mock_tokenizer_instance.return_value = {"input_ids": "mocked_input"}
            mock_model_instance.generate.return_value = "mocked_output"
            # batch_decodeがリストを返すようにする
            mock_tokenizer_instance.batch_decode = lambda *args, **kwargs: ["こんにちは"]
            mock_marian_tokenizer_instance.batch_decode = lambda *args, **kwargs: ["こんにちは"]
            provider = _load_langprovider(config_path)
            result = provider._get_response("Hello")
            assert result == "こんにちは"

    def test_load_langprovider_config_validation(self):
        """Test that the function validates configuration properly."""
        config = {
            "langproviders": [
                {
                    "model_type": "local.LocalHFTranslator"
                }
            ]
        }
        config_path = os.path.join(self.temp_dir, "invalid_config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(config, f)
        with pytest.raises(KeyError):
            _load_langprovider(config_path)

    def test_load_langprovider_with_default_config(self):
        """Test loading with the default configuration file."""
        # Call without specifying config path should raise an error
        with pytest.raises(PluginConfigurationError) as exc_info:
            _load_langprovider()
        assert "No configuration file provided" in str(exc_info.value)