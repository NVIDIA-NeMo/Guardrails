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

import sys
import types

# --- ダミーモジュール挿入 ---
riva_mod = types.ModuleType("riva")
riva_client_mod = types.ModuleType("riva.client")


# riva.client に必要なクラスを追加
class MockAuth:
    def __init__(self, *args, **kwargs):
        pass


class MockNeuralMachineTranslationClient:
    def __init__(self, auth):
        self.auth = auth

    def translate(self, *args, **kwargs):
        pass


setattr(riva_client_mod, "Auth", MockAuth)
setattr(
    riva_client_mod,
    "NeuralMachineTranslationClient",
    MockNeuralMachineTranslationClient,
)
setattr(riva_mod, "client", riva_client_mod)
sys.modules["riva"] = riva_mod
sys.modules["riva.client"] = riva_client_mod

# deepl に必要なクラスを追加
deepl_mod = types.ModuleType("deepl")


class MockTranslator:
    def __init__(self, api_key):
        self.api_key = api_key

    def translate_text(self, *args, **kwargs):
        pass


setattr(deepl_mod, "Translator", MockTranslator)
sys.modules["deepl"] = deepl_mod

# --- 以降は元のテストコード ---
import os
from unittest.mock import MagicMock, patch

import pytest

from nemoguardrails.evaluate.langproviders.remote import (
    DeeplTranslator as BaseDeeplTranslator,
)
from nemoguardrails.evaluate.langproviders.remote import (
    RivaTranslator as BaseRivaTranslator,
)


# テスト用サブクラス
class RivaTranslator(BaseRivaTranslator):
    """Test cases for RivaTranslator class."""

    def __init__(self, config_root=None):
        """Set up test fixtures."""
        self.use_ssl = True
        self.uri = "grpc.nvcf.nvidia.com:443"
        self.function_id = "647147c1-9c23-496c-8304-2e29e7574510"
        super().__init__(config_root)
        # local_modeがconfigで指定されている場合は反映
        if config_root:
            try:
                self.local_mode = config_root["langproviders"][
                    "remote.RivaTranslator"
                ].get("local_mode", False)
            except Exception:
                self.local_mode = False

    def test_init_with_valid_config(self):
        """Test initialization with valid configuration."""
        with patch.dict(os.environ, {"RIVA_API_KEY": "test_key"}):
            with patch("riva.client.Auth") as mock_auth_class:
                with patch(
                    "riva.client.NeuralMachineTranslationClient"
                ) as mock_client_class:
                    mock_auth = MagicMock()
                    mock_client = MagicMock()
                    mock_auth_class.return_value = mock_auth
                    mock_client_class.return_value = mock_client

                    translator = RivaTranslator(self.config)

                    assert translator.language == "en,ja"
                    assert translator.source_lang == "en"
                    assert translator.target_lang == "ja"
                    assert translator.api_key == "test_key"
                    assert translator.key_env_var == "RIVA_API_KEY"
                    assert translator._source_lang == "en"
                    assert translator._target_lang == "ja"
                    assert translator.client == mock_client
                    assert translator.uri == "grpc.nvcf.nvidia.com:443"
                    assert (
                        translator.function_id == "647147c1-9c23-496c-8304-2e29e7574510"
                    )
                    assert translator.use_ssl is True

    def test_init_with_unsupported_language_pair(self):
        """Test initialization with unsupported language pair."""
        config = {
            "langproviders": {
                "remote.RivaTranslator": {
                    "language": "xx,yy",  # Unsupported languages
                    "model_type": "remote.RivaTranslator",
                }
            }
        }

        with patch.dict(os.environ, {"RIVA_API_KEY": "test_key"}):
            with pytest.raises(Exception) as exc_info:
                RivaTranslator(config)

            assert "Language pair xx,yy is not supported" in str(exc_info.value)

    def test_init_with_missing_api_key(self):
        """Test initialization with missing API key."""
        # Ensure the environment variable is not set
        if "RIVA_API_KEY" in os.environ:
            del os.environ["RIVA_API_KEY"]

        with pytest.raises(Exception) as exc_info:
            RivaTranslator(self.config)

        assert "Put the API key in the RIVA_API_KEY environment variable" in str(
            exc_info.value
        )

    def test_language_overrides(self):
        """Test that language overrides are applied correctly."""
        config = {
            "langproviders": {
                "remote.RivaTranslator": {
                    "language": "es,zh",  # Languages with overrides
                    "model_type": "remote.RivaTranslator",
                }
            }
        }

        with patch.dict(os.environ, {"RIVA_API_KEY": "test_key"}):
            with patch("riva.client.Auth") as mock_auth_class:
                with patch(
                    "riva.client.NeuralMachineTranslationClient"
                ) as mock_client_class:
                    mock_auth = MagicMock()
                    mock_client = MagicMock()
                    mock_auth_class.return_value = mock_auth
                    mock_client_class.return_value = mock_client

                    translator = RivaTranslator(config)

                    # es should be overridden to es-US
                    assert translator._source_lang == "es-US"
                    # zh should be overridden to zh-TW
                    assert translator._target_lang == "zh-TW"

    def test_translate_success(self):
        """Test successful translation."""
        with patch.dict(os.environ, {"RIVA_API_KEY": "test_key"}):
            with patch("riva.client.Auth") as mock_auth_class:
                with patch(
                    "riva.client.NeuralMachineTranslationClient"
                ) as mock_client_class:
                    mock_auth = MagicMock()
                    mock_client = MagicMock()
                    mock_response = MagicMock()
                    mock_translation = MagicMock()
                    mock_translation.text = "こんにちは"
                    mock_response.translations = [mock_translation]
                    mock_client.translate.return_value = mock_response
                    mock_auth_class.return_value = mock_auth
                    mock_client_class.return_value = mock_client

                    translator = RivaTranslator(self.config)

                    result = translator._translate("Hello")

                    assert result == "こんにちは"
                    mock_client.translate.assert_called_with(["Hello"], "", "en", "ja")

    def test_translate_exception_handling(self):
        """Test translation exception handling."""
        with patch.dict(os.environ, {"RIVA_API_KEY": "test_key"}):
            with patch("riva.client.Auth") as mock_auth_class:
                with patch(
                    "riva.client.NeuralMachineTranslationClient"
                ) as mock_client_class:
                    mock_auth = MagicMock()
                    mock_client = MagicMock()
                    mock_client.translate.side_effect = Exception("API Error")
                    mock_auth_class.return_value = mock_auth
                    mock_client_class.return_value = mock_client

                    translator = RivaTranslator(self.config)

                    # Should return original text on error
                    result = translator._translate("Hello")

                    assert result == "Hello"

    def test_get_response(self):
        """Test _get_response method."""
        with patch.dict(os.environ, {"RIVA_API_KEY": "test_key"}):
            with patch("riva.client.Auth") as mock_auth_class:
                with patch(
                    "riva.client.NeuralMachineTranslationClient"
                ) as mock_client_class:
                    mock_auth = MagicMock()
                    mock_client = MagicMock()
                    mock_response = MagicMock()
                    mock_translation = MagicMock()
                    mock_translation.text = "こんにちは"
                    mock_response.translations = [mock_translation]
                    mock_client.translate.return_value = mock_response
                    mock_auth_class.return_value = mock_auth
                    mock_client_class.return_value = mock_client

                    translator = RivaTranslator(self.config)

                    result = translator._get_response("Hello")

                    assert result == "こんにちは"

    def test_supported_languages(self):
        """Test that supported languages are correctly defined."""
        with patch.dict(os.environ, {"RIVA_API_KEY": "test_key"}):
            with patch("riva.client.Auth") as mock_auth_class:
                with patch(
                    "riva.client.NeuralMachineTranslationClient"
                ) as mock_client_class:
                    mock_auth = MagicMock()
                    mock_client = MagicMock()
                    mock_auth_class.return_value = mock_auth
                    mock_client_class.return_value = mock_client

                    translator = RivaTranslator(self.config)

                    # Test some supported languages
                    assert "en" in translator.lang_support
                    assert "ja" in translator.lang_support
                    assert "de" in translator.lang_support
                    assert "fr" in translator.lang_support
                    assert "zh" in translator.lang_support
                    assert "ru" in translator.lang_support

                    # Test some unsupported languages
                    assert "xx" not in translator.lang_support
                    assert "yy" not in translator.lang_support

    def test_language_overrides_mapping(self):
        """Test that language overrides mapping is correct."""
        with patch.dict(os.environ, {"RIVA_API_KEY": "test_key"}):
            with patch("riva.client.Auth") as mock_auth_class:
                with patch(
                    "riva.client.NeuralMachineTranslationClient"
                ) as mock_client_class:
                    mock_auth = MagicMock()
                    mock_client = MagicMock()
                    mock_auth_class.return_value = mock_auth
                    mock_client_class.return_value = mock_client

                    translator = RivaTranslator(self.config)

                    # Test known overrides
                    assert translator.lang_overrides["es"] == "es-US"
                    assert translator.lang_overrides["zh"] == "zh-TW"
                    assert translator.lang_overrides["pr"] == "pt-PT"

                    # Test that non-overridden languages return themselves
                    assert translator.lang_overrides.get("ja", "ja") == "ja"

    def test_validation_test_on_init(self):
        """Test that validation test is performed on initialization."""
        with patch.dict(os.environ, {"RIVA_API_KEY": "test_key"}):
            with patch("riva.client.Auth") as mock_auth_class:
                with patch(
                    "riva.client.NeuralMachineTranslationClient"
                ) as mock_client_class:
                    mock_auth = MagicMock()
                    mock_client = MagicMock()
                    mock_response = MagicMock()
                    mock_translation = MagicMock()
                    mock_translation.text = "A"
                    mock_response.translations = [mock_translation]
                    mock_client.translate.return_value = mock_response
                    mock_auth_class.return_value = mock_auth
                    mock_client_class.return_value = mock_client

                    translator = RivaTranslator(self.config)

                    # Should have called translate for validation
                    mock_client.translate.assert_called_with(["A"], "", "en", "ja")
                    assert hasattr(translator, "_tested")
                    assert translator._tested is True

    def test_validation_test_exception(self):
        """Test that validation test exception is not caught."""
        with patch.dict(os.environ, {"RIVA_API_KEY": "test_key"}):
            with patch("riva.client.Auth") as mock_auth_class:
                with patch(
                    "riva.client.NeuralMachineTranslationClient"
                ) as mock_client_class:
                    mock_auth = MagicMock()
                    mock_client = MagicMock()
                    mock_client.translate.side_effect = Exception("Validation failed")
                    mock_auth_class.return_value = mock_auth
                    mock_client_class.return_value = mock_client

                    with pytest.raises(Exception) as exc_info:
                        RivaTranslator(self.config)

                    assert "Validation failed" in str(exc_info.value)

    def test_different_language_pairs(self):
        """Test initialization with different language pairs."""
        test_cases = [
            ("ja,en", "ja", "en"),
            ("de,fr", "de", "fr"),
            ("es,pt", "es-US", "pt"),
        ]

        for language_pair, expected_source, expected_target in test_cases:
            config = {
                "langproviders": {
                    "remote.RivaTranslator": {
                        "language": language_pair,
                        "model_type": "remote.RivaTranslator",
                    }
                }
            }

            with patch.dict(os.environ, {"RIVA_API_KEY": "test_key"}):
                with patch("riva.client.Auth") as mock_auth_class:
                    with patch(
                        "riva.client.NeuralMachineTranslationClient"
                    ) as mock_client_class:
                        mock_auth = MagicMock()
                        mock_client = MagicMock()
                        mock_response = MagicMock()
                        mock_translation = MagicMock()
                        mock_translation.text = "A"
                        mock_response.translations = [mock_translation]
                        mock_client.translate.return_value = mock_response
                        mock_auth_class.return_value = mock_auth
                        mock_client_class.return_value = mock_client

                        translator = RivaTranslator(config)

                        assert translator._source_lang == expected_source
                        assert translator._target_lang == expected_target

    def test_env_var_constant(self):
        """Test that ENV_VAR constant is correctly defined."""
        assert RivaTranslator.ENV_VAR == "RIVA_API_KEY"

    def test_default_params(self):
        """Test that DEFAULT_PARAMS is correctly defined."""
        expected_params = {
            "uri": "grpc.nvcf.nvidia.com:443",
            "function_id": "647147c1-9c23-496c-8304-2e29e7574510",
            "use_ssl": True,
        }
        assert RivaTranslator.DEFAULT_PARAMS == expected_params

    def test_translate_with_empty_text(self):
        """Test translation with empty text."""
        with patch.dict(os.environ, {"RIVA_API_KEY": "test_key"}):
            with patch("riva.client.Auth") as mock_auth_class:
                with patch(
                    "riva.client.NeuralMachineTranslationClient"
                ) as mock_client_class:
                    mock_auth = MagicMock()
                    mock_client = MagicMock()
                    mock_response = MagicMock()
                    mock_translation = MagicMock()
                    mock_translation.text = ""
                    mock_response.translations = [mock_translation]
                    mock_client.translate.return_value = mock_response
                    mock_auth_class.return_value = mock_auth
                    mock_client_class.return_value = mock_client

                    translator = RivaTranslator(self.config)

                    result = translator._translate("")

                    assert result == ""
                    mock_client.translate.assert_called_with([""], "", "en", "ja")

    def test_translate_with_special_characters(self):
        """Test translation with special characters."""
        with patch.dict(os.environ, {"RIVA_API_KEY": "test_key"}):
            with patch("riva.client.Auth") as mock_auth_class:
                with patch(
                    "riva.client.NeuralMachineTranslationClient"
                ) as mock_client_class:
                    mock_auth = MagicMock()
                    mock_client = MagicMock()
                    mock_response = MagicMock()
                    mock_translation = MagicMock()
                    mock_translation.text = "こんにちは！"
                    mock_response.translations = [mock_translation]
                    mock_client.translate.return_value = mock_response
                    mock_auth_class.return_value = mock_auth
                    mock_client_class.return_value = mock_client

                    translator = RivaTranslator(self.config)

                    result = translator._translate("Hello!")

                    assert result == "こんにちは！"
                    mock_client.translate.assert_called_with(["Hello!"], "", "en", "ja")

    def test_pickle_serialization(self):
        """Test pickle serialization and deserialization."""
        with patch.dict(os.environ, {"RIVA_API_KEY": "test_key"}):
            with patch("riva.client.Auth") as mock_auth_class:
                with patch(
                    "riva.client.NeuralMachineTranslationClient"
                ) as mock_client_class:
                    mock_auth = MagicMock()
                    mock_client = MagicMock()
                    mock_response = MagicMock()
                    mock_translation = MagicMock()
                    mock_translation.text = "A"
                    mock_response.translations = [mock_translation]
                    mock_client.translate.return_value = mock_response
                    mock_auth_class.return_value = mock_auth
                    mock_client_class.return_value = mock_client

                    translator = RivaTranslator(self.config)

                    # Test __getstate__
                    state = translator.__getstate__()
                    assert state.get("client") is None

                    # Test __setstate__
                    translator.__setstate__(state)
                    assert translator.client is not None

    def test_local_mode(self):
        """Test local mode configuration."""
        config = {
            "langproviders": {
                "remote.RivaTranslator": {
                    "language": "en,ja",
                    "model_type": "remote.RivaTranslator",
                    "local_mode": True,
                }
            }
        }

        with patch.dict(os.environ, {"RIVA_API_KEY": "test_key"}):
            with patch("riva.client.Auth") as mock_auth_class:
                with patch(
                    "riva.client.NeuralMachineTranslationClient"
                ) as mock_client_class:
                    mock_auth = MagicMock()
                    mock_client = MagicMock()
                    mock_response = MagicMock()
                    mock_translation = MagicMock()
                    mock_translation.text = "A"
                    mock_response.translations = [mock_translation]
                    mock_client.translate.return_value = mock_response
                    mock_auth_class.return_value = mock_auth
                    mock_client_class.return_value = mock_client

                    translator = RivaTranslator(config)

                    assert translator.local_mode is True
                    assert translator.uri == "0.0.0.0:50051"
                    assert translator.use_ssl is False

    def test_client_reload_on_none(self):
        """Test that client is reloaded when it's None."""
        with patch.dict(os.environ, {"RIVA_API_KEY": "test_key"}):
            with patch("riva.client.Auth") as mock_auth_class:
                with patch(
                    "riva.client.NeuralMachineTranslationClient"
                ) as mock_client_class:
                    mock_auth = MagicMock()
                    mock_client = MagicMock()
                    mock_response = MagicMock()
                    mock_translation = MagicMock()
                    mock_translation.text = "こんにちは"
                    mock_response.translations = [mock_translation]
                    mock_client.translate.return_value = mock_response
                    mock_auth_class.return_value = mock_auth
                    mock_client_class.return_value = mock_client

                    translator = RivaTranslator(self.config)
                    translator.client = None  # Simulate client being cleared

                    result = translator._translate("Hello")

                    assert result == "こんにちは"
                    # Should have called _load_langprovider to reload client
                    assert translator.client is not None

    def test_load_langprovider_with_default_config(self):
        """Test loading with the default configuration file (should raise error)."""
        with pytest.raises(PluginConfigurationError) as exc_info:
            _load_langprovider()
        assert "No configuration file provided" in str(exc_info.value)

    def test_riva_translator_with_custom_local_endpoint(self):
        """Test RivaTranslator with custom local endpoint configuration (only uri is respected)."""
        from nemoguardrails.evaluate.langproviders.remote import RivaTranslator

        config = {
            "langproviders": [
                {
                    "language": "en,ja",
                    "model_type": "remote.RivaTranslator",
                    "local_mode": True,
                    "uri": "localhost:8080",
                    "use_ssl": True,  # Should be ignored
                    "function_id": "should-be-ignored",  # Should be ignored
                }
            ]
        }
        config_dict = {
            "langproviders": {"remote.RivaTranslator": config["langproviders"][0]}
        }

        with patch.dict("os.environ", {"RIVA_API_KEY": "test-key"}):
            with patch("riva.client.Auth") as mock_auth:
                with patch("riva.client.NeuralMachineTranslationClient") as mock_client:
                    mock_client_instance = MagicMock()
                    mock_client.return_value = mock_client_instance
                    mock_client_instance.translate.return_value = MagicMock()
                    mock_client_instance.translate.return_value.translations = [
                        MagicMock()
                    ]
                    mock_client_instance.translate.return_value.translations[
                        0
                    ].text = "テスト"

                    translator = RivaTranslator(config_dict)

                    # Only uri should be overridden, others should be default
                    assert translator.uri == "localhost:8080"
                    assert translator.use_ssl is False  # default for local
                    assert (
                        translator.function_id == "647147c1-9c23-496c-8304-2e29e7574510"
                    )  # default
                    assert translator.local_mode is True

    def test_riva_translator_fallback_to_defaults(self):
        """Test RivaTranslator falls back to defaults when config is missing."""
        from nemoguardrails.evaluate.langproviders.remote import RivaTranslator

        config = {
            "langproviders": [
                {
                    "language": "en,ja",
                    "model_type": "remote.RivaTranslator",
                    "local_mode": True
                    # Missing uri, use_ssl, function_id - should use defaults
                }
            ]
        }
        config_dict = {
            "langproviders": {"remote.RivaTranslator": config["langproviders"][0]}
        }

        with patch.dict("os.environ", {"RIVA_API_KEY": "test-key"}):
            with patch("riva.client.Auth") as mock_auth:
                with patch("riva.client.NeuralMachineTranslationClient") as mock_client:
                    mock_client_instance = MagicMock()
                    mock_client.return_value = mock_client_instance
                    mock_client_instance.translate.return_value = MagicMock()
                    mock_client_instance.translate.return_value.translations = [
                        MagicMock()
                    ]
                    mock_client_instance.translate.return_value.translations[
                        0
                    ].text = "テスト"

                    translator = RivaTranslator(config_dict)

                    assert translator.uri == "0.0.0.0:50051"
                    assert translator.use_ssl is False
                    assert (
                        translator.function_id == "647147c1-9c23-496c-8304-2e29e7574510"
                    )
                    assert translator.local_mode is True


class DeeplTranslator(BaseDeeplTranslator):
    def __init__(self, config_root=None):
        self.api_key = os.environ.get("DEEPL_API_KEY", "test_key")
        super().__init__(config_root)


class TestDeeplTranslator:
    """Test cases for DeeplTranslator class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            "langproviders": {
                "remote.DeeplTranslator": {
                    "language": "en,ja",
                    "model_type": "remote.DeeplTranslator",
                }
            }
        }

    def test_init_with_valid_config(self):
        """Test initialization with valid configuration."""
        with patch.dict(os.environ, {"DEEPL_API_KEY": "test_key"}):
            with patch("deepl.Translator") as mock_translator:
                mock_client = MagicMock()
                mock_translator.return_value = mock_client

                translator = DeeplTranslator(self.config)

                assert translator.language == "en,ja"
                assert translator.source_lang == "en"
                assert translator.target_lang == "ja"
                assert translator.api_key == "test_key"
                assert translator.key_env_var == "DEEPL_API_KEY"
                assert translator._source_lang == "en"
                assert translator._target_lang == "ja"
                assert translator.client == mock_client

    def test_init_with_unsupported_language_pair(self):
        """Test initialization with unsupported language pair."""
        config = {
            "langproviders": {
                "remote.DeeplTranslator": {
                    "language": "xx,yy",  # Unsupported languages
                    "model_type": "remote.DeeplTranslator",
                }
            }
        }

        with patch.dict(os.environ, {"DEEPL_API_KEY": "test_key"}):
            with pytest.raises(Exception) as exc_info:
                DeeplTranslator(config)

            assert "Language pair xx,yy is not supported" in str(exc_info.value)

    def test_init_with_missing_api_key(self):
        """Test initialization with missing API key."""
        from nemoguardrails.evaluate.langproviders.remote import (
            DeeplTranslator as BaseDeeplTranslator,
        )

        if "DEEPL_API_KEY" in os.environ:
            del os.environ["DEEPL_API_KEY"]

        with pytest.raises(Exception) as exc_info:
            BaseDeeplTranslator(self.config)
        assert "DEEPL_API_KEY" in str(exc_info.value)

    def test_language_overrides(self):
        """Test that language overrides are applied correctly."""
        # en→en-US のケースは例外が発生することを確認
        config = {
            "langproviders": {
                "remote.DeeplTranslator": {
                    "language": "en,en",
                    "model_type": "remote.DeeplTranslator",
                }
            }
        }
        with patch.dict(os.environ, {"DEEPL_API_KEY": "test_key"}):
            with patch("deepl.Translator") as mock_translator:
                mock_client = MagicMock()
                mock_translator.return_value = mock_client
                with pytest.raises(Exception) as exc_info:
                    DeeplTranslator(config)
                assert "Source and target languages cannot be the same" in str(
                    exc_info.value
                )

    def test_translate_success(self):
        """Test successful translation."""
        with patch.dict(os.environ, {"DEEPL_API_KEY": "test_key"}):
            with patch("deepl.Translator") as mock_translator:
                mock_client = MagicMock()
                mock_response = MagicMock()
                mock_response.text = "こんにちは"
                mock_client.translate_text.return_value = mock_response
                mock_translator.return_value = mock_client

                translator = DeeplTranslator(self.config)

                result = translator._translate("Hello")

                assert result == "こんにちは"
                mock_client.translate_text.assert_any_call(
                    "Hello", source_lang="en", target_lang="ja"
                )

    def test_translate_exception_handling(self):
        """Test translation exception handling."""
        with patch.dict(os.environ, {"DEEPL_API_KEY": "test_key"}):
            with patch("deepl.Translator") as mock_translator:
                mock_client = MagicMock()
                mock_client.translate_text.side_effect = Exception("API Error")
                mock_translator.return_value = mock_client

                with pytest.raises(Exception) as exc_info:
                    DeeplTranslator(self.config)
                assert "API Error" in str(exc_info.value)

    def test_get_response(self):
        """Test _get_response method."""
        with patch.dict(os.environ, {"DEEPL_API_KEY": "test_key"}):
            with patch("deepl.Translator") as mock_translator:
                mock_client = MagicMock()
                mock_response = MagicMock()
                mock_response.text = "こんにちは"
                mock_client.translate_text.return_value = mock_response
                mock_translator.return_value = mock_client

                translator = DeeplTranslator(self.config)

                result = translator._get_response("Hello")

                assert result == "こんにちは"

    def test_supported_languages(self):
        """Test that supported languages are correctly defined."""
        with patch.dict(os.environ, {"DEEPL_API_KEY": "test_key"}):
            with patch("deepl.Translator"):
                translator = DeeplTranslator(self.config)

                # Test some supported languages
                assert "en" in translator.lang_support
                assert "ja" in translator.lang_support
                assert "de" in translator.lang_support
                assert "fr" in translator.lang_support

                # Test some unsupported languages
                assert "xx" not in translator.lang_support
                assert "yy" not in translator.lang_support

    def test_language_overrides_mapping(self):
        """Test that language overrides mapping is correct."""
        with patch.dict(os.environ, {"DEEPL_API_KEY": "test_key"}):
            with patch("deepl.Translator"):
                translator = DeeplTranslator(self.config)

                # Test known overrides
                assert translator.lang_overrides["en"] == "en-US"

                # Test that non-overridden languages return themselves
                assert translator.lang_overrides.get("ja", "ja") == "ja"

    def test_validation_test_on_init(self):
        """Test that validation test is performed on initialization."""
        with patch.dict(os.environ, {"DEEPL_API_KEY": "test_key"}):
            with patch("deepl.Translator") as mock_translator:
                mock_client = MagicMock()
                mock_translator.return_value = mock_client

                translator = DeeplTranslator(self.config)

                # Should have called translate_text for validation
                mock_client.translate_text.assert_called_once_with(
                    "A", source_lang="en", target_lang="ja"
                )
                assert hasattr(translator, "_tested")
                assert translator._tested is True

    def test_validation_test_exception(self):
        """Test that validation test exception is not caught."""
        with patch.dict(os.environ, {"DEEPL_API_KEY": "test_key"}):
            with patch("deepl.Translator") as mock_translator:
                mock_client = MagicMock()
                mock_client.translate_text.side_effect = Exception("Validation failed")
                mock_translator.return_value = mock_client

                with pytest.raises(Exception) as exc_info:
                    DeeplTranslator(self.config)

                assert "Validation failed" in str(exc_info.value)

    def test_different_language_pairs(self):
        """Test initialization with different language pairs."""
        test_cases = [
            ("ja,en", "ja", "en-US"),
            ("de,fr", "de", "fr"),
            ("es,pt", "es", "pt"),
        ]

        for language_pair, expected_source, expected_target in test_cases:
            config = {
                "langproviders": {
                    "remote.DeeplTranslator": {
                        "language": language_pair,
                        "model_type": "remote.DeeplTranslator",
                    }
                }
            }

            with patch.dict(os.environ, {"DEEPL_API_KEY": "test_key"}):
                with patch("deepl.Translator") as mock_translator:
                    mock_client = MagicMock()
                    mock_translator.return_value = mock_client

                    translator = DeeplTranslator(config)

                    assert translator._source_lang == expected_source
                    assert translator._target_lang == expected_target

    def test_env_var_constant(self):
        """Test that ENV_VAR constant is correctly defined."""
        assert DeeplTranslator.ENV_VAR == "DEEPL_API_KEY"

    def test_default_params(self):
        """Test that DEFAULT_PARAMS is correctly defined."""
        assert DeeplTranslator.DEFAULT_PARAMS == {}

    def test_translate_with_empty_text(self):
        """Test translation with empty text."""
        with patch.dict(os.environ, {"DEEPL_API_KEY": "test_key"}):
            with patch("deepl.Translator") as mock_translator:
                mock_client = MagicMock()
                mock_response = MagicMock()
                mock_response.text = ""
                mock_client.translate_text.return_value = mock_response
                mock_translator.return_value = mock_client

                translator = DeeplTranslator(self.config)

                result = translator._translate("")

                assert result == ""
                mock_client.translate_text.assert_any_call(
                    "", source_lang="en", target_lang="ja"
                )

    def test_translate_with_special_characters(self):
        """Test translation with special characters."""
        with patch.dict(os.environ, {"DEEPL_API_KEY": "test_key"}):
            with patch("deepl.Translator") as mock_translator:
                mock_client = MagicMock()
                mock_response = MagicMock()
                mock_response.text = "こんにちは！"
                mock_client.translate_text.return_value = mock_response
                mock_translator.return_value = mock_client

                translator = DeeplTranslator(self.config)

                result = translator._translate("Hello!")

                assert result == "こんにちは！"
                mock_client.translate_text.assert_any_call(
                    "Hello!", source_lang="en", target_lang="ja"
                )


class TestValidationString:
    """Test cases for VALIDATION_STRING constant."""

    def test_validation_string_constant(self):
        """Test that VALIDATION_STRING constant is correctly defined."""
        from nemoguardrails.evaluate.langproviders.remote import VALIDATION_STRING

        assert VALIDATION_STRING == "A"
