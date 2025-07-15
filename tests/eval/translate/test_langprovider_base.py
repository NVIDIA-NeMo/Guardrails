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
from unittest.mock import MagicMock, patch

import pytest

from nemoguardrails.evaluate.langproviders.base import TranslationProvider


class MockTranslationProvider(TranslationProvider):
    """Mock implementation of TranslationProvider for testing."""

    ENV_VAR = "MOCK_API_KEY"

    def _load_langprovider(self):
        """Mock implementation of _load_langprovider."""
        self.loaded = True

    def _translate(self, text: str) -> str:
        """Mock implementation of _translate."""
        return f"translated_{text}"


class TestTranslationProvider:
    """Test cases for TranslationProvider base class."""

    def test_init_with_config(self):
        """Test initialization with valid configuration."""
        config = {
            "langproviders": {
                "mock.MockTranslator": {
                    "language": "en,ja",
                    "model_type": "mock.MockTranslator",
                }
            }
        }

        with patch.dict(os.environ, {"MOCK_API_KEY": "test_key"}):
            provider = MockTranslationProvider(config)

            assert provider.language == "en,ja"
            assert provider.source_lang == "en"
            assert provider.target_lang == "ja"
            assert provider.api_key == "test_key"
            assert provider.key_env_var == "MOCK_API_KEY"
            assert hasattr(provider, "loaded")
            assert provider.loaded is True

    def test_init_without_config(self):
        """Test initialization without configuration."""
        with pytest.raises(Exception):
            provider = MockTranslationProvider()

    def test_init_same_source_target_language(self):
        """Test initialization with same source and target language raises exception."""
        config = {
            "langproviders": {
                "mock.MockTranslator": {
                    "language": "en,en",
                    "model_type": "mock.MockTranslator",
                }
            }
        }

        with pytest.raises(Exception) as exc_info:
            MockTranslationProvider(config)

        assert "Source and target languages cannot be the same: en" in str(
            exc_info.value
        )

    def test_init_missing_env_var(self):
        """Test initialization with missing environment variable raises exception."""
        config = {
            "langproviders": {
                "mock.MockTranslator": {
                    "language": "en,ja",
                    "model_type": "mock.MockTranslator",
                }
            }
        }

        # Ensure the environment variable is not set
        if "MOCK_API_KEY" in os.environ:
            del os.environ["MOCK_API_KEY"]

        with pytest.raises(Exception) as exc_info:
            MockTranslationProvider(config)

        assert "Put the API key in the MOCK_API_KEY environment variable" in str(
            exc_info.value
        )

    def test_init_with_existing_api_key(self):
        """Test initialization when api_key is already set."""
        config = {
            "langproviders": {
                "mock.MockTranslator": {
                    "language": "en,ja",
                    "model_type": "mock.MockTranslator",
                }
            }
        }

        # Create provider with existing api_key
        provider = MockTranslationProvider.__new__(MockTranslationProvider)
        provider.api_key = "existing_key"

        with patch.object(provider, "_load_langprovider"):
            provider.__init__(config)

            assert provider.api_key == "existing_key"

    def test_get_response(self):
        """Test _get_response method."""
        config = {
            "langproviders": {
                "mock.MockTranslator": {
                    "language": "en,ja",
                    "model_type": "mock.MockTranslator",
                }
            }
        }

        with patch.dict(os.environ, {"MOCK_API_KEY": "test_key"}):
            provider = MockTranslationProvider(config)

            result = provider._get_response("hello")
            assert result == "translated_hello"

    def test_validate_env_var_without_env_var_attr(self):
        """Test _validate_env_var when class doesn't have ENV_VAR attribute."""

        class NoEnvVarProvider(TranslationProvider):
            def _load_langprovider(self):
                pass

            def _translate(self, text: str) -> str:
                return text

        config = {
            "langproviders": {
                "mock.NoEnvVarProvider": {
                    "language": "en,ja",
                    "model_type": "mock.NoEnvVarProvider",
                }
            }
        }

        # Should not raise exception when no ENV_VAR attribute
        provider = NoEnvVarProvider(config)
        assert not hasattr(provider, "key_env_var")

    def test_validate_env_var_with_empty_env_var(self):
        """Test _validate_env_var with empty environment variable."""
        config = {
            "langproviders": {
                "mock.MockTranslator": {
                    "language": "en,ja",
                    "model_type": "mock.MockTranslator",
                }
            }
        }

        with patch.dict(os.environ, {"MOCK_API_KEY": ""}):
            with pytest.raises(Exception) as exc_info:
                MockTranslationProvider(config)

            assert "Put the API key in the MOCK_API_KEY environment variable" in str(
                exc_info.value
            )

    def test_config_with_multiple_langproviders(self):
        """Test initialization with multiple language providers (should use first one)."""
        config = {
            "langproviders": {
                "mock.MockTranslator1": {
                    "language": "en,ja",
                    "model_type": "mock.MockTranslator1",
                },
                "mock.MockTranslator2": {
                    "language": "ja,en",
                    "model_type": "mock.MockTranslator2",
                },
            }
        }

        with patch.dict(os.environ, {"MOCK_API_KEY": "test_key"}):
            provider = MockTranslationProvider(config)

            # Should use the first language provider
            assert provider.language == "en,ja"
            assert provider.source_lang == "en"
            assert provider.target_lang == "ja"

    def test_config_with_empty_langproviders(self):
        """Test initialization with empty langproviders configuration."""
        config = {"langproviders": {}}
        with pytest.raises(Exception):
            provider = MockTranslationProvider(config)

    def test_translate_method_implementation(self):
        """Test that _translate method is properly called."""
        config = {
            "langproviders": {
                "mock.MockTranslator": {
                    "language": "en,ja",
                    "model_type": "mock.MockTranslator",
                }
            }
        }

        with patch.dict(os.environ, {"MOCK_API_KEY": "test_key"}):
            provider = MockTranslationProvider(config)

            # Test direct _translate call
            result = provider._translate("test message")
            assert result == "translated_test message"

            # Test through _get_response
            result = provider._get_response("another message")
            assert result == "translated_another message"

    def test_language_parsing_edge_cases(self):
        """Test language parsing with various edge cases."""
        test_cases = [
            ("en,ja", ("en", "ja")),
            ("ja,en", ("ja", "en")),
            ("fr,de", ("fr", "de")),
        ]

        for language_pair, expected in test_cases:
            config = {
                "langproviders": {
                    "mock.MockTranslator": {
                        "language": language_pair,
                        "model_type": "mock.MockTranslator",
                    }
                }
            }

            with patch.dict(os.environ, {"MOCK_API_KEY": "test_key"}):
                provider = MockTranslationProvider(config)

                assert provider.source_lang == expected[0]
                assert provider.target_lang == expected[1]

    def test_error_message_format(self):
        """Test that error messages are properly formatted."""
        config = {
            "langproviders": {
                "mock.MockTranslator": {
                    "language": "en,en",
                    "model_type": "mock.MockTranslator",
                }
            }
        }

        with pytest.raises(Exception) as exc_info:
            MockTranslationProvider(config)

        error_message = str(exc_info.value)
        assert "Source and target languages cannot be the same: en" in error_message

    def test_env_var_error_message_format(self):
        """Test that environment variable error messages are properly formatted."""
        config = {
            "langproviders": {
                "mock.MockTranslator": {
                    "language": "en,ja",
                    "model_type": "mock.MockTranslator",
                }
            }
        }

        # Ensure the environment variable is not set
        if "MOCK_API_KEY" in os.environ:
            del os.environ["MOCK_API_KEY"]

        with pytest.raises(Exception) as exc_info:
            MockTranslationProvider(config)

        error_message = str(exc_info.value)
        assert "MOCK_API_KEY" in error_message
        assert "environment variable" in error_message
        assert "export MOCK_API_KEY=" in error_message
