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


import logging
import os
import re
import string
import unicodedata
from typing import List


class TranslationProvider:
    """Base class for objects that provision language translation services."""

    def __init__(self, config: dict = None) -> None:
        """
        Initialize the translation provider with optional configuration.

        Args:
            config (dict, optional): Configuration dictionary containing translation provider settings.
                Expected to have a 'langproviders' key with provider-specific configuration.

        Attributes:
            ENV_VAR (str): Name of the environment variable that should contain the API key for the translation provider.
                If the subclass defines this attribute, the API key will be loaded from the specified environment variable.

        Raises:
            Exception: If config, langproviders_config, or language is missing or invalid.
        """
        self.language = ""
        self.local_mode = False
        self.config = config  # Store config for subclasses to access
        if not config:
            raise Exception(
                "config must be provided for TranslationProvider initialization."
            )
        # Extract configuration from the config
        langproviders_config = config.get("langproviders", {})
        if not langproviders_config:
            raise Exception("'langproviders' configuration is missing in config.")
        # Get the first (and typically only) language provider config
        found_config = False
        for _, each_config in langproviders_config.items():
            self.language = each_config.get("language", "")
            model_type = each_config.get("model_type", "")
            local_mode = each_config.get("local_mode", False)
            if model_type == "remote.RivaTranslator":
                self.local_mode = local_mode
            found_config = True
            break
        if not found_config:
            raise Exception(
                "No valid language provider configuration found in 'langproviders'."
            )
        if not self.language:
            raise Exception(
                "'language' must be specified in the language provider configuration."
            )
        self.source_lang, self.target_lang = self.language.split(",")
        if self.source_lang == self.target_lang:
            raise Exception(
                f"Source and target languages cannot be the same: {self.source_lang}"
            )
        # Validate environment variable and set API key before loading the provider
        if hasattr(self, "ENV_VAR"):
            self.key_env_var = self.ENV_VAR
            self._validate_env_var()
        self._load_langprovider()

    def _load_langprovider(self):
        raise NotImplementedError

    def _translate(self, text: str) -> str:
        raise NotImplementedError

    def _get_response(self, input_text: str):
        return self._translate(input_text)

    def _translate_with_cache(self, text: str) -> str:
        """Translate text with caching support."""
        from nemoguardrails.evaluate.utils import get_translation_cache

        cache = get_translation_cache()
        target_lang = getattr(self, "target_lang", "unknown")

        # Check cache first
        cached_translation = cache.get(text, target_lang)
        if cached_translation:
            return cached_translation

        # Translate and cache
        translated_text = self._translate(text)
        cache.set(text, target_lang, translated_text)

        return translated_text

    def _validate_env_var(self):
        if hasattr(self, "key_env_var"):
            if not hasattr(self, "api_key") or self.api_key is None:
                self.api_key = os.getenv(self.key_env_var, default=None)
            # Empty strings are also considered as not set
            if self.api_key is None or self.api_key == "":
                raise Exception(
                    f'🛑 Put the API key in the {self.key_env_var} environment variable (this was empty)\n \
                    e.g.: export {self.key_env_var}="XXXXXXX"'
                )
