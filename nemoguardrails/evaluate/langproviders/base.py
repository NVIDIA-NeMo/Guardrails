# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


"""Translator that translates a prompt."""


from typing import List
import re
import unicodedata
import string
import logging
import os


class LangProvider():
    """Base class for objects that provision language"""

    def __init__(self, config_root: dict = None) -> None:
        self.language = ""
        self.local_mode = False
        if config_root:
            # Extract configuration from the config_root
            langproviders_config = config_root.get("langproviders", {})
            # Get the first (and typically only) language provider config
            for model_type, config in langproviders_config.items():
                self.language = config.get("language", "")
                model_type = config.get("model_type", "")
                local_mode = config.get("local_mode", False)
                if model_type == "remote.RivaTranslator":
                    self.local_mode = local_mode
                break

        if self.language:
            self.source_lang, self.target_lang = self.language.split(",")
            if self.source_lang == self.target_lang:
                raise Exception(f"Source and target languages cannot be the same: {self.source_lang}")

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
        target_lang = getattr(self, 'target_lang', 'unknown')

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