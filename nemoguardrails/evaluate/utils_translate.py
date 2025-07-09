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

import hashlib
import importlib
import json
import logging
import os
from pathlib import Path

import yaml
from tqdm import tqdm

from nemoguardrails.evaluate.langproviders.base import LangProvider


class TranslationCache:
    """Cache for translation results to avoid repeated API calls."""

    def __init__(
        self, cache_dir: str = "translation_cache", service_name: str = "default"
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        # Generate cache file name based on service name
        safe_service_name = (
            service_name.replace("/", "_").replace("\\", "_").replace(":", "_")
        )
        self.cache_file = self.cache_dir / f"translations_{safe_service_name}.json"
        logging.debug(f"cache_file: {self.cache_file}")
        self.cache = self._load_cache()

    def _load_cache(self):
        """Load existing cache from file."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logging.warning(f"Failed to load translation cache: {e}")
                return {}
        return {}

    def _save_cache(self):
        """Save cache to file."""
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except IOError as e:
            logging.error(f"Failed to save translation cache: {e}")

    def _get_cache_key(self, text: str, target_lang: str) -> str:
        """Generate cache key from text and target language."""
        # Create a hash of the text and target language
        content = f"{text}:{target_lang}"
        return content

    def get(self, text: str, target_lang: str) -> str:
        """Get translated text from cache if available."""
        cache_key = self._get_cache_key(text, target_lang)
        return self.cache.get(cache_key)

    def set(self, text: str, target_lang: str, translated_text: str):
        """Store translated text in cache."""
        cache_key = self._get_cache_key(text, target_lang)
        self.cache[cache_key] = translated_text
        self._save_cache()

    def get_cache_stats(self):
        """Get statistics about the cache."""
        cache_size_bytes = (
            os.path.getsize(self.cache_file) if self.cache_file.exists() else 0
        )
        cache_size_mb = cache_size_bytes / (1024 * 1024)

        return {
            "total_entries": len(self.cache),
            "cache_size_bytes": cache_size_bytes,
            "cache_size_mb": cache_size_mb,
            "cache_file": str(self.cache_file),
        }


# Global dictionary to store translation cache instances
_translation_caches = {}


def get_translation_cache(service_name: str = "default") -> TranslationCache:
    """Get or create translation cache instance for the specified service."""
    if service_name not in _translation_caches:
        _translation_caches[service_name] = TranslationCache(service_name=service_name)
    return _translation_caches[service_name]


def get_translation_cache_name(translator: LangProvider) -> str:
    # Get translation service information to create cache instance
    service_name = translator.__class__.__name__

    # For local services, include model name as well
    if hasattr(translator, "model_name"):
        # Generate safe filename from model name
        safe_model_name = (
            translator.model_name.replace("/", "_").replace("\\", "_").replace(":", "_")
        )
        service_name = f"{service_name}_{safe_model_name}"
    return service_name


def load_dataset(dataset_path: str, translation_config: str = None):
    """Loads a dataset from a file with optional translation."""

    with open(dataset_path, "r") as f:
        if dataset_path.endswith(".json"):
            dataset = json.load(f)
        else:
            dataset = f.readlines()

    # If translation is needed
    if translation_config:
        translator = _load_langprovider(translation_config)
        service_name = get_translation_cache_name(translator)
        cache = get_translation_cache(service_name)

        translated_dataset = []

        print(f"🔄 Starting translation...")
        print(f"📊 Total items to process: {len(dataset)}")
        print(f"🔧 Using translation service: {service_name}")

        # Display progress bar with tqdm
        for item in tqdm(dataset, desc="Translating", unit="item"):
            if isinstance(item, dict):
                # For JSON format, translate specific fields
                translated_item = item.copy()
                for field in ["answer", "question", "evidence"]:
                    if field in translated_item:
                        original_text = translated_item[field]
                        # Check cache first
                        cached_translation = cache.get(
                            original_text, translator.target_lang
                        )
                        if cached_translation:
                            translated_item[field] = cached_translation
                        else:
                            # Translate and cache
                            translated_text = translator._translate(original_text)
                            translated_item[field] = translated_text
                            cache.set(
                                original_text, translator.target_lang, translated_text
                            )
                translated_dataset.append(translated_item)
            else:
                # For text format
                original_text = item.strip()
                # Check cache first
                cached_translation = cache.get(original_text, translator.target_lang)
                if cached_translation:
                    translated_dataset.append(cached_translation)
                else:
                    # Translate and cache
                    translated_text = translator._translate(original_text)
                    translated_dataset.append(translated_text)
                    cache.set(original_text, translator.target_lang, translated_text)

        # Print cache statistics
        stats = cache.get_cache_stats()
        print(f"✅ Translation completed!")
        print(
            f"📈 Translation cache stats: {stats['total_entries']} entries, {stats['cache_size_mb']:.2f} MB"
        )
        print(f"💾 Cache file: {stats['cache_file']}")

        return translated_dataset

    return dataset


class PluginConfigurationError(Exception):
    """Exception raised when a plugin configuration is invalid."""

    pass


def _load_plugin(path: str, config_root: dict):
    """Load a plugin class from the given path."""
    try:
        # Split the path to get module and class name
        module_path, class_name = path.rsplit(".", 1)

        # Import the module
        module = importlib.import_module(module_path)

        # Get the class from the module
        plugin_class = getattr(module, class_name)

        # Create an instance with the config
        instance = plugin_class(config_root)
        return instance

    except (ImportError, AttributeError, ValueError) as e:
        raise PluginConfigurationError(
            f"Failed to load plugin '{path}': {str(e)}"
        ) from e


def _extract_target_language(config_yaml: str) -> str:
    """Extract target language from translation configuration file."""
    with open(config_yaml, "r") as f:
        config = yaml.safe_load(f)
    language_service = config["langproviders"][0]
    source_lang, target_lang = language_service["language"].split(",")
    return target_lang


def _load_langprovider(config_yaml: str = None) -> LangProvider:
    """Load a single language provider based on the configuration provided."""
    langprovider_instance = None

    # If no config file is provided, raise an error
    if config_yaml is None:
        raise PluginConfigurationError(
            "No configuration file provided. Please specify a translation configuration file."
        )

    with open(config_yaml, "r") as f:
        config = yaml.safe_load(f)
    language_service = config["langproviders"][0]
    langprovider_config = {
        "langproviders": {language_service["model_type"]: language_service}
    }
    logging.debug(f"language provision service: {language_service['language']}")
    source_lang, target_lang = language_service["language"].split(",")
    model_type = language_service["model_type"]
    try:
        if "." in model_type:
            # For formats like remote.RivaTranslator
            module_name, class_name = model_type.split(".", 1)
            module_path = f"nemoguardrails.evaluate.langproviders.{module_name}"

        path = f"{module_path}.{class_name}"
        langprovider_instance = _load_plugin(
            path=path,
            config_root=langprovider_config,
        )
        print(f"langprovider_instance: {langprovider_instance}")
    except Exception as e:
        raise PluginConfigurationError(
            f"Failed to load '{language_service['language']}' langprovider of type '{model_type}': {str(e)}"
        ) from e
    return langprovider_instance
