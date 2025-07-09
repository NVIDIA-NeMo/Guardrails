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

import json

from tqdm import tqdm

from nemoguardrails.evaluate.utils_translate import (
    _extract_target_language,
    _load_langprovider,
    get_translation_cache,
    get_translation_cache_name,
)
from nemoguardrails.llm.models.initializer import init_llm_model
from nemoguardrails.rails.llm.config import Model


def initialize_llm(model_config: Model):
    """Initializes the model from LLM provider."""

    return init_llm_model(
        model_name=model_config.model,
        provider_name=model_config.engine,
        kwargs=model_config.parameters,
    )


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
        translate_to = _extract_target_language(translation_config)
        service_name = get_translation_cache_name(translator)
        cache = get_translation_cache(service_name)
        translated_dataset = []

        print(f"🔄 Starting translation to {translate_to}...")
        print(f"📊 Total items to process: {len(dataset)}")

        # Display progress bar with tqdm
        for item in tqdm(dataset, desc="Translating", unit="item"):
            if isinstance(item, dict):
                # For JSON format, translate specific fields
                translated_item = item.copy()
                for field in ["answer", "question", "evidence"]:
                    if field in translated_item:
                        original_text = translated_item[field]
                        # Check cache first
                        cached_translation = cache.get(original_text, translate_to)
                        if cached_translation:
                            translated_item[field] = cached_translation
                        else:
                            # Translate and cache
                            translated_text = translator._translate(original_text)
                            translated_item[field] = translated_text
                            cache.set(original_text, translate_to, translated_text)
                translated_dataset.append(translated_item)
            else:
                # For text format
                original_text = item.strip()
                # Check cache first
                cached_translation = cache.get(original_text, translate_to)
                if cached_translation:
                    translated_dataset.append(cached_translation)
                else:
                    # Translate and cache
                    translated_text = translator._translate(original_text)
                    translated_dataset.append(translated_text)
                    cache.set(original_text, translate_to, translated_text)

        # Print cache statistics
        stats = cache.get_cache_stats()
        print(f"✅ Translation completed!")
        print(
            f"📈 Translation cache stats: {stats['total_entries']} entries, {stats['cache_size_mb']:.2f} MB"
        )

        return translated_dataset

    return dataset
