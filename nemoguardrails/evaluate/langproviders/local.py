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


"""Local language providers & translators."""


from typing import List

import torch

from nemoguardrails.evaluate.langproviders.base import TranslationProvider


class LocalHFTranslator(TranslationProvider):
    """Local translation using Huggingface transformer models: Many-2-Many m2m100 or MarianMT Helsinki-NLP/opus-mt-* models

    Reference:
      - https://huggingface.co/facebook/m2m100_1.2B
      - https://huggingface.co/facebook/m2m100_418M
      - https://huggingface.co/docs/transformers/model_doc/marian
    """

    DEFAULT_PARAMS = {
        "model_name": "Helsinki-NLP/opus-mt-{}",
    }
    lang_overrides = {
        "ja": "jap",
    }

    def __init__(self, config: dict = {}) -> None:
        self._load_config(config=config)

        import torch.multiprocessing as mp

        # set_start_method for consistency, translation does not utilize multiprocessing
        mp.set_start_method("spawn", force=True)

        self.device = self._select_hf_device()
        super().__init__(config=config)

    def _load_config(self, config: dict = {}):
        """Load configuration from config."""
        if config:
            # Extract configuration from the config
            langproviders_config = config.get("langproviders", {})
            # Get the first (and typically only) language provider config
            for _, each_config in langproviders_config.items():
                self.model_name = each_config.get(
                    "model_name", self.DEFAULT_PARAMS["model_name"]
                )
                break
        else:
            self.model_name = self.DEFAULT_PARAMS["model_name"]

    def _select_hf_device(self):
        """Select the appropriate device for HuggingFace models."""
        try:
            if torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        except Exception as e:
            return "cpu"

    def _load_langprovider(self):
        if "m2m100" in self.model_name:
            from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

            # fmt: off
            # Reference: https://huggingface.co/facebook/m2m100_418M#languages-covered
            lang_support = {
                "af", "am", "ar", "ast", "az",
                "ba", "be", "bg", "bn", "br",
                "bs", "ca", "ceb", "cs", "cy",
                "da", "de", "el", "en", "es",
                "et", "fa", "ff", "fi", "fr",
                "fy", "ga", "gd", "gl", "gu",
                "ha", "he", "hi", "hr", "ht",
                "hu", "hy", "id", "ig", "ilo",
                "is", "it", "ja", "jv", "ka",
                "kk", "km", "kn", "ko", "lb",
                "lg", "ln", "lo", "lt", "lv",
                "mg", "mk", "ml", "mn", "mr",
                "ms", "my", "ne", "nl", "no",
                "ns", "oc", "or", "pa", "pl",
                "ps", "pt", "ro", "ru", "sd",
                "si", "sk", "sl", "so", "sq",
                "sr", "ss", "su", "sv", "sw",
                "ta", "th", "tl", "tn", "tr",
                "uk", "ur", "uz", "vi", "wo",
                "xh", "yi", "yo", "zh", "zu",
            }
            # fmt: on
            if not (
                self.source_lang in lang_support and self.target_lang in lang_support
            ):
                raise Exception(
                    f"Language pair {self.language} is not supported for this translation service."
                )

            self.model = M2M100ForConditionalGeneration.from_pretrained(
                self.model_name
            ).to(self.device)
            self.tokenizer = M2M100Tokenizer.from_pretrained(self.model_name)
        else:
            from transformers import MarianMTModel, MarianTokenizer

            # if model is not m2m100 expect the model name to be "Helsinki-NLP/opus-mt-{}" where the format string
            # is replace with the language path defined in the configuration as self.source_lang-self.target_lang
            # validation of all supported pairs is deferred in favor of allowing the download to raise exception
            # when no published model exists with the pair requested in the name.
            self.target_lang = self.lang_overrides.get(
                self.target_lang, self.target_lang
            )
            model_suffix = f"{self.source_lang}-{self.target_lang}"
            model_name = self.model_name.format(model_suffix)
            # Save the processed model_name
            self.model_name = model_name
            self.model = MarianMTModel.from_pretrained(model_name).to(self.device)
            self.tokenizer = MarianTokenizer.from_pretrained(model_name)

    def _translate(self, text: str) -> str:
        if "m2m100" in self.model_name:
            self.tokenizer.src_lang = self.source_lang

            encoded_text = self.tokenizer(text, return_tensors="pt").to(self.device)

            translated = self.model.generate(
                **encoded_text,
                forced_bos_token_id=self.tokenizer.get_lang_id(self.target_lang),
            )

            translated_text = self.tokenizer.batch_decode(
                translated, skip_special_tokens=True
            )[0]

            return translated_text
        else:
            # this assumes MarianMTModel type
            source_text = self.tokenizer([text], return_tensors="pt").to(self.device)

            translated = self.model.generate(**source_text)

            translated_text = self.tokenizer.batch_decode(
                translated, skip_special_tokens=True
            )[0]

            return translated_text
