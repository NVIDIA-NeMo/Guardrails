# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import logging
from typing import Dict, FrozenSet, List, Optional

from nemoguardrails.manifests.config_schema import (
    Field,
    RailConfigBaseModel,
    RailConfigSpec,
    model_validator,
    rail_field,
)

FASTTEXT_LANGUAGES: FrozenSet[str] = frozenset(
    {
        "af",
        "als",
        "am",
        "an",
        "ar",
        "arz",
        "as",
        "ast",
        "av",
        "az",
        "azb",
        "ba",
        "bar",
        "bcl",
        "be",
        "bg",
        "bh",
        "bn",
        "bo",
        "bpy",
        "br",
        "bs",
        "bxr",
        "ca",
        "cbk",
        "ce",
        "ceb",
        "ckb",
        "co",
        "cs",
        "cv",
        "cy",
        "da",
        "de",
        "diq",
        "dsb",
        "dty",
        "dv",
        "el",
        "eml",
        "en",
        "eo",
        "es",
        "et",
        "eu",
        "fa",
        "fi",
        "fr",
        "frr",
        "fy",
        "ga",
        "gd",
        "gl",
        "gn",
        "gom",
        "gu",
        "gv",
        "he",
        "hi",
        "hif",
        "hr",
        "hsb",
        "ht",
        "hu",
        "hy",
        "ia",
        "id",
        "ie",
        "ilo",
        "io",
        "is",
        "it",
        "ja",
        "jbo",
        "jv",
        "ka",
        "kk",
        "km",
        "kn",
        "ko",
        "krc",
        "ku",
        "kv",
        "kw",
        "ky",
        "la",
        "lb",
        "lez",
        "li",
        "lmo",
        "lo",
        "lrc",
        "lt",
        "lv",
        "mai",
        "mg",
        "mhr",
        "min",
        "mk",
        "ml",
        "mn",
        "mr",
        "mrj",
        "ms",
        "mt",
        "mwl",
        "my",
        "myv",
        "mzn",
        "nah",
        "nap",
        "nds",
        "ne",
        "new",
        "nl",
        "nn",
        "no",
        "oc",
        "or",
        "os",
        "pa",
        "pam",
        "pfl",
        "pl",
        "pms",
        "pnb",
        "ps",
        "pt",
        "qu",
        "rm",
        "ro",
        "ru",
        "rue",
        "sa",
        "sah",
        "sc",
        "scn",
        "sco",
        "sd",
        "sh",
        "si",
        "sk",
        "sl",
        "so",
        "sq",
        "sr",
        "su",
        "sv",
        "sw",
        "ta",
        "te",
        "tg",
        "th",
        "tk",
        "tl",
        "tr",
        "tt",
        "tyv",
        "ug",
        "uk",
        "ur",
        "uz",
        "vec",
        "vep",
        "vi",
        "vls",
        "vo",
        "wa",
        "war",
        "wuu",
        "xal",
        "xmf",
        "yi",
        "yo",
        "yue",
        "zh",
    }
)

log = logging.getLogger(__name__)


class LanguageDetectorConfig(RailConfigBaseModel):
    """Configuration for the language detector input rail."""

    allowed_languages: Optional[List[str]] = Field(
        default=None,
        description="Language codes (fastText) that the user message may be written in. "
        "Messages in any other language are blocked. "
        "Mutually exclusive with blocked_languages. "
        "See https://fasttext.cc/docs/en/language-identification.html for the supported codes.",
    )
    blocked_languages: Optional[List[str]] = Field(
        default=None,
        description="Language codes (fastText) that the user message may not be written in. "
        "Messages in any other language are allowed. "
        "Mutually exclusive with allowed_languages. "
        "Example: ['en', 'es']",
    )
    refusal_messages: Optional[Dict[str, str]] = Field(
        default=None,
        description="Custom refusal messages per language code, used when a message is blocked. "
        "Languages without a message fall back to the built-in defaults, then to English. "
        "Example: {'en': 'Sorry, English only.', 'es': 'Lo siento, solo inglés.'}",
    )
    num_characters_for_prediction: Optional[int] = Field(
        default=80,
        description="The number of characters to use for language detection."
        "The first $N characters will be used, and any subsequent characters will not be considered in the prediction."
        "Defaults to 80 - longer windows will be more robust ",
    )
    fail_open: bool = Field(
        default=False,
        description=(
            "If True, allow the message when language detection fails, regardless of the language list. "
            "If False, block the message when language detection fails."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _reject_both_language_lists(cls, data: object) -> object:
        if (
            isinstance(data, dict)
            and data.get("allowed_languages") is not None
            and data.get("blocked_languages") is not None
        ):
            raise ValueError("Specify either allowed_languages or blocked_languages, not both.")
        return data

    @model_validator(mode="before")
    @classmethod
    def _set_default_language_fallback(cls, data: object) -> object:
        if isinstance(data, dict) and data.get("allowed_languages") is None and data.get("blocked_languages") is None:
            data["allowed_languages"] = ["en"]  # type: ignore
        return data

    @model_validator(mode="after")
    def _normalize_and_verify_language_lists(self) -> "LanguageDetectorConfig":
        specified_languages = []
        if self.allowed_languages is not None:
            self.allowed_languages = self._normalize(self.allowed_languages, "allowed_languages")
            specified_languages.extend(self.allowed_languages)
        elif self.blocked_languages is not None:
            self.blocked_languages = self._normalize(self.blocked_languages, "blocked_languages")
            specified_languages.extend(self.blocked_languages)

        invalid_languages = [
            language_code for language_code in specified_languages if language_code not in FASTTEXT_LANGUAGES
        ]
        if invalid_languages:
            raise ValueError("Received invalid or unsupported language codes: {}".format(", ".join(invalid_languages)))
        return self

    @staticmethod
    def _normalize(languages: List[str], field_name: str) -> List[str]:
        if not languages:
            raise ValueError(f"{field_name} must contain at least one language code")
        normalized: List[str] = []
        for lang in languages:
            code = lang.strip().lower()
            if code not in normalized:
                normalized.append(code)
        return normalized


def build_config_spec() -> RailConfigSpec:
    return RailConfigSpec(
        annotation=Optional[LanguageDetectorConfig],
        field_info=rail_field(
            default_factory=LanguageDetectorConfig,
            description="Configuration for the language detector rail.",
        ),
        exports={
            "LanguageDetectorConfig": LanguageDetectorConfig,
        },
    )
