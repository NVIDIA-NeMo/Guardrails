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

from typing import Dict, List, Optional

from nemoguardrails.manifests.config_schema import (
    Field,
    RailConfigBaseModel,
    RailConfigSpec,
    model_validator,
    rail_field,
)


class LanguageDetectorConfig(RailConfigBaseModel):
    """Configuration for the language detector input rail."""

    allowed_languages: List[str] = Field(
        default_factory=lambda: ["en"],
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

    @model_validator(mode="after")
    def _normalize_language_lists(self) -> "LanguageDetectorConfig":
        normalized_allowed = self._normalize(self.allowed_languages, "allowed_languages")
        if self.blocked_languages is not None:
            self.blocked_languages = self._normalize(self.blocked_languages, "blocked_languages")
        self.allowed_languages = normalized_allowed
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
