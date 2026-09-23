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

from typing import Dict, Optional

from nemoguardrails.manifests.config_schema import (
    Field,
    RailConfigBaseModel,
    RailConfigSpec,
    rail_field,
)


class MultilingualConfig(RailConfigBaseModel):
    """Configuration for multilingual refusal messages."""

    enabled: bool = Field(
        default=False,
        description="If True, detect the language of user input and return refusal messages in the same language. "
        "Supported languages: en (English), es (Spanish), zh (Chinese), de (German), fr (French), "
        "hi (Hindi), ja (Japanese), ar (Arabic), th (Thai).",
    )
    refusal_messages: Optional[Dict[str, str]] = Field(
        default=None,
        description="Custom refusal messages per language code. "
        "If not specified, built-in defaults are used. "
        "Example: {'en': 'Sorry, I cannot help.', 'es': 'Lo siento, no puedo ayudar.'}",
    )


class ReasoningConfig(RailConfigBaseModel):
    """Configuration for reasoning mode in content safety models."""

    enabled: bool = Field(
        default=False,
        description="If True, enable reasoning mode (with <think> traces) for content safety models. "
        "If False, use low-latency mode without reasoning traces.",
    )


class ContentSafetyConfig(RailConfigBaseModel):
    """Configuration data for content safety rails."""

    multilingual: MultilingualConfig = Field(
        default_factory=MultilingualConfig,
        description="Configuration for multilingual refusal messages.",
    )

    reasoning: ReasoningConfig = Field(
        default_factory=ReasoningConfig,
        description="Configuration for reasoning mode in content safety models.",
    )


def build_config_spec() -> RailConfigSpec:
    return RailConfigSpec(
        annotation=Optional[ContentSafetyConfig],
        field_info=rail_field(
            default_factory=ContentSafetyConfig,
            description="Configuration for content safety rails.",
        ),
        exports={
            "ContentSafetyConfig": ContentSafetyConfig,
            "MultilingualConfig": MultilingualConfig,
            "ReasoningConfig": ReasoningConfig,
        },
    )
