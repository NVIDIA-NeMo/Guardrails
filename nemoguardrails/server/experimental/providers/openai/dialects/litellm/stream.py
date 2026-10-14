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

"""Declare LiteLLM stream fields while retaining OpenAI event semantics.

LiteLLM's root and delta extension bags are optional, nullable, and closed.
No nonempty streaming extensions have been reviewed. Source-declared content
arrays may be absent, null, or empty, but never carry uninspected content.
The classifier derives coverage and text paths from these rebound models;
the endpoint uses the existing per-stream OpenAI lifecycle hooks.
"""

from dataclasses import replace
from typing import Annotated, Any, ClassVar

from nemoguardrails.server.experimental.provider.projection_policy import (
    ObjectPolicy,
    PolicyModel,
    constrained,
    field_coverage,
    guarded,
)
from nemoguardrails.server.experimental.provider.stream_classifier import StreamClassifierDefinition
from nemoguardrails.server.experimental.provider.stream_policy import build_policy_stream_classifier, stream_text_path
from nemoguardrails.server.experimental.providers.openai.chat_completions.stream_classifier import (
    STREAM_CONTRACT as OPENAI_STREAM_CONTRACT,
)
from nemoguardrails.server.experimental.providers.openai.chat_completions.stream_classifier import (
    STREAM_NON_DATA_SHAPE,
    STREAM_SENTINELS,
)
from nemoguardrails.server.experimental.providers.openai.chat_completions.stream_classifier import (
    STREAM_RULES as OPENAI_STREAM_RULES,
)
from nemoguardrails.server.experimental.providers.openai.chat_completions.stream_projection import (
    ChatCompletionsStreamChoiceProjection,
    ChatCompletionsStreamDeltaProjection,
    ChatCompletionsStreamPayloadProjection,
)


class LiteLLMStreamFields(PolicyModel):
    """Recognize an empty extension bag without authorizing arbitrary contents."""


class LiteLLMStreamDeltaProjection(ChatCompletionsStreamDeltaProjection):
    """Accept source-declared empty content fields, preserving the text target."""

    policy: ClassVar[ObjectPolicy] = replace(ChatCompletionsStreamDeltaProjection.policy, source="Delta")

    provider_specific_fields: Annotated[LiteLLMStreamFields | None, constrained()] = None
    images: Annotated[list[Any] | None, constrained(reason="core_capability.multimodal_content", max_length=0)] = None
    thinking_blocks: Annotated[
        list[Any] | None, constrained(reason="core_capability.reasoning_content", max_length=0)
    ] = None
    reasoning_items: Annotated[
        list[Any] | None, constrained(reason="core_capability.reasoning_content", max_length=0)
    ] = None
    annotations: Annotated[list[Any] | None, constrained(reason="core_capability.citation_content", max_length=0)] = (
        None
    )


class LiteLLMStreamChoiceProjection(ChatCompletionsStreamChoiceProjection):
    """Keep single-choice constraints and bind the LiteLLM delta model."""

    delta: Annotated[LiteLLMStreamDeltaProjection, guarded()]


class LiteLLMStreamPayloadProjection(ChatCompletionsStreamPayloadProjection):
    """Recognize LiteLLM's root extension bag, including on usage-only events."""

    policy: ClassVar[ObjectPolicy] = replace(
        ChatCompletionsStreamPayloadProjection.policy, source="ModelResponseStream"
    )

    choices: Annotated[list[LiteLLMStreamChoiceProjection], guarded(max_length=1)]
    provider_specific_fields: Annotated[LiteLLMStreamFields | None, constrained()] = None


STREAM_RULES = tuple(
    replace(rule, model=LiteLLMStreamPayloadProjection, text_path=stream_text_path(LiteLLMStreamPayloadProjection))
    if rule.model is ChatCompletionsStreamPayloadProjection
    else rule
    for rule in OPENAI_STREAM_RULES
)
STREAM_CONTRACT = replace(
    OPENAI_STREAM_CONTRACT,
    projection_id="openai.litellm.chat_completions.stream.text.v1",
    fields=field_coverage(LiteLLMStreamPayloadProjection),
)
STREAM_CLASSIFIER = build_policy_stream_classifier(
    StreamClassifierDefinition(
        subject=STREAM_CONTRACT.projection_id,
        contract=STREAM_CONTRACT,
        rules=STREAM_RULES,
        sentinels=STREAM_SENTINELS,
        non_data_shape=STREAM_NON_DATA_SHAPE,
    )
)
