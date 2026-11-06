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

"""Bind reviewed LiteLLM Responses metadata without relaxing text boundaries.

The event model annotation extends the native surface. Requests, buffered
responses, lifecycle response objects, and output-item policy remain OpenAI's.
See responses.md for source, recording provenance and explicit selection.
"""

from dataclasses import replace
from typing import Annotated, Any

from pydantic import BaseModel

from nemoguardrails.server.experimental.provider.projection_policy import (
    PolicyModel,
    opaque,
)
from nemoguardrails.server.experimental.provider.stream import StreamBinding
from nemoguardrails.server.experimental.provider.stream_classifier import StreamClassifierDefinition
from nemoguardrails.server.experimental.provider.stream_policy import (
    build_policy_stream_classifier,
    stream_shape_coverage,
    stream_text_path,
)
from nemoguardrails.server.experimental.providers.openai.responses import stream_projection as models
from nemoguardrails.server.experimental.providers.openai.responses.endpoint import RESPONSES_ENDPOINT
from nemoguardrails.server.experimental.providers.openai.responses.stream_classifier import (
    STREAM_CONTRACT as OPENAI_STREAM_CONTRACT,
)
from nemoguardrails.server.experimental.providers.openai.responses.stream_classifier import (
    STREAM_NON_DATA_SHAPE,
    STREAM_SENTINELS,
)
from nemoguardrails.server.experimental.providers.openai.responses.stream_classifier import (
    STREAM_RULES as OPENAI_STREAM_RULES,
)
from nemoguardrails.server.experimental.providers.openai.responses.stream_hooks import ResponsesStreamHooks


class LiteLLMResponsesEventMetadata(PolicyModel):
    """Accept the gateway's model annotation without using it to select events."""

    model: Annotated[Any, opaque()] = None


class LiteLLMTextDeltaEvent(models.ResponsesTextDeltaEventProjection, LiteLLMResponsesEventMetadata):
    """Keep the delta as the only newly generated text subject."""


class LiteLLMTextDoneEvent(models.ResponsesTextDoneEventProjection, LiteLLMResponsesEventMetadata):
    """Keep completed text subject to snapshot consistency checks."""


class LiteLLMPartAddedEvent(models.ResponsesContentPartAddedEventProjection, LiteLLMResponsesEventMetadata):
    """Retain the initial text part and its inherited content restrictions."""


class LiteLLMPartDoneEvent(models.ResponsesContentPartDoneEventProjection, LiteLLMResponsesEventMetadata):
    """Retain the completed text part as a checked snapshot."""


class LiteLLMMessageAddedEvent(models.ResponsesMessageAddedEventProjection, LiteLLMResponsesEventMetadata):
    """Preserve message identity and single-part cardinality."""


class LiteLLMMessageDoneEvent(models.ResponsesMessageDoneEventProjection, LiteLLMResponsesEventMetadata):
    """Preserve message completion checks and the original item identity."""


class LiteLLMReasoningAddedEvent(models.ResponsesReasoningAddedEventProjection, LiteLLMResponsesEventMetadata):
    """Permit reviewed reasoning metadata, never new plaintext reasoning."""


class LiteLLMReasoningDoneEvent(models.ResponsesReasoningDoneEventProjection, LiteLLMResponsesEventMetadata):
    """Keep reasoning summaries and content constrained to empty values."""


class LiteLLMCreatedEvent(models.ResponsesCreatedEventProjection, LiteLLMResponsesEventMetadata):
    """Check the creation snapshot with the reviewed response metadata."""


class LiteLLMInProgressEvent(models.ResponsesInProgressEventProjection, LiteLLMResponsesEventMetadata):
    """Check the progress snapshot without introducing a separate text target."""


class LiteLLMQueuedEvent(models.ResponsesQueuedEventProjection, LiteLLMResponsesEventMetadata):
    """Recognize queued metadata without enabling background requests."""


class LiteLLMCompletedEvent(models.ResponsesCompletedEventProjection, LiteLLMResponsesEventMetadata):
    """Check the native terminal snapshot against accumulated deltas."""


class LiteLLMFailedEvent(models.ResponsesFailedEventProjection, LiteLLMResponsesEventMetadata):
    """Keep any output in failed responses subject to snapshot checks."""


class LiteLLMIncompleteEvent(models.ResponsesIncompleteEventProjection, LiteLLMResponsesEventMetadata):
    """Keep partial output guarded even when generation is incomplete."""


class LiteLLMErrorEvent(models.ResponsesErrorEventProjection, LiteLLMResponsesEventMetadata):
    """Accept model metadata on the inherited flat diagnostic error shape."""


_EVENT_MODELS: dict[type[BaseModel], type[PolicyModel]] = {
    models.ResponsesTextDeltaEventProjection: LiteLLMTextDeltaEvent,
    models.ResponsesTextDoneEventProjection: LiteLLMTextDoneEvent,
    models.ResponsesContentPartAddedEventProjection: LiteLLMPartAddedEvent,
    models.ResponsesContentPartDoneEventProjection: LiteLLMPartDoneEvent,
    models.ResponsesMessageAddedEventProjection: LiteLLMMessageAddedEvent,
    models.ResponsesMessageDoneEventProjection: LiteLLMMessageDoneEvent,
    models.ResponsesReasoningAddedEventProjection: LiteLLMReasoningAddedEvent,
    models.ResponsesReasoningDoneEventProjection: LiteLLMReasoningDoneEvent,
    models.ResponsesCreatedEventProjection: LiteLLMCreatedEvent,
    models.ResponsesInProgressEventProjection: LiteLLMInProgressEvent,
    models.ResponsesQueuedEventProjection: LiteLLMQueuedEvent,
    models.ResponsesCompletedEventProjection: LiteLLMCompletedEvent,
    models.ResponsesFailedEventProjection: LiteLLMFailedEvent,
    models.ResponsesIncompleteEventProjection: LiteLLMIncompleteEvent,
    models.ResponsesErrorEventProjection: LiteLLMErrorEvent,
}
STREAM_RULES = tuple(
    replace(
        rule,
        model=_EVENT_MODELS[rule.model],
        text_path=stream_text_path(_EVENT_MODELS[rule.model]) if rule.text_path is not None else None,
    )
    for rule in OPENAI_STREAM_RULES
)
STREAM_CONTRACT = replace(
    OPENAI_STREAM_CONTRACT,
    projection_id="openai.litellm.responses.stream.text.v1",
    shapes=stream_shape_coverage(STREAM_RULES, sentinels=STREAM_SENTINELS, non_data_shape=STREAM_NON_DATA_SHAPE),
)
STREAM_CLASSIFIER = build_policy_stream_classifier(
    StreamClassifierDefinition(
        subject=STREAM_CONTRACT.projection_id,
        contract=STREAM_CONTRACT,
        rules=STREAM_RULES,
        event_type_field="type",
        sentinels=STREAM_SENTINELS,
        non_data_shape=STREAM_NON_DATA_SHAPE,
    )
)


LITELLM_RESPONSES_ENDPOINT = replace(
    RESPONSES_ENDPOINT,
    contract_name="litellm_responses",
    operation="LiteLLM Responses",
    stream=StreamBinding(STREAM_CLASSIFIER, ResponsesStreamHooks),
)
