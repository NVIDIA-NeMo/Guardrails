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

"""Bind reviewed Responses event roles to text paths derived from typed policy."""

from nemoguardrails.server.experimental.provider.projection_policy import PolicyModel
from nemoguardrails.server.experimental.provider.stream import (
    StreamCapabilityProfile,
    StreamEventRole,
    StreamProjectionContract,
)
from nemoguardrails.server.experimental.provider.stream_classifier import StreamClassifierDefinition, StreamEventRule
from nemoguardrails.server.experimental.provider.stream_policy import (
    build_policy_stream_classifier,
    stream_shape_coverage,
    stream_text_path,
)
from nemoguardrails.server.experimental.providers.openai.responses import stream_projection as models

PROJECTION_ID = "openai.responses.stream.text.v1"
STREAM_SENTINELS = ((b"[DONE]", "done"),)
STREAM_NON_DATA_SHAPE = "keepalive"


def _snapshot(
    shape: str, model: type[PolicyModel], *, item_type: str | None = None, allow_empty: bool = False
) -> StreamEventRule:
    """Build one explicit event rule; extraction and coverage remain derived."""
    event_type = shape.split(":", 1)[0]
    match = (("type", event_type),) + ((("item.type", item_type),) if item_type else ())
    return StreamEventRule(
        shape=shape,
        model=model,
        role=StreamEventRole.TEXT_SNAPSHOT,
        match=match,
        text_path=stream_text_path(model),
        missing_text_role=StreamEventRole.TEXT_SNAPSHOT if allow_empty else None,
        missing_text_value="" if allow_empty else None,
    )


STREAM_RULES = (
    StreamEventRule(
        shape="response.output_text.delta",
        model=models.ResponsesTextDeltaEventProjection,
        role=StreamEventRole.GUARDED_TEXT,
        match=(("type", "response.output_text.delta"),),
        text_path=stream_text_path(models.ResponsesTextDeltaEventProjection),
    ),
    _snapshot("response.output_text.done", models.ResponsesTextDoneEventProjection),
    _snapshot("response.content_part.added", models.ResponsesContentPartAddedEventProjection),
    _snapshot("response.content_part.done", models.ResponsesContentPartDoneEventProjection),
    _snapshot(
        "response.output_item.added:message",
        models.ResponsesMessageAddedEventProjection,
        item_type="message",
        allow_empty=True,
    ),
    _snapshot(
        "response.output_item.done:message",
        models.ResponsesMessageDoneEventProjection,
        item_type="message",
        allow_empty=True,
    ),
    _snapshot("response.created", models.ResponsesCreatedEventProjection, allow_empty=True),
    _snapshot("response.in_progress", models.ResponsesInProgressEventProjection, allow_empty=True),
    _snapshot("response.queued", models.ResponsesQueuedEventProjection, allow_empty=True),
    _snapshot("response.completed", models.ResponsesCompletedEventProjection, allow_empty=True),
    _snapshot("response.failed", models.ResponsesFailedEventProjection, allow_empty=True),
    _snapshot("response.incomplete", models.ResponsesIncompleteEventProjection, allow_empty=True),
    StreamEventRule(
        shape="response.output_item.added:reasoning",
        model=models.ResponsesReasoningAddedEventProjection,
        role=StreamEventRole.OPAQUE_METADATA,
        match=(("type", "response.output_item.added"), ("item.type", "reasoning")),
    ),
    StreamEventRule(
        shape="response.output_item.done:reasoning",
        model=models.ResponsesReasoningDoneEventProjection,
        role=StreamEventRole.OPAQUE_METADATA,
        match=(("type", "response.output_item.done"), ("item.type", "reasoning")),
    ),
    StreamEventRule(
        shape="error",
        model=models.ResponsesErrorEventProjection,
        role=StreamEventRole.PROVIDER_ERROR,
        match=(("type", "error"),),
    ),
)

STREAM_CONTRACT = StreamProjectionContract(
    projection_id=PROJECTION_ID,
    profile=StreamCapabilityProfile.SINGLE_TEXT_DELTA_V1,
    shapes=stream_shape_coverage(STREAM_RULES, sentinels=STREAM_SENTINELS, non_data_shape=STREAM_NON_DATA_SHAPE),
)
STREAM_CLASSIFIER = build_policy_stream_classifier(
    StreamClassifierDefinition(
        subject=PROJECTION_ID,
        contract=STREAM_CONTRACT,
        rules=STREAM_RULES,
        event_type_field="type",
        sentinels=STREAM_SENTINELS,
        non_data_shape=STREAM_NON_DATA_SHAPE,
    )
)
