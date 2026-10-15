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

"""Typed Responses SSE shapes; protocol ordering belongs to stream hooks."""

from dataclasses import replace
from typing import Annotated, Any, ClassVar, Literal

from nemoguardrails.server.experimental.provider.payload import GuardedContentModel
from nemoguardrails.server.experimental.provider.projection_policy import (
    ArrayVariant,
    ObjectPolicy,
    PolicyModel,
    constrained,
    disabled,
    guarded,
)
from nemoguardrails.server.experimental.providers.openai.responses.response_projection import (
    RESPONSE_METADATA_FIELDS,
    ResponsesAssistantMessageProjection,
    ResponsesOutputTextProjection,
    ResponsesReasoningItemProjection,
)


class ResponsesSnapshotTextProjection(ResponsesOutputTextProjection):
    """An empty initial text snapshot is valid; annotations and logprobs stay restricted."""

    text: Annotated[str, guarded("assistant", replaceable=False)]


class ResponsesSnapshotMessageProjection(ResponsesAssistantMessageProjection):
    """Allow an initially empty message, but never a second guarded text part."""

    policy: ClassVar[ObjectPolicy] = replace(
        ResponsesAssistantMessageProjection.policy,
        opaque=("phase", "status"),
        variant=ArrayVariant((("type", "message"), ("role", "assistant")), "at_most_one"),
    )
    id: Annotated[str | None, constrained()] = None
    content: Annotated[list[ResponsesSnapshotTextProjection], guarded(max_length=1)]


class ResponsesLifecycleResponseProjection(PolicyModel, GuardedContentModel):
    """Describe a full response snapshot without demanding a message before generation."""

    policy: ClassVar[ObjectPolicy] = ObjectPolicy(source="Response", opaque=RESPONSE_METADATA_FIELDS)
    output: Annotated[list[ResponsesReasoningItemProjection | ResponsesSnapshotMessageProjection], guarded()]
    output_text: Annotated[None, disabled("provider_integrity.duplicate_text")] = None


class ResponsesEventProjection(PolicyModel):
    """Validate sequence numbers when supplied, without requiring unrelated metadata."""

    sequence_number: Annotated[int | None, constrained(strict=True, ge=0)] = None


class ResponsesIndexedEventProjection(ResponsesEventProjection):
    """Track an output position without assuming reasoning always precedes text."""

    output_index: Annotated[int | None, constrained(strict=True, ge=0)] = None


class ResponsesTextEventProjection(ResponsesIndexedEventProjection):
    """Restrict content to the single text part and validate any supplied identity."""

    item_id: Annotated[str | None, constrained()] = None
    content_index: Annotated[int | None, constrained(strict=True, ge=0, le=0)] = None


class ResponsesTextDeltaEventProjection(ResponsesTextEventProjection):
    """Only this event introduces new guarded assistant text."""

    policy: ClassVar[ObjectPolicy] = ObjectPolicy(source="ResponseTextDeltaEvent", opaque=("obfuscation",))
    type: Annotated[Literal["response.output_text.delta"], constrained()]
    delta: Annotated[str, guarded("assistant", replaceable=False)]
    logprobs: Annotated[list[Any] | None, constrained(reason="provider_integrity.token_logprobs", max_length=0)] = None


class ResponsesTextDoneEventProjection(ResponsesTextEventProjection):
    """Repeat the accumulated text rather than introducing a second text source."""

    policy: ClassVar[ObjectPolicy] = ObjectPolicy(source="ResponseTextDoneEvent")
    type: Annotated[Literal["response.output_text.done"], constrained()]
    text: Annotated[str, guarded("assistant", replaceable=False)]
    logprobs: Annotated[list[Any] | None, constrained(reason="provider_integrity.token_logprobs", max_length=0)] = None


class ResponsesContentPartAddedEventProjection(ResponsesTextEventProjection):
    """A reviewed text part snapshot, including its initial empty state."""

    policy: ClassVar[ObjectPolicy] = ObjectPolicy(source="ResponseContentPartAddedEvent")
    type: Annotated[Literal["response.content_part.added"], constrained()]
    part: Annotated[ResponsesSnapshotTextProjection, guarded()]


class ResponsesContentPartDoneEventProjection(ResponsesTextEventProjection):
    """The final snapshot of the single text part."""

    policy: ClassVar[ObjectPolicy] = ObjectPolicy(source="ResponseContentPartDoneEvent")
    type: Annotated[Literal["response.content_part.done"], constrained()]
    part: Annotated[ResponsesSnapshotTextProjection, guarded()]


class ResponsesMessageAddedEventProjection(ResponsesIndexedEventProjection):
    """The assistant item is a snapshot, not opaque output."""

    policy: ClassVar[ObjectPolicy] = ObjectPolicy(source="ResponseOutputItemAddedEvent")
    type: Annotated[Literal["response.output_item.added"], constrained()]
    item: Annotated[ResponsesSnapshotMessageProjection, guarded()]


class ResponsesMessageDoneEventProjection(ResponsesIndexedEventProjection):
    """The completed assistant item must agree with previously observed deltas."""

    policy: ClassVar[ObjectPolicy] = ObjectPolicy(source="ResponseOutputItemDoneEvent")
    type: Annotated[Literal["response.output_item.done"], constrained()]
    item: Annotated[ResponsesSnapshotMessageProjection, guarded()]


class ResponsesReasoningAddedEventProjection(ResponsesIndexedEventProjection):
    """Allow only the reviewed, non-plaintext reasoning item."""

    policy: ClassVar[ObjectPolicy] = ObjectPolicy(source="ResponseOutputItemAddedEvent")
    type: Annotated[Literal["response.output_item.added"], constrained()]
    item: Annotated[ResponsesReasoningItemProjection, constrained()]


class ResponsesReasoningDoneEventProjection(ResponsesIndexedEventProjection):
    """Preserve completed reasoning metadata without inspecting encrypted values."""

    policy: ClassVar[ObjectPolicy] = ObjectPolicy(source="ResponseOutputItemDoneEvent")
    type: Annotated[Literal["response.output_item.done"], constrained()]
    item: Annotated[ResponsesReasoningItemProjection, constrained()]


class ResponsesLifecycleEventProjection(ResponsesEventProjection):
    """Share the response snapshot field, not its concrete event identity."""

    response: Annotated[ResponsesLifecycleResponseProjection, guarded()]


class ResponsesCreatedEventProjection(ResponsesLifecycleEventProjection):
    """Creation is a nonterminal snapshot."""

    policy: ClassVar[ObjectPolicy] = ObjectPolicy(source="ResponseCreatedEvent")
    type: Annotated[Literal["response.created"], constrained()]


class ResponsesInProgressEventProjection(ResponsesLifecycleEventProjection):
    """Progress is a nonterminal snapshot."""

    policy: ClassVar[ObjectPolicy] = ObjectPolicy(source="ResponseInProgressEvent")
    type: Annotated[Literal["response.in_progress"], constrained()]


class ResponsesQueuedEventProjection(ResponsesLifecycleEventProjection):
    """Queue metadata does not make background execution an accepted request mode."""

    policy: ClassVar[ObjectPolicy] = ObjectPolicy(source="ResponseQueuedEvent")
    type: Annotated[Literal["response.queued"], constrained()]


class ResponsesCompletedEventProjection(ResponsesLifecycleEventProjection):
    """Successful native completion carries a final verified snapshot."""

    policy: ClassVar[ObjectPolicy] = ObjectPolicy(source="ResponseCompletedEvent")
    type: Annotated[Literal["response.completed"], constrained()]


class ResponsesFailedEventProjection(ResponsesLifecycleEventProjection):
    """Failure may still contain text and therefore remains a checked snapshot."""

    policy: ClassVar[ObjectPolicy] = ObjectPolicy(source="ResponseFailedEvent")
    type: Annotated[Literal["response.failed"], constrained()]


class ResponsesIncompleteEventProjection(ResponsesLifecycleEventProjection):
    """An incomplete response is terminal, but its partial text is not opaque."""

    policy: ClassVar[ObjectPolicy] = ObjectPolicy(source="ResponseIncompleteEvent")
    type: Annotated[Literal["response.incomplete"], constrained()]


class ResponsesErrorEventProjection(ResponsesEventProjection):
    """A provider error is terminal diagnostic data, not assistant output."""

    policy: ClassVar[ObjectPolicy] = ObjectPolicy(source="ResponseErrorEvent", opaque=("code", "param"))
    type: Annotated[Literal["error"], constrained()]
    message: Annotated[str, constrained()]
