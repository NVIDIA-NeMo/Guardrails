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

"""Declare one assistant text output alongside reviewed reasoning metadata."""

from typing import Annotated, Any, ClassVar, Literal

from nemoguardrails.server.experimental.provider.payload import GuardedContentModel, GuardedPayloadModel
from nemoguardrails.server.experimental.provider.projection_policy import (
    ArrayVariant,
    ObjectPolicy,
    PolicyModel,
    constrained,
    disabled,
    guarded,
)


class ResponsesOutputTextProjection(PolicyModel, GuardedContentModel):
    """Guard output text and reject uninspected citation and token text."""

    policy: ClassVar[ObjectPolicy] = ObjectPolicy(source="OutputTextContent", unknown_fields="configurable")
    type: Annotated[Literal["output_text"], constrained()]
    text: Annotated[str, guarded("assistant", replaceable=False, min_length=1)]
    annotations: Annotated[list[Any] | None, constrained(reason="core_capability.citation_content", max_length=0)] = (
        None
    )
    logprobs: Annotated[list[Any] | None, constrained(reason="provider_integrity.token_logprobs", max_length=0)] = None


class ResponsesReasoningItemProjection(PolicyModel, GuardedContentModel):
    """Preserve opaque reasoning metadata, not separate plaintext reasoning."""

    policy: ClassVar[ObjectPolicy] = ObjectPolicy(
        source="ReasoningItem",
        opaque=("encrypted_content", "id", "status"),
        unknown_fields="configurable",
        variant=ArrayVariant((("type", "reasoning"),), "any"),
    )
    type: Annotated[Literal["reasoning"], constrained()]
    content: Annotated[list[Any] | None, constrained(reason="core_capability.reasoning_content", max_length=0)] = None
    summary: Annotated[list[Any] | None, constrained(reason="core_capability.reasoning_content", max_length=0)] = None


class ResponsesAssistantMessageProjection(PolicyModel, GuardedContentModel):
    """Select exactly one assistant message without assuming its output index."""

    policy: ClassVar[ObjectPolicy] = ObjectPolicy(
        source="OutputMessage",
        opaque=("id", "phase", "status"),
        unknown_fields="configurable",
        variant=ArrayVariant((("type", "message"), ("role", "assistant")), "exactly_one"),
    )
    type: Annotated[Literal["message"], constrained()]
    role: Annotated[Literal["assistant"], constrained()]
    content: Annotated[list[ResponsesOutputTextProjection], guarded(min_length=1, max_length=1)]


RESPONSE_METADATA_FIELDS = (
    "access_programs",
    "background",
    "billing",
    "completed_at",
    "conversation",
    "created_at",
    "error",
    "frequency_penalty",
    "id",
    "incomplete_details",
    "instructions",
    "max_output_tokens",
    "max_tool_calls",
    "metadata",
    "model",
    "moderation",
    "object",
    "parallel_tool_calls",
    "presence_penalty",
    "previous_response_id",
    "prompt",
    "prompt_cache_diagnostics",
    "prompt_cache_key",
    "prompt_cache_options",
    "prompt_cache_retention",
    "reasoning",
    "safety_identifier",
    "service_tier",
    "status",
    "store",
    "temperature",
    "text",
    "tool_choice",
    "tool_usage",
    "tools",
    "top_logprobs",
    "top_p",
    "truncation",
    "usage",
    "user",
)


class ResponsesGuardedResponseProjection(PolicyModel, GuardedPayloadModel):
    """Keep provider output ordering and select one reviewed text-bearing item."""

    policy: ClassVar[ObjectPolicy] = ObjectPolicy(source="Response", opaque=RESPONSE_METADATA_FIELDS)
    output: Annotated[list[ResponsesReasoningItemProjection | ResponsesAssistantMessageProjection], guarded()]
    output_text: Annotated[None, disabled("provider_integrity.duplicate_text")] = None
