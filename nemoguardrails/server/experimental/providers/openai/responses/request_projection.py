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

"""Declare the single-text, foreground Responses request boundary."""

from typing import Annotated, ClassVar, Literal

from pydantic import StrictBool

from nemoguardrails.server.experimental.provider.payload import GuardedContentModel, GuardedRequestModel, StrictFalse
from nemoguardrails.server.experimental.provider.projection_policy import (
    ObjectPolicy,
    PolicyModel,
    constrained,
    disabled,
    guarded,
)


class ResponsesReasoningConfigProjection(PolicyModel, GuardedContentModel):
    """Allow reviewed reasoning controls, but no separate reasoning text input."""

    policy: ClassVar[ObjectPolicy] = ObjectPolicy(source="Reasoning", opaque=("effort",), unknown_fields="configurable")
    context: Annotated[None, disabled("core_capability.reasoning_content")] = None
    generate_summary: Annotated[None, disabled("core_capability.reasoning_content")] = None
    mode: Annotated[None, disabled("core_capability.reasoning_content")] = None
    summary: Annotated[None, disabled("core_capability.reasoning_content")] = None


class ResponsesTextFormatProjection(PolicyModel, GuardedContentModel):
    """Disallow structured-output instructions outside the guarded input."""

    policy: ClassVar[ObjectPolicy] = ObjectPolicy(source="ResponseFormatText", unknown_fields="configurable")
    type: Annotated[Literal["text"], constrained(reason="projection_policy.plain_text_output")]


class ResponsesTextConfigProjection(PolicyModel, GuardedContentModel):
    """Retain verbosity while restricting any supplied format to plain text."""

    policy: ClassVar[ObjectPolicy] = ObjectPolicy(
        source="ResponseTextParam", opaque=("verbosity",), unknown_fields="configurable"
    )
    format: Annotated[ResponsesTextFormatProjection | None, constrained()] = None


class ResponsesGuardedRequestProjection(PolicyModel, GuardedRequestModel):
    """Inspect one string input without provider-managed history or tools.

    Provider controls are reviewed passthrough values, not inspected subjects.
    Unknown root fields fail closed; no backend dialect is inferred here.
    """

    policy: ClassVar[ObjectPolicy] = ObjectPolicy(
        source="CreateResponse",
        opaque=(
            "max_output_tokens",
            "max_tool_calls",
            "metadata",
            "model",
            "moderation",
            "parallel_tool_calls",
            "prompt_cache_key",
            "prompt_cache_options",
            "prompt_cache_retention",
            "safety_identifier",
            "service_tier",
            "stream_options",
            "temperature",
            "top_p",
            "truncation",
            "user",
        ),
    )
    input: Annotated[str, guarded("user", replaceable=False, min_length=1)]
    store: Annotated[StrictFalse, constrained(reason="core_capability.response_storage")]
    stream: Annotated[StrictBool, constrained()] = False
    background: Annotated[StrictFalse, constrained(reason="core_capability.background_execution")] = False
    context_management: Annotated[None, disabled("core_capability.conversation_state")] = None
    conversation: Annotated[None, disabled("core_capability.conversation_state")] = None
    include: Annotated[None, disabled("core_capability.additional_content")] = None
    instructions: Annotated[None, disabled("core_capability.additional_input")] = None
    previous_response_id: Annotated[None, disabled("core_capability.conversation_state")] = None
    prompt: Annotated[None, disabled("core_capability.additional_input")] = None
    reasoning: Annotated[ResponsesReasoningConfigProjection | None, constrained()] = None
    text: Annotated[ResponsesTextConfigProjection | None, constrained()] = None
    tool_choice: Annotated[None, disabled("core_capability.tool_content")] = None
    tools: Annotated[None, disabled("core_capability.tool_content")] = None
    top_logprobs: Annotated[
        int | None, constrained(reason="provider_integrity.token_logprobs", strict=True, ge=0, le=0)
    ] = None
