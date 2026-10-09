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

"""Declare buffered Chat response policy and text replacement restrictions."""

from typing import Annotated, Any, ClassVar, Literal

from nemoguardrails.server.experimental.provider.payload import GuardedContentModel, GuardedPayloadModel
from nemoguardrails.server.experimental.provider.projection_policy import (
    ObjectPolicy,
    PolicyModel,
    constrained,
    disabled,
    guarded,
    opaque,
)


class ChatCompletionsAssistantMessageProjection(PolicyModel, GuardedContentModel):
    """Accept assistant text while preventing replacement of annotated content."""

    policy: ClassVar[ObjectPolicy] = ObjectPolicy(source="ChatCompletionResponseMessage", unknown_fields="configurable")
    # OpenAI allows empty or null content, for example when generation stops
    # at a length limit or a content filter. Every other reviewed field is
    # content-free here, so such a response has nothing to inspect.
    content: Annotated[
        str | None,
        guarded(
            "assistant",
            replaceable=True,
            blocked_by="annotations",
            replacement_reason="provider_integrity.annotated_text",
        ),
    ]
    role: Annotated[Literal["assistant"], constrained()]
    # Citation annotations carry provider text, such as titles, that output
    # rails do not inspect. Web search is disabled on the request, so an
    # OpenAI response without citations has null or empty annotations.
    annotations: Annotated[list[Any] | None, constrained(reason="core_capability.citation_content", max_length=0)] = (
        None
    )
    audio: Annotated[None, disabled("core_capability.audio_content")] = None
    function_call: Annotated[None, disabled("core_capability.tool_content")] = None
    reasoning_content: Annotated[None, disabled("core_capability.reasoning_content", extension=True)] = None
    refusal: Annotated[None, disabled("core_capability.refusal_content")] = None
    # OpenAI's schema allows an empty list here, and it carries no tool content.
    tool_calls: Annotated[list[Any] | None, constrained(reason="core_capability.tool_content", max_length=0)] = None


class ChatCompletionsChoiceProjection(PolicyModel, GuardedContentModel):
    """Expose the guarded message while preserving opaque choice metadata."""

    policy: ClassVar[ObjectPolicy] = ObjectPolicy(unknown_fields="configurable")
    message: Annotated[ChatCompletionsAssistantMessageProjection, guarded()]
    finish_reason: Annotated[Any, opaque()] = None
    index: Annotated[Any, opaque()] = None
    logprobs: Annotated[None, disabled("provider_integrity.token_logprobs")] = None


class ChatCompletionsGuardedResponseProjection(PolicyModel, GuardedPayloadModel):
    """Require one guarded choice and retain reviewed provider-owned response fields.

    The response is closed to OpenAI's fields: a member outside them could
    carry generated text that output rails never inspect.
    """

    policy: ClassVar[ObjectPolicy] = ObjectPolicy(
        opaque=(
            "service_tier",
            "created",
            "object",
            "system_fingerprint",
            "usage",
            "id",
            "model",
            "metadata",
            "moderation",
        )
    )
    choices: Annotated[list[ChatCompletionsChoiceProjection], guarded(min_length=1, max_length=1)]
