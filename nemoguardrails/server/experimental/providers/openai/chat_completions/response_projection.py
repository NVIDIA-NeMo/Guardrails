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

from typing import Annotated, Any, ClassVar, Literal

from nemoguardrails.server.experimental.provider.authoring import (
    Policy,
    PolicyModel,
    constrained,
    disabled,
    guarded,
    opaque,
)
from nemoguardrails.server.experimental.provider.payload import GuardedContentModel, GuardedPayloadModel


class ChatCompletionsAssistantMessageProjection(PolicyModel, GuardedContentModel):
    policy: ClassVar[Policy] = Policy(source="ChatCompletionResponseMessage", unknown_fields="configurable")
    content: Annotated[
        str,
        guarded(
            "assistant",
            replaceable=True,
            blocked_by="annotations",
            replacement_reason="provider_integrity.annotated_text",
            min_length=1,
        ),
    ]
    role: Annotated[Literal["assistant"], constrained()]
    annotations: Annotated[list[Any] | None, constrained()] = None
    audio: Annotated[None, disabled("core_capability.audio_content")] = None
    function_call: Annotated[None, disabled("core_capability.tool_content")] = None
    reasoning_content: Annotated[None, disabled("core_capability.reasoning_content", extension=True)] = None
    refusal: Annotated[None, disabled("core_capability.refusal_content")] = None
    tool_calls: Annotated[None, disabled("core_capability.tool_content")] = None


class ChatCompletionsChoiceProjection(PolicyModel, GuardedContentModel):
    policy: ClassVar[Policy] = Policy(unknown_fields="configurable")
    message: Annotated[ChatCompletionsAssistantMessageProjection, guarded()]
    finish_reason: Annotated[Any, opaque()] = None
    index: Annotated[Any, opaque()] = None
    logprobs: Annotated[None, disabled("provider_integrity.token_logprobs")] = None


class ChatCompletionsGuardedResponseProjection(PolicyModel, GuardedPayloadModel):
    policy: ClassVar[Policy] = Policy(
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
