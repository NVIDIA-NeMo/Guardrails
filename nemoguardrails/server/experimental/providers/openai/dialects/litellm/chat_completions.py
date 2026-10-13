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

"""Bind LiteLLM Chat extensions to the single-text OpenAI pipeline.

Reviewed finish metadata is opaque; extension containers remain closed to
unreviewed content. See README.md for source/version evidence and field policy.

The OpenAI request binding and stream lifecycle hooks are reused. Python declarations
are authoritative; the exported contract describes these effective bindings and
is never loaded to construct them. No LiteLLM dependency is needed at runtime.
"""

from dataclasses import replace
from typing import Annotated, Any, ClassVar

from nemoguardrails.server.experimental.provider.projection_policy import (
    ObjectPolicy,
    PolicyModel,
    constrained,
    disabled,
    guarded,
    payload_contract,
    text_location,
)
from nemoguardrails.server.experimental.provider.stream import StreamBinding
from nemoguardrails.server.experimental.providers.openai.chat_completions.endpoint import CHAT_COMPLETIONS_ENDPOINT
from nemoguardrails.server.experimental.providers.openai.chat_completions.response_projection import (
    ChatCompletionsAssistantMessageProjection,
    ChatCompletionsChoiceProjection,
    ChatCompletionsGuardedResponseProjection,
)
from nemoguardrails.server.experimental.providers.openai.chat_completions.stream_hooks import ChatCompletionsStreamHooks
from nemoguardrails.server.experimental.providers.openai.dialects.litellm.stream import STREAM_CLASSIFIER


class LiteLLMMessageFields(PolicyModel):
    """Allow the recorded null refusal without accepting uninspected content."""

    refusal: Annotated[None, disabled("core_capability.refusal_content")] = None
    thinking_blocks: Annotated[
        list[Any] | None, constrained(reason="core_capability.reasoning_content", max_length=0)
    ] = None


class LiteLLMChoiceFields(PolicyModel):
    """Preserve LiteLLM's original finish reason without inspecting metadata."""

    policy: ClassVar[ObjectPolicy] = ObjectPolicy(opaque=("native_finish_reason",))


class LiteLLMAssistantMessageProjection(ChatCompletionsAssistantMessageProjection):
    """Add a closed message extension without changing the guarded text target."""

    policy: ClassVar[ObjectPolicy] = replace(
        ChatCompletionsAssistantMessageProjection.policy, source=None, unknown_fields="forbid"
    )
    provider_specific_fields: Annotated[LiteLLMMessageFields | None, constrained()] = None
    images: Annotated[list[Any] | None, constrained(reason="core_capability.multimodal_content", max_length=0)] = None
    thinking_blocks: Annotated[
        list[Any] | None, constrained(reason="core_capability.reasoning_content", max_length=0)
    ] = None
    reasoning_items: Annotated[
        list[Any] | None, constrained(reason="core_capability.reasoning_content", max_length=0)
    ] = None


class LiteLLMChoiceProjection(ChatCompletionsChoiceProjection):
    """Rebind the message type and constrain the provider-specific choice object."""

    policy: ClassVar[ObjectPolicy] = replace(ChatCompletionsChoiceProjection.policy, unknown_fields="forbid")
    message: Annotated[LiteLLMAssistantMessageProjection, guarded()]
    provider_specific_fields: Annotated[LiteLLMChoiceFields | None, constrained()] = None


class LiteLLMResponseProjection(ChatCompletionsGuardedResponseProjection):
    """Retain the single-text response boundary with the reviewed nested types."""

    choices: Annotated[list[LiteLLMChoiceProjection], guarded(min_length=1, max_length=1)]


class LiteLLMGuardedResponse(LiteLLMResponseProjection):
    """Derive coverage and text extraction from the complete dialect model graph."""

    guarded_text_location = text_location(LiteLLMResponseProjection)
    projection_contract = payload_contract(
        LiteLLMResponseProjection,
        projection_id="openai.litellm.chat_completions.response.text.v1",
        direction="response",
    )


LITELLM_CHAT_COMPLETIONS_ENDPOINT = replace(
    CHAT_COMPLETIONS_ENDPOINT,
    contract_name="litellm_chat_completions",
    operation="LiteLLM Chat Completions",
    guarded_response_model=LiteLLMGuardedResponse,
    stream=StreamBinding(STREAM_CLASSIFIER, ChatCompletionsStreamHooks),
)
