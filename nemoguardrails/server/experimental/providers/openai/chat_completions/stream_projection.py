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

"""Declare guarded text, closed content boundaries, and opaque stream metadata.

These handwritten models own per-event field policy. The classifier binds them
to event roles; handwritten hooks enforce ordering and terminal-state behavior.
"""

from __future__ import annotations

from typing import Annotated, Any, ClassVar, Literal

from pydantic import BeforeValidator

from nemoguardrails.server.experimental.provider.payload import GuardedContentModel, GuardedProjectionModel
from nemoguardrails.server.experimental.provider.projection_policy import (
    ObjectPolicy,
    PolicyModel,
    constrained,
    disabled,
    guarded,
    opaque,
)


def _require_int(value: object) -> object:
    """Reject booleans, strings, and floats before Literal[0] can coerce them."""
    if type(value) is not int:
        raise ValueError("index must be the integer 0")
    return value


class ChatCompletionsStreamDeltaProjection(PolicyModel, GuardedContentModel):
    """Accept assistant text or metadata-only deltas within a closed boundary."""

    policy: ClassVar[ObjectPolicy] = ObjectPolicy(source="ChatCompletionStreamResponseDelta")

    audio: Annotated[None, disabled("core_capability.audio_content", extension=True)] = None
    content: Annotated[str | None, guarded("assistant")] = None
    function_call: Annotated[None, disabled("core_capability.tool_content")] = None
    reasoning_content: Annotated[None, disabled("core_capability.reasoning_content", extension=True)] = None
    refusal: Annotated[None, disabled("core_capability.refusal_content")] = None
    role: Annotated[Literal["assistant"] | None, constrained()] = None
    tool_calls: Annotated[None, disabled("core_capability.tool_content")] = None


class ChatCompletionsStreamErrorDetailsProjection(PolicyModel, GuardedContentModel):
    """Validate the native error message without treating it as guarded text."""

    policy: ClassVar[ObjectPolicy] = ObjectPolicy(source="Error", unknown_fields="configurable")

    code: Annotated[Any, opaque()] = None
    message: Annotated[str, constrained()]
    misalignment: Annotated[Any, opaque()] = None
    param: Annotated[Any, opaque()] = None
    type: Annotated[Any, opaque()] = None


class ChatCompletionsStreamErrorProjection(PolicyModel, GuardedProjectionModel):
    """Accept only an error envelope, never an error mixed with generated content."""

    policy: ClassVar[ObjectPolicy] = ObjectPolicy(source="ErrorResponse")

    error: Annotated[ChatCompletionsStreamErrorDetailsProjection, constrained()]


class ChatCompletionsStreamChoiceProjection(PolicyModel, GuardedContentModel):
    """Require integer choice zero and reject unreviewed choice-level content."""

    delta: Annotated[ChatCompletionsStreamDeltaProjection, guarded()]
    finish_reason: Annotated[Any, opaque()] = None
    index: Annotated[Literal[0], BeforeValidator(_require_int), constrained()]
    # Log probabilities carry token text that output rails do not inspect, as in
    # buffered responses; requests for them are already rejected.
    logprobs: Annotated[None, disabled("provider_integrity.token_logprobs")] = None


class ChatCompletionsStreamPayloadProjection(PolicyModel, GuardedProjectionModel):
    """Allow one text choice or an empty usage chunk, preserving root metadata."""

    policy: ClassVar[ObjectPolicy] = ObjectPolicy(
        source="CreateChatCompletionStreamResponse",
        opaque=("created", "id", "model", "moderation", "obfuscation", "service_tier", "system_fingerprint", "usage"),
    )

    choices: Annotated[list[ChatCompletionsStreamChoiceProjection], guarded(max_length=1)]
    object: Annotated[Literal["chat.completion.chunk"], constrained()]
