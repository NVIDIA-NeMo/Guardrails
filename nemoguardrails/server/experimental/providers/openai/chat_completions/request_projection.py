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

"""Declare Chat request field policy without duplicating runtime binding metadata."""

from typing import Annotated, Any, ClassVar, Literal

from pydantic import BeforeValidator, StrictBool

from nemoguardrails.server.experimental.provider.payload import GuardedContentModel, GuardedRequestModel
from nemoguardrails.server.experimental.provider.projection_policy import (
    ObjectPolicy,
    PolicyModel,
    constrained,
    disabled,
    guarded,
    opaque,
)


def _require_int(value: object) -> object:
    """Reject booleans, strings, and floats before Literal[1] can coerce them."""
    if type(value) is not int:
        raise ValueError("n must be the integer 1")
    return value


class ChatCompletionsUserMessageProjection(PolicyModel, GuardedContentModel):
    """Accept one user text message with reviewed optional provider metadata."""

    policy: ClassVar[ObjectPolicy] = ObjectPolicy(
        source="ChatCompletionRequestUserMessage", unknown_fields="configurable"
    )
    content: Annotated[str, guarded("user", replaceable=True, min_length=1)]
    name: Annotated[Any, opaque()] = None
    role: Annotated[Literal["user"], constrained()]


class ChatCompletionsGuardedRequestProjection(PolicyModel, GuardedRequestModel):
    """Declare the single-message request shape and explicit feature restrictions.

    The request binding attaches extraction, coverage, and response-mode metadata.
    Recognizing the stream flag does not imply that an endpoint supports streaming.
    """

    policy: ClassVar[ObjectPolicy] = ObjectPolicy(
        opaque=(
            "safety_identifier",
            "logit_bias",
            "presence_penalty",
            "reasoning_effort",
            "store",
            "stop",
            "top_p",
            "verbosity",
            "seed",
            "stream_options",
            "moderation",
            "service_tier",
            "user",
            "prompt_cache_options",
            "temperature",
            "max_tokens",
            "max_completion_tokens",
            "frequency_penalty",
            "top_logprobs",
            "prompt_cache_key",
            "model",
            "metadata",
            "logprobs",
            "prompt_cache_retention",
        )
    )
    messages: Annotated[list[ChatCompletionsUserMessageProjection], guarded(min_length=1, max_length=1)]
    n: Annotated[
        Literal[1], BeforeValidator(_require_int), constrained(reason="core_capability.single_text_target")
    ] = 1
    stream: Annotated[StrictBool, constrained()] = False
    audio: Annotated[None, disabled("core_capability.audio_content")] = None
    function_call: Annotated[None, disabled("core_capability.tool_content")] = None
    functions: Annotated[None, disabled("core_capability.tool_content")] = None
    modalities: Annotated[None, disabled("core_capability.multimodal_content")] = None
    parallel_tool_calls: Annotated[None, disabled("core_capability.tool_content")] = None
    prediction: Annotated[None, disabled("core_capability.predicted_content")] = None
    response_format: Annotated[None, disabled("projection_policy.plain_text_output")] = None
    tool_choice: Annotated[None, disabled("core_capability.tool_content")] = None
    tools: Annotated[None, disabled("core_capability.tool_content")] = None
    web_search_options: Annotated[None, disabled("core_capability.tool_content")] = None
