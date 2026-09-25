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

"""Generated binding for the openai.chat_completions.request.text.v1 Payload Projection."""

from nemoguardrails.server.experimental.provider.payload import (
    GuardedTextLocation,
    PayloadCapabilityProfile,
    PayloadProjectionContract,
    ProjectionFieldCoverage,
    ProjectionModelContract,
)
from nemoguardrails.server.experimental.providers.openai.chat_completions.request_projection import (
    ChatCompletionsGuardedRequestProjection,
    ChatCompletionsUserMessageProjection,
)

PROVIDER_DOCUMENT_URL = (
    "https://github.com/openai/openai-openapi/blob/df63773f69f542ef875b9f00c3837c25ba5f4f2a/openapi.yaml"
)
PROVIDER_REVISION = "df63773f69f542ef875b9f00c3837c25ba5f4f2a"
PROVIDER_DOCUMENT_VERSION = "2.3.0"
PROVIDER_DOCUMENT_SHA256 = "f2dae1a9aced09b91310db89edda51bf1e36ecbfb05230c3a50b239c07708469"
CAPABILITY_PROFILE = "single_text.v1"
REQUEST_SOURCE_SCHEMA = "CreateChatCompletionRequest"
REQUEST_GUARDED_FIELDS = frozenset(["messages"])
REQUEST_CONSTRAINED_FIELDS = frozenset(
    [
        "audio",
        "function_call",
        "functions",
        "modalities",
        "n",
        "parallel_tool_calls",
        "prediction",
        "response_format",
        "stream",
        "tool_choice",
        "tools",
        "web_search_options",
    ]
)
REQUEST_OPAQUE_FIELDS = frozenset(
    [
        "frequency_penalty",
        "logit_bias",
        "logprobs",
        "max_completion_tokens",
        "max_tokens",
        "metadata",
        "model",
        "moderation",
        "presence_penalty",
        "prompt_cache_key",
        "prompt_cache_options",
        "prompt_cache_retention",
        "reasoning_effort",
        "safety_identifier",
        "seed",
        "service_tier",
        "stop",
        "store",
        "stream_options",
        "temperature",
        "top_logprobs",
        "top_p",
        "user",
        "verbosity",
    ]
)
REQUEST_CONTENT_SCHEMAS = (("ChatCompletionRequestUserMessage", frozenset(["content", "name", "role"])),)
STREAM_SELECTOR_FIELD = "stream"
GUARDED_TEXT_LOCATION = GuardedTextLocation(
    role="user",
    object_path=("messages", 0),
    member="content",
    allows_replacement=True,
    replacement_blocked_by=None,
)
PAYLOAD_CONTRACT = PayloadProjectionContract(
    projection_id="openai.chat_completions.request.text.v1",
    direction="request",
    profile=PayloadCapabilityProfile("single_text.v1"),
    root=ProjectionFieldCoverage(
        guarded_fields=frozenset(["messages"]),
        constrained_fields=frozenset(
            [
                "audio",
                "function_call",
                "functions",
                "modalities",
                "n",
                "parallel_tool_calls",
                "prediction",
                "response_format",
                "stream",
                "tool_choice",
                "tools",
                "web_search_options",
            ]
        ),
        opaque_fields=frozenset(
            [
                "frequency_penalty",
                "logit_bias",
                "logprobs",
                "max_completion_tokens",
                "max_tokens",
                "metadata",
                "model",
                "moderation",
                "presence_penalty",
                "prompt_cache_key",
                "prompt_cache_options",
                "prompt_cache_retention",
                "reasoning_effort",
                "safety_identifier",
                "seed",
                "service_tier",
                "stop",
                "store",
                "stream_options",
                "temperature",
                "top_logprobs",
                "top_p",
                "user",
                "verbosity",
            ]
        ),
        local_extension_fields=frozenset([]),
    ),
    content_models=(
        ProjectionModelContract(
            model=ChatCompletionsUserMessageProjection,
            source_schema="ChatCompletionRequestUserMessage",
            coverage=ProjectionFieldCoverage(
                guarded_fields=frozenset(["content"]),
                constrained_fields=frozenset(["role"]),
                opaque_fields=frozenset(["name"]),
                local_extension_fields=frozenset([]),
            ),
        ),
    ),
)


class ChatCompletionsGuardedRequest(ChatCompletionsGuardedRequestProjection):
    """Bind the generated projection to its guarded runtime semantics."""

    guarded_text_location = GUARDED_TEXT_LOCATION
    projection_contract = PAYLOAD_CONTRACT
    stream_selector_field = STREAM_SELECTOR_FIELD
