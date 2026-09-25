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

"""Generated binding for the openai.chat_completions.response.text.v1 Payload Projection."""

from nemoguardrails.server.experimental.provider.payload import (
    GuardedTextLocation,
    PayloadCapabilityProfile,
    PayloadProjectionContract,
    ProjectionFieldCoverage,
    ProjectionModelContract,
)
from nemoguardrails.server.experimental.providers.openai.chat_completions.response_projection import (
    ChatCompletionsAssistantMessageProjection,
    ChatCompletionsChoiceProjection,
    ChatCompletionsGuardedResponseProjection,
)

PROVIDER_DOCUMENT_URL = (
    "https://github.com/openai/openai-openapi/blob/df63773f69f542ef875b9f00c3837c25ba5f4f2a/openapi.yaml"
)
PROVIDER_REVISION = "df63773f69f542ef875b9f00c3837c25ba5f4f2a"
PROVIDER_DOCUMENT_VERSION = "2.3.0"
PROVIDER_DOCUMENT_SHA256 = "f2dae1a9aced09b91310db89edda51bf1e36ecbfb05230c3a50b239c07708469"
CAPABILITY_PROFILE = "single_text.v1"
RESPONSE_SOURCE_SCHEMA = "CreateChatCompletionResponse"
RESPONSE_GUARDED_FIELDS = frozenset(["choices"])
RESPONSE_CONSTRAINED_FIELDS = frozenset([])
RESPONSE_OPAQUE_FIELDS = frozenset(
    ["created", "id", "metadata", "model", "moderation", "object", "service_tier", "system_fingerprint", "usage"]
)
RESPONSE_CONTENT_SCHEMAS = (
    (
        "ChatCompletionResponseMessage",
        frozenset(["annotations", "audio", "content", "function_call", "refusal", "role", "tool_calls"]),
    ),
)
GUARDED_TEXT_LOCATION = GuardedTextLocation(
    role="assistant",
    object_path=("choices", 0, "message"),
    member="content",
    allows_replacement=True,
    replacement_blocked_by="annotations",
)
PAYLOAD_CONTRACT = PayloadProjectionContract(
    projection_id="openai.chat_completions.response.text.v1",
    direction="response",
    profile=PayloadCapabilityProfile("single_text.v1"),
    root=ProjectionFieldCoverage(
        guarded_fields=frozenset(["choices"]),
        constrained_fields=frozenset([]),
        opaque_fields=frozenset(
            [
                "created",
                "id",
                "metadata",
                "model",
                "moderation",
                "object",
                "service_tier",
                "system_fingerprint",
                "usage",
            ]
        ),
        local_extension_fields=frozenset([]),
    ),
    content_models=(
        ProjectionModelContract(
            model=ChatCompletionsChoiceProjection,
            source_schema=None,
            coverage=ProjectionFieldCoverage(
                guarded_fields=frozenset(["message"]),
                constrained_fields=frozenset(["logprobs"]),
                opaque_fields=frozenset(["finish_reason", "index"]),
                local_extension_fields=frozenset([]),
            ),
        ),
        ProjectionModelContract(
            model=ChatCompletionsAssistantMessageProjection,
            source_schema="ChatCompletionResponseMessage",
            coverage=ProjectionFieldCoverage(
                guarded_fields=frozenset(["content"]),
                constrained_fields=frozenset(
                    ["annotations", "audio", "function_call", "refusal", "role", "tool_calls"]
                ),
                opaque_fields=frozenset([]),
                local_extension_fields=frozenset(["reasoning_content"]),
            ),
        ),
    ),
)


class ChatCompletionsGuardedResponse(ChatCompletionsGuardedResponseProjection):
    """Bind the generated projection to its guarded runtime semantics."""

    guarded_text_location = GUARDED_TEXT_LOCATION
    projection_contract = PAYLOAD_CONTRACT
