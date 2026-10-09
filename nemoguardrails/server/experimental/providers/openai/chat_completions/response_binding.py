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

"""Bind the handwritten Chat Completions response policy to the runtime."""

from nemoguardrails.server.experimental.provider.projection_policy import payload_contract, text_location
from nemoguardrails.server.experimental.providers.openai.chat_completions.response_projection import (
    ChatCompletionsGuardedResponseProjection,
)

CAPABILITY_PROFILE = "single_text.v1"
RESPONSE_SOURCE_SCHEMA = "CreateChatCompletionResponse"
PAYLOAD_CONTRACT = payload_contract(
    ChatCompletionsGuardedResponseProjection,
    projection_id="openai.chat_completions.response.text.v1",
    direction="response",
)
RESPONSE_GUARDED_FIELDS = PAYLOAD_CONTRACT.root.guarded_fields
RESPONSE_CONSTRAINED_FIELDS = PAYLOAD_CONTRACT.root.constrained_fields
RESPONSE_OPAQUE_FIELDS = PAYLOAD_CONTRACT.root.opaque_fields
RESPONSE_CONTENT_SCHEMAS = tuple(
    (entry.source_schema, entry.coverage.reviewed_fields - entry.coverage.local_extension_fields)
    for entry in PAYLOAD_CONTRACT.content_models
    if entry.source_schema is not None
)
GUARDED_TEXT_LOCATION = text_location(ChatCompletionsGuardedResponseProjection)


class ChatCompletionsGuardedResponse(ChatCompletionsGuardedResponseProjection):
    """Bind the handwritten projection to its guarded runtime semantics."""

    guarded_text_location = GUARDED_TEXT_LOCATION
    projection_contract = PAYLOAD_CONTRACT
