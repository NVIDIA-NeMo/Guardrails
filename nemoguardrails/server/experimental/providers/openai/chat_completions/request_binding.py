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

"""Bind the handwritten Chat Completions request policy to the runtime."""

from nemoguardrails.server.experimental.provider.authoring import payload_contract, text_location
from nemoguardrails.server.experimental.providers.openai.chat_completions.request_projection import (
    ChatCompletionsGuardedRequestProjection,
)

PROVIDER_DOCUMENT_URL = (
    "https://github.com/openai/openai-openapi/blob/df63773f69f542ef875b9f00c3837c25ba5f4f2a/openapi.yaml"
)
PROVIDER_REVISION = "df63773f69f542ef875b9f00c3837c25ba5f4f2a"
PROVIDER_DOCUMENT_VERSION = "2.3.0"
PROVIDER_DOCUMENT_SHA256 = "f2dae1a9aced09b91310db89edda51bf1e36ecbfb05230c3a50b239c07708469"
CAPABILITY_PROFILE = "single_text.v1"
REQUEST_SOURCE_SCHEMA = "CreateChatCompletionRequest"
PAYLOAD_CONTRACT = payload_contract(
    ChatCompletionsGuardedRequestProjection,
    projection_id="openai.chat_completions.request.text.v1",
    direction="request",
)
REQUEST_GUARDED_FIELDS = PAYLOAD_CONTRACT.root.guarded_fields
REQUEST_CONSTRAINED_FIELDS = PAYLOAD_CONTRACT.root.constrained_fields
REQUEST_OPAQUE_FIELDS = PAYLOAD_CONTRACT.root.opaque_fields
REQUEST_CONTENT_SCHEMAS = tuple(
    (entry.source_schema, entry.coverage.reviewed_fields - entry.coverage.local_extension_fields)
    for entry in PAYLOAD_CONTRACT.content_models
    if entry.source_schema is not None
)
GUARDED_TEXT_LOCATION = text_location(ChatCompletionsGuardedRequestProjection)
STREAM_SELECTOR_FIELD = "stream"


class ChatCompletionsGuardedRequest(ChatCompletionsGuardedRequestProjection):
    """Bind the handwritten projection to its guarded runtime semantics."""

    guarded_text_location = GUARDED_TEXT_LOCATION
    projection_contract = PAYLOAD_CONTRACT
    stream_selector_field = STREAM_SELECTOR_FIELD
