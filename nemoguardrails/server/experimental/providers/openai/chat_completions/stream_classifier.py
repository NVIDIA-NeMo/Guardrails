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

"""Bind typed Chat stream projections to explicit event and transport rules.

The declarations own event selection and missing-text behavior. Field coverage,
shape inventories, and the text path are derived; stateful lifecycle checks stay
in stream_hooks. Importing this module does not enable an endpoint.
"""

from nemoguardrails.server.experimental.provider.projection_policy import field_coverage
from nemoguardrails.server.experimental.provider.stream import (
    StreamCapabilityProfile,
    StreamEventRole,
    StreamProjectionContract,
)
from nemoguardrails.server.experimental.provider.stream_classifier import (
    StreamClassifierDefinition,
    StreamEventRule,
)
from nemoguardrails.server.experimental.provider.stream_policy import (
    build_policy_stream_classifier,
    stream_shape_coverage,
    stream_text_path,
)
from nemoguardrails.server.experimental.providers.openai.chat_completions.stream_projection import (
    ChatCompletionsStreamErrorProjection,
    ChatCompletionsStreamPayloadProjection,
)

PROJECTION_ID = "openai.chat_completions.stream.text.v1"
STREAM_SENTINELS = ((b"[DONE]", "[DONE]"),)
STREAM_NON_DATA_SHAPE = "keepalive"
STREAM_RULES = (
    StreamEventRule(
        shape="chat.completion.chunk:content",
        model=ChatCompletionsStreamPayloadProjection,
        role=StreamEventRole.GUARDED_TEXT,
        match=(("object", "chat.completion.chunk"),),
        text_path=stream_text_path(ChatCompletionsStreamPayloadProjection),
        missing_text_role=StreamEventRole.OPAQUE_METADATA,
        missing_text_shape="chat.completion.chunk:metadata",
    ),
    StreamEventRule(
        shape="error",
        model=ChatCompletionsStreamErrorProjection,
        role=StreamEventRole.PROVIDER_ERROR,
        required_fields=frozenset({"error"}),
    ),
)

STREAM_CONTRACT = StreamProjectionContract(
    projection_id=PROJECTION_ID,
    profile=StreamCapabilityProfile.SINGLE_TEXT_DELTA_V1,
    shapes=stream_shape_coverage(
        STREAM_RULES,
        sentinels=STREAM_SENTINELS,
        non_data_shape=STREAM_NON_DATA_SHAPE,
    ),
    fields=field_coverage(ChatCompletionsStreamPayloadProjection),
)

STREAM_CLASSIFIER = build_policy_stream_classifier(
    StreamClassifierDefinition(
        subject=PROJECTION_ID,
        contract=STREAM_CONTRACT,
        non_data_shape=STREAM_NON_DATA_SHAPE,
        sentinels=STREAM_SENTINELS,
        rules=STREAM_RULES,
    )
)
