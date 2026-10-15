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

"""Derive buffered Responses coverage and heterogeneous text selection."""

from nemoguardrails.server.experimental.provider.projection_policy import payload_contract, text_location
from nemoguardrails.server.experimental.providers.openai.responses.response_projection import (
    ResponsesGuardedResponseProjection,
)


class ResponsesGuardedResponse(ResponsesGuardedResponseProjection):
    """Bind one assistant text target; replacement is deliberately unsupported."""

    guarded_text_location = text_location(ResponsesGuardedResponseProjection)
    projection_contract = payload_contract(
        ResponsesGuardedResponseProjection, projection_id="openai.responses.response.text.v1", direction="response"
    )
