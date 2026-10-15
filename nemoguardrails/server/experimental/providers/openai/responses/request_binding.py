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

"""Derive request coverage and text extraction from Responses field policies."""

from nemoguardrails.server.experimental.provider.projection_policy import payload_contract, text_location
from nemoguardrails.server.experimental.providers.openai.responses.request_projection import (
    ResponsesGuardedRequestProjection,
)


class ResponsesGuardedRequest(ResponsesGuardedRequestProjection):
    """Bind the declared input subject and strict response-mode selector."""

    guarded_text_location = text_location(ResponsesGuardedRequestProjection)
    projection_contract = payload_contract(
        ResponsesGuardedRequestProjection, projection_id="openai.responses.request.text.v1", direction="request"
    )
    stream_selector_field = "stream"
