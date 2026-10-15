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

"""Compose the guarded Responses operation from its typed bindings."""

from nemoguardrails.server.experimental._http_paths import GuardedOperationPath
from nemoguardrails.server.experimental.provider.endpoint import GuardedJsonEndpoint
from nemoguardrails.server.experimental.provider.stream import StreamBinding
from nemoguardrails.server.experimental.providers.openai.responses.request_binding import ResponsesGuardedRequest
from nemoguardrails.server.experimental.providers.openai.responses.response_binding import ResponsesGuardedResponse
from nemoguardrails.server.experimental.providers.openai.responses.stream_classifier import STREAM_CLASSIFIER
from nemoguardrails.server.experimental.providers.openai.responses.stream_hooks import ResponsesStreamHooks

RESPONSES_ENDPOINT = GuardedJsonEndpoint(
    provider_operation_id="createResponse",
    contract_name="responses",
    route_path="/v1/responses",
    operation_name="responses.create",
    operation="OpenAI Responses",
    unsupported_request_code="unsupported_responses_shape",
    unsupported_response_code="unsupported_responses_response_shape",
    guarded_request_model=ResponsesGuardedRequest,
    guarded_response_model=ResponsesGuardedResponse,
    method="POST",
    operation_paths=(
        GuardedOperationPath("/v1/responses", frozenset({"POST"})),
        GuardedOperationPath("/{api_version}/responses", frozenset({"POST"})),
    ),
    stream=StreamBinding(STREAM_CLASSIFIER, ResponsesStreamHooks),
)
