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

"""Staged guarded endpoint declaration for OpenAI Chat Completions."""

from nemoguardrails.server.experimental._http_kernel import GuardedOperationPath
from nemoguardrails.server.experimental.provider.endpoint import GuardedJsonEndpoint
from nemoguardrails.server.experimental.providers.openai.chat_completions.request_binding import (
    ChatCompletionsGuardedRequest,
)
from nemoguardrails.server.experimental.providers.openai.chat_completions.response_binding import (
    ChatCompletionsGuardedResponse,
)

CHAT_COMPLETIONS_ENDPOINT = GuardedJsonEndpoint(
    route_path="/v1/chat/completions",
    operation_name="chat_completions.create",
    operation="OpenAI Chat Completions",
    unsupported_request_code="unsupported_chat_completions_shape",
    unsupported_response_code="unsupported_chat_completions_response_shape",
    guarded_request_model=ChatCompletionsGuardedRequest,
    guarded_response_model=ChatCompletionsGuardedResponse,
    method="POST",
    operation_paths=(
        GuardedOperationPath(
            "/{api_version}/chat/completions",
            frozenset({"POST"}),
        ),
    ),
)
