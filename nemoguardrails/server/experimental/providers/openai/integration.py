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

"""Compose the buffered OpenAI Chat route and provider-native errors."""

from collections.abc import Collection, Mapping

from fastapi import APIRouter

from nemoguardrails.server.experimental._content_checker import ContentChecker
from nemoguardrails.server.experimental._guarded_proxy import create_buffered_guarded_http_operation
from nemoguardrails.server.experimental._http_kernel import (
    DEFAULT_MAX_REQUEST_BODY_BYTES,
    DEFAULT_MAX_RESPONSE_BODY_BYTES,
    HttpDispatch,
    create_http_proxy_router,
)
from nemoguardrails.server.experimental.providers.openai.chat_completions.endpoint import (
    CHAT_COMPLETIONS_ENDPOINT,
)
from nemoguardrails.server.experimental.providers.openai.errors import OPENAI_ERROR_MAPPING


def create_openai_chat_router(
    *,
    checker: ContentChecker,
    dispatch: HttpDispatch,
    reserved_routes: Mapping[str, Collection[str]] | None = None,
    max_request_body_bytes: int = DEFAULT_MAX_REQUEST_BODY_BYTES,
    max_response_body_bytes: int = DEFAULT_MAX_RESPONSE_BODY_BYTES,
) -> APIRouter:
    """Create the private buffered OpenAI Chat proxy surface."""

    return create_http_proxy_router(
        operations=(create_buffered_guarded_http_operation(CHAT_COMPLETIONS_ENDPOINT, OPENAI_ERROR_MAPPING),),
        checker=checker,
        dispatch=dispatch,
        render_outcome=OPENAI_ERROR_MAPPING.renderer,
        reserved_routes=reserved_routes,
        max_request_body_bytes=max_request_body_bytes,
        max_response_body_bytes=max_response_body_bytes,
    )
