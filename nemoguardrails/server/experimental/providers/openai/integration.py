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

"""Compose explicit OpenAI operation sets and provider-native errors."""

import os
from collections.abc import Collection, Mapping
from dataclasses import replace

from fastapi import APIRouter

from nemoguardrails.server.experimental._content_checker import ContentChecker
from nemoguardrails.server.experimental._guarded_proxy import create_guarded_http_operation
from nemoguardrails.server.experimental._http_kernel import (
    DEFAULT_MAX_REQUEST_BODY_BYTES,
    DEFAULT_MAX_RESPONSE_BODY_BYTES,
    HttpDispatch,
    create_http_proxy_router,
)
from nemoguardrails.server.experimental._streaming_http import StreamingHttpDispatch
from nemoguardrails.server.experimental.provider.endpoint import GuardedJsonEndpoint
from nemoguardrails.server.experimental.provider.integration import ProviderIntegration, UpstreamSettings
from nemoguardrails.server.experimental.providers.openai.chat_completions.endpoint import (
    CHAT_COMPLETIONS_ENDPOINT,
)
from nemoguardrails.server.experimental.providers.openai.errors import OPENAI_ERROR_MAPPING
from nemoguardrails.server.experimental.providers.openai.responses.endpoint import RESPONSES_ENDPOINT


def create_openai_chat_integration(
    *,
    instance_id: str = "openai",
    base_url: str | None = None,
    mount_path: str = "",
    endpoint: GuardedJsonEndpoint | None = None,
) -> ProviderIntegration:
    """Create an OpenAI-protocol instance with an explicitly reviewed endpoint.

    Omitting ``endpoint`` retains the strict OpenAI Chat boundary. A trusted
    application may instead supply an OpenAI-wire-compatible dialect endpoint;
    it must use the OpenAI error mapping. Other instances are unaffected.
    """

    resolved_base_url = (
        base_url
        if base_url is not None
        else os.getenv("NEMO_GUARDRAILS_OPENAI_PROXY_UPSTREAM_BASE_URL", "https://api.openai.com/v1")
    )
    return ProviderIntegration(
        provider_type="openai",
        instance_id=instance_id,
        upstream=UpstreamSettings(base_url=resolved_base_url, mount_path=mount_path),
        display_name="OpenAI",
        endpoints=(CHAT_COMPLETIONS_ENDPOINT if endpoint is None else endpoint,),
        errors=OPENAI_ERROR_MAPPING,
    )


def create_openai_integration(
    *,
    instance_id: str = "openai",
    base_url: str | None = None,
    mount_path: str = "",
) -> ProviderIntegration:
    """Guard both Chat Completions and Responses on one OpenAI instance.

    This is an explicit opt-in to a second guarded operation, not an alias of
    the Chat-only factory. Other routes retain the application's passthrough
    behavior. A compatible upstream URL does not establish dialect support for
    either operation; this factory selects the strict OpenAI boundaries.
    """
    chat = create_openai_chat_integration(instance_id=instance_id, base_url=base_url, mount_path=mount_path)
    return replace(chat, endpoints=(*chat.endpoints, RESPONSES_ENDPOINT))


def create_openai_chat_router(
    *,
    checker: ContentChecker,
    dispatch: HttpDispatch,
    stream_dispatch: StreamingHttpDispatch | None = None,
    reserved_routes: Mapping[str, Collection[str]] | None = None,
    max_request_body_bytes: int = DEFAULT_MAX_REQUEST_BODY_BYTES,
    max_response_body_bytes: int = DEFAULT_MAX_RESPONSE_BODY_BYTES,
    endpoint: GuardedJsonEndpoint | None = None,
) -> APIRouter:
    """Create the private buffered and streaming OpenAI Chat proxy surface.

    A trusted application may explicitly supply a reviewed OpenAI-wire-compatible
    endpoint, such as the LiteLLM subset. Omitting it retains the OpenAI policy.
    This does not detect a backend, load a contract, or select a dialect from
    request data. The supplied endpoint must use the OpenAI error mapping.
    """

    return create_http_proxy_router(
        operations=(
            create_guarded_http_operation(
                CHAT_COMPLETIONS_ENDPOINT if endpoint is None else endpoint,
                OPENAI_ERROR_MAPPING,
                stream_dispatch=stream_dispatch,
                max_stream_event_bytes=max_response_body_bytes,
                max_pending_stream_bytes=max_response_body_bytes,
            ),
        ),
        checker=checker,
        dispatch=dispatch,
        render_outcome=OPENAI_ERROR_MAPPING.renderer,
        reserved_routes=reserved_routes,
        max_request_body_bytes=max_request_body_bytes,
        max_response_body_bytes=max_response_body_bytes,
    )
