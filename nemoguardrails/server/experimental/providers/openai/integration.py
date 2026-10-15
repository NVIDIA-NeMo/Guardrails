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
from collections.abc import Collection, Mapping, Sequence
from dataclasses import replace
from typing import Literal

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

OpenAIOperation = Literal["chat_completions", "responses"]
"""Built-in operation names accepted by the OpenAI integration factory."""


def create_openai_integration(
    *,
    instance_id: str = "openai",
    base_url: str | None = None,
    mount_path: str = "",
    operations: Sequence[OpenAIOperation] = ("chat_completions", "responses"),
) -> ProviderIntegration:
    """Create an OpenAI instance with the selected built-in guard bindings.

    ``operations`` is a nonempty sequence of unique names, in binding order.
    The fixed default selects Chat Completions and Responses; it does not mean
    all operations a future version might support. A bare string is invalid.

    Selection installs guard bindings, not a route allowlist. Unselected routes
    retain the application's passthrough behavior and are not thereby guarded.
    These are strict OpenAI boundaries: an upstream URL does not select a
    dialect. Trusted applications can customize explicit endpoint bindings.

    Raises:
        TypeError: ``operations`` is not a sequence of operation names.
        ValueError: The selection is empty, unknown, or contains duplicates.
    """

    if isinstance(operations, (str, bytes)) or not isinstance(operations, Sequence):
        raise TypeError("operations must be a sequence of operation names, not a bare string")
    selected = tuple(operations)
    if not selected:
        raise ValueError("operations must select at least one OpenAI operation")
    endpoints: dict[OpenAIOperation, GuardedJsonEndpoint] = {
        "chat_completions": CHAT_COMPLETIONS_ENDPOINT,
        "responses": RESPONSES_ENDPOINT,
    }
    for operation in selected:
        if not isinstance(operation, str) or operation not in endpoints:
            raise ValueError(f"Unsupported OpenAI operation: {operation!r}; expected chat_completions or responses")
    if len(set(selected)) != len(selected):
        raise ValueError("operations must not contain duplicate names")

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
        endpoints=tuple(endpoints[operation] for operation in selected),
        errors=OPENAI_ERROR_MAPPING,
    )


def create_openai_chat_integration(
    *,
    instance_id: str = "openai",
    base_url: str | None = None,
    mount_path: str = "",
    endpoint: GuardedJsonEndpoint | None = None,
) -> ProviderIntegration:
    """Create a Chat-only OpenAI instance with an explicitly reviewed endpoint.

    Omitting ``endpoint`` selects the strict OpenAI Chat boundary through
    :func:`create_openai_integration`. A trusted application may instead supply
    an OpenAI-wire-compatible dialect endpoint; it must use the OpenAI error
    mapping. Other instances are unaffected. Unselected routes retain the
    application's passthrough behavior.
    """
    integration = create_openai_integration(
        instance_id=instance_id, base_url=base_url, mount_path=mount_path, operations=("chat_completions",)
    )
    return integration if endpoint is None else replace(integration, endpoints=(endpoint,))


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
