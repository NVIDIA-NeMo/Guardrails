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

"""Construction helpers for managed outbound HTTP clients."""

from __future__ import annotations

import httpx

from nemoguardrails.http.client import ClosableHTTPClient
from nemoguardrails.http.retry import RetryingHTTPClient, RetryPolicy
from nemoguardrails.http.transport import HttpxHTTPClient


def create_http_client(
    *,
    httpx_client: httpx.AsyncClient | None = None,
    timeout: float | None = 30.0,
    limits: httpx.Limits | None = None,
    retry_policy: RetryPolicy | None = None,
    follow_redirects: bool = False,
) -> ClosableHTTPClient:
    """Create a transport-neutral HTTP client with optional retry behavior.

    Args:
        httpx_client: Optional caller-owned HTTPX client.
        timeout: Total timeout for a client created by this function.
        limits: Connection-pool limits for a client created by this function.
        retry_policy: Optional policy applied to created or injected clients.
        follow_redirects: Whether a client created by this function follows
            redirects.

    Returns:
        A closable transport-neutral HTTP client.

    Raises:
        ValueError: If non-default owned-client options are supplied with an
            injected client.

    When ``httpx_client`` is provided, it remains caller-owned and ``timeout``,
    ``limits``, and ``follow_redirects`` must retain their defaults.
    ``retry_policy`` is still applied when supplied.
    """

    transport = HttpxHTTPClient(
        httpx_client,
        timeout=timeout,
        limits=limits,
        follow_redirects=follow_redirects,
    )
    client: ClosableHTTPClient = transport
    if retry_policy is not None:
        client = RetryingHTTPClient(transport, retry_policy)
    return client
