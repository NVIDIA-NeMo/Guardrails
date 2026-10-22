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

"""OAuth2 client-credentials tokens for the HiddenLayer API.

Tokens are cached at module level rather than on the config, because the synchronous IORails
entry points deep-copy and pickle the config.
"""

import asyncio
import base64
import functools
import hashlib
import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from nemoguardrails.http import HTTPClient, HTTPResponse, HTTPResponseDecodeError

REQUEST_TIMEOUT_SECONDS = 30.0

_TOKEN_PATH = "/oauth2/token"
_DEFAULT_EXPIRES_IN_SECONDS = 3600.0
_EXPIRY_MARGIN_SECONDS = 10.0


class HiddenLayerTransportError(Exception):
    """A HiddenLayer request failed. The message holds only status, content type and length."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


def _describe(response: HTTPResponse) -> str:
    content_type = next(
        (value for name, value in response.headers.items() if name.lower() == "content-type"),
        "unknown",
    )
    return f"status={response.status_code} content_type={content_type} body_length={len(response.content)}"


@dataclass(frozen=True)
class _CachedToken:
    token: str
    refresh_at: float


_TokenKey = Tuple[str, str, str]

_FetchKey = Tuple[asyncio.AbstractEventLoop, _TokenKey]

_tokens: Dict[_TokenKey, _CachedToken] = {}
# One in-flight fetch per credentials, shared by every caller that misses the cache while it runs,
# so an outage costs one fetch rather than one per queued caller. A task belongs to its loop, and
# the sync IORails entry points run asyncio.run per call, so the loop is part of the key.
_fetches: Dict[_FetchKey, "asyncio.Task[str]"] = {}
_guard = threading.Lock()


def _now() -> float:
    return time.monotonic()


def _cached(key: _TokenKey) -> Optional[str]:
    cached = _tokens.get(key)
    if cached is not None and _now() < cached.refresh_at:
        return cached.token
    return None


def _invalidate(key: _TokenKey, rejected: str) -> None:
    """Drop the cached token only if it is still the rejected one, so concurrent 401s refresh once."""
    with _guard:
        cached = _tokens.get(key)
        if cached is not None and cached.token == rejected:
            del _tokens[key]


async def _fetch(client: HTTPClient, api_url: str, client_id: str, client_secret: str) -> Tuple[str, float]:
    credentials = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    response = await client.request(
        "POST",
        f"{api_url}{_TOKEN_PATH}",
        headers={
            "Authorization": f"Basic {credentials}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        content="grant_type=client_credentials",
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    if response.status_code != 200:
        raise HiddenLayerTransportError(
            f"HiddenLayer token request failed: {_describe(response)}", status_code=response.status_code
        )
    try:
        body = response.json()
    except HTTPResponseDecodeError:
        raise HiddenLayerTransportError(
            f"HiddenLayer token response is not valid JSON: {_describe(response)}", status_code=response.status_code
        ) from None
    token = body.get("access_token") if isinstance(body, dict) else None
    if not isinstance(token, str) or not token:
        raise HiddenLayerTransportError(
            f"HiddenLayer token response has no access_token: {_describe(response)}", status_code=response.status_code
        )
    expires_in = body.get("expires_in")
    if not isinstance(expires_in, (int, float)) or isinstance(expires_in, bool):
        expires_in = _DEFAULT_EXPIRES_IN_SECONDS
    return token, float(expires_in)


async def _access_token(
    client: HTTPClient,
    api_url: str,
    client_id: str,
    client_secret: str,
    *,
    rejected: Optional[str] = None,
) -> str:
    """Return a bearer token for the credentials, fetching one when none is cached and fresh.

    Args:
        client: Client used for the token request.
        api_url: HiddenLayer API base URL, without a trailing slash.
        client_id: OAuth2 client ID.
        client_secret: OAuth2 client secret.
        rejected: A token the API answered with 401. It is dropped from the cache before a new
            one is fetched, unless another request has already replaced it.

    Raises:
        HiddenLayerTransportError: The token endpoint returned a non-200 status or an unusable body.
        HTTPConnectionError, HTTPTimeoutError: The token request failed in transport.
    """
    key = (api_url, client_id, hashlib.sha256(client_secret.encode()).hexdigest())
    if rejected is not None:
        _invalidate(key, rejected)
    token = _cached(key)
    if token is not None:
        return token
    loop = asyncio.get_running_loop()
    fetch_key = (loop, key)
    with _guard:
        for stale in [known for known in _fetches if known[0].is_closed()]:
            del _fetches[stale]
        task = _fetches.get(fetch_key)
        if task is None:
            task = _fetches[fetch_key] = loop.create_task(
                _fetch_and_cache(key, client, api_url, client_id, client_secret)
            )
            task.add_done_callback(functools.partial(_finish, fetch_key))
    # Shielded so that one cancelled caller does not cancel the fetch the others are awaiting.
    return await asyncio.shield(task)


async def _fetch_and_cache(key: _TokenKey, client: HTTPClient, api_url: str, client_id: str, client_secret: str) -> str:
    token, expires_in = await _fetch(client, api_url, client_id, client_secret)
    _tokens[key] = _CachedToken(token, _now() + expires_in - _EXPIRY_MARGIN_SECONDS)
    return token


def _finish(fetch_key: _FetchKey, task: "asyncio.Task[str]") -> None:
    """Retire a completed fetch, so the next cache miss starts a new one."""
    with _guard:
        if _fetches.get(fetch_key) is task:
            del _fetches[fetch_key]
    # Marks a failure as retrieved, so a fetch whose callers were all cancelled logs nothing.
    if not task.cancelled():
        task.exception()


def _reset_token_cache() -> None:
    """Clear every cached token and in-flight fetch (tests only)."""
    with _guard:
        _tokens.clear()
        _fetches.clear()
