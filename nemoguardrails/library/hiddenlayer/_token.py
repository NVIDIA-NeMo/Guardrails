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

import base64
import hashlib
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from nemoguardrails.http import (
    ClosableHTTPClient,
    HTTPClient,
    HTTPResponse,
    HTTPResponseDecodeError,
    http_call,
)
from nemoguardrails.library.hiddenlayer.errs import HiddenLayerTransportError

REQUEST_TIMEOUT_SECONDS = 30.0

_TOKEN_PATH = "/oauth2/token"
_DEFAULT_EXPIRES_IN_SECONDS = 3600.0
_EXPIRY_MARGIN_SECONDS = 10.0


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

_tokens: Dict[_TokenKey, _CachedToken] = {}


def _now() -> float:
    return time.monotonic()


async def _fetch(
    client: Optional[HTTPClient],
    api_url: str,
    client_id: str,
    client_secret: str,
    factory: Callable[[], ClosableHTTPClient],
) -> Tuple[str, float]:
    credentials = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    response = await http_call(
        client,
        "POST",
        f"{api_url}{_TOKEN_PATH}",
        headers={
            "Authorization": f"Basic {credentials}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        content="grant_type=client_credentials",
        timeout=REQUEST_TIMEOUT_SECONDS,
        raise_for_status=False,
        factory=factory,
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
    client: Optional[HTTPClient],
    api_url: str,
    client_id: str,
    client_secret: str,
    factory: Callable[[], ClosableHTTPClient],
    *,
    rejected: Optional[str] = None,
) -> str:
    """Return a bearer token for the credentials, fetching one when none is cached and fresh.

    Args:
        client: Caller-owned client, or None to let factory supply one for the request.
        api_url: HiddenLayer API base URL, without a trailing slash.
        factory: Builds the client used when client is None.
        client_id: OAuth2 client ID.
        client_secret: OAuth2 client secret.
        rejected: Set when the API answered 401, which drops the cached token so this call
            fetches a fresh one.

    Raises:
        HiddenLayerTransportError: The token endpoint returned a non-200 status or an unusable body.
        HTTPConnectionError, HTTPTimeoutError: The token request failed in transport.
    """
    key = (api_url, client_id, hashlib.sha256(client_secret.encode()).hexdigest())
    if rejected is not None:
        # Drop only the token that was rejected: a concurrent 401 may already have replaced it,
        # and popping unconditionally would discard that good token and refetch again.
        stale = _tokens.get(key)
        if stale is not None and stale.token == rejected:
            _tokens.pop(key, None)
    cached = _tokens.get(key)
    if cached is not None and _now() < cached.refresh_at:
        return cached.token
    token, expires_in = await _fetch(client, api_url, client_id, client_secret, factory)
    _tokens[key] = _CachedToken(token, _now() + expires_in - _EXPIRY_MARGIN_SECONDS)
    return token


def _reset_token_cache() -> None:
    """Clear every cached token (tests only)."""
    _tokens.clear()
