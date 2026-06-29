# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Module for handling Polygraf PII detection requests."""

import asyncio
from typing import Any, Dict, List, Optional

import aiohttp

# Default per-request timeout for Polygraf calls. Matches the timeout pattern
# used by other community guardrail integrations and prevents hung rails when
# the Polygraf endpoint is unresponsive.
DEFAULT_TIMEOUT_SECONDS = 30


async def polygraf_request(
    text: str,
    server_endpoint: str,
    api_key: Optional[str] = None,
    session: Optional[aiohttp.ClientSession] = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> List[Dict[str, Any]]:
    """Send a PII detection request to the Polygraf API.

    Args:
        text: The text to analyze.
        server_endpoint: The API endpoint URL.
        api_key: The API key for the Polygraf service.
        session: Optional shared aiohttp session. Passing a session lets callers
            reuse connections across multiple PII checks.
        timeout: Per-request timeout in seconds. Applied to both caller-provided
            and internally created sessions.

    Returns:
        The list of entities detected by the Polygraf server.

    Raises:
        ValueError: If the API call fails, times out, or the response cannot
            be parsed as JSON.
    """
    # Polygraf request payload. Some deployments accept/require additional flags
    # controlling PII/PID detection and aggregation.
    payload = {
        "text": text,
        # NOTE: Kept as `detect_pid` to match the working Polygraf API format
        # provided by users of this integration.
        "detect_pid": True,
        "pid_granularity": 3,
        "aggregate_entities": True,
    }
    headers: Dict[str, str] = {"Content-Type": "application/json"}

    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    client_timeout = aiohttp.ClientTimeout(total=timeout)

    if session is not None:
        return await _send_polygraf_request(session, server_endpoint, payload, headers, client_timeout)

    async with aiohttp.ClientSession(timeout=client_timeout) as request_session:
        return await _send_polygraf_request(request_session, server_endpoint, payload, headers, client_timeout)


async def _send_polygraf_request(
    session: aiohttp.ClientSession,
    server_endpoint: str,
    payload: Dict[str, Any],
    headers: Dict[str, str],
    timeout: aiohttp.ClientTimeout,
) -> List[Dict[str, Any]]:
    try:
        try:
            post_ctx = session.post(server_endpoint, json=payload, headers=headers, timeout=timeout)
        except TypeError:
            # Some test doubles do not accept a `timeout` kwarg; fall back to the
            # session-level timeout instead.
            post_ctx = session.post(server_endpoint, json=payload, headers=headers)

        async with post_ctx as resp:
            if resp.status != 200:
                raise ValueError(f"Polygraf call failed with status code {resp.status}.\nDetails: {await resp.text()}")

            try:
                data = await resp.json()
            except aiohttp.ContentTypeError as err:
                raise ValueError(
                    f"Failed to parse Polygraf response as JSON. Status: {resp.status}, Content: {await resp.text()}"
                ) from err
    except asyncio.TimeoutError as err:
        # `aiohttp.ClientTimeout` surfaces timeouts as asyncio.TimeoutError on
        # both the connect and read paths. Normalize so callers see a single
        # ValueError contract instead of asyncio plumbing exceptions.
        raise ValueError(f"Polygraf call timed out after {timeout.total} seconds.") from err
    except aiohttp.ClientError as err:
        # DNS failures, connection resets, TLS errors, etc. should also surface
        # as ValueError so the documented contract holds across all network
        # failure modes.
        raise ValueError(f"Polygraf call failed: {type(err).__name__}: {err}") from err

    # Polygraf may return either a raw list of entities or a wrapper object.
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if "entities" in data:
            entities = data["entities"]
            if entities is None:
                return []
            if isinstance(entities, list):
                return entities

    raise ValueError(
        "Invalid response from Polygraf service: expected a list of entities or an object with an 'entities' list."
    )
