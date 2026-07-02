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

from typing import Any, Dict, List, Optional

from nemoguardrails.http import (
    HTTPClient,
    HTTPClientError,
    HTTPResponseDecodeError,
    HTTPTimeoutError,
    http_call,
    resolve_http_client,
)

# Default per-request timeout for Polygraf calls. Matches the timeout pattern
# used by other community guardrail integrations and prevents hung rails when
# the Polygraf endpoint is unresponsive.
DEFAULT_TIMEOUT_SECONDS = 30


async def polygraf_request(
    text: str,
    server_endpoint: str,
    api_key: Optional[str] = None,
    http_client: Optional[HTTPClient] = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> List[Dict[str, Any]]:
    """Send a PII detection request to the Polygraf API.

    Args:
        text: The text to analyze.
        server_endpoint: The API endpoint URL.
        api_key: The API key for the Polygraf service.
        http_client: Optional shared HTTP client. Passing a client lets callers
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

    try:
        async with resolve_http_client(http_client) as client:
            response = await http_call(
                client,
                "POST",
                server_endpoint,
                json=payload,
                headers=headers,
                timeout=timeout,
                raise_for_status=False,
            )
    except HTTPTimeoutError as err:
        raise ValueError(f"Polygraf call timed out after {timeout} seconds.") from err
    except HTTPClientError as err:
        raise ValueError(f"Polygraf call failed: {type(err).__name__}: {err}") from err

    if response.status_code != 200:
        raise ValueError(f"Polygraf call failed with status code {response.status_code}.\nDetails: {response.text}")

    try:
        data = response.json()
    except HTTPResponseDecodeError as err:
        raise ValueError(
            f"Failed to parse Polygraf response as JSON. Status: {response.status_code}, Content: {response.text}"
        ) from err

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
