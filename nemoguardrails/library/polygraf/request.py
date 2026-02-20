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

import logging
from typing import Any, Dict, List, Optional

import aiohttp

log = logging.getLogger(__name__)


async def polygraf_request(
    text: str, server_endpoint: str, api_key: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Send a PII detection request to the Polygraf API.

    Args:
        text: The text to analyze.
        server_endpoint: The API endpoint URL.
        api_key: The API key for the Polygraf service.

    Returns:
        The list of entities detected by the Polygraf server.

    Raises:
        ValueError: If the API call fails or the response cannot be parsed as JSON.
    """
    payload = {"text": text}
    headers: Dict[str, str] = {"Content-Type": "application/json"}

    if api_key:
        headers["API_Key"] = f"Bearer {api_key}"

    async with aiohttp.ClientSession() as session:
        async with session.post(
            server_endpoint, json=payload, headers=headers
        ) as resp:
            if resp.status != 200:
                raise ValueError(
                    f"Polygraf call failed with status code {resp.status}.\nDetails: {await resp.text()}"
                )

            try:
                return await resp.json()
            except aiohttp.ContentTypeError:
                raise ValueError(
                    f"Failed to parse Polygraf response as JSON. Status: {resp.status}, Content: {await resp.text()}"
                )
