# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Module for handling Private AI detection requests."""

import logging
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import aiohttp

log = logging.getLogger(__name__)


async def private_ai_request(
    text: str,
    enabled_entities: List[str],
    server_endpoint: str,
    api_key: Optional[str] = None,
):
    """Send a PII detection request to the Private AI API.

    Args:
        text: The text to analyze.
        enabled_entities: List of entity types to detect.
        server_endpoint: The API endpoint URL.
        api_key: The API key for the Private AI service.

    Returns:
        The response from the Private AI API. See Private AI API reference for more details:
        https://docs.private-ai.com/reference/latest/operation/process_text_process_text_post/

    Raises:
        ValueError: If api_key is missing for cloud API, if the API call fails,
            or if the response cannot be parsed as JSON.
    """
    parsed_url = urlparse(server_endpoint)
    if parsed_url.hostname == "api.private-ai.com" and not api_key:
        raise ValueError("'api_key' is required for Private AI cloud API.")

    payload: Dict[str, Any] = {
        "text": [text],
        "link_batch": False,
        "entity_detection": {"accuracy": "high_automatic", "return_entity": False},
    }

    headers: Dict[str, str] = {
        "Content-Type": "application/json",
    }

    if api_key:
        headers["x-api-key"] = api_key

    if enabled_entities:
        payload["entity_detection"]["entity_types"] = [{"type": "ENABLE", "value": enabled_entities}]

    async with aiohttp.ClientSession() as session:
        async with session.post(server_endpoint, json=payload, headers=headers) as resp:
            if resp.status != 200:
                raise ValueError(
                    f"Private AI call failed with status code {resp.status}.\nDetails: {await resp.text()}"
                )

            try:
                return await resp.json()
            except aiohttp.ContentTypeError:
                raise ValueError(
                    f"Failed to parse Private AI response as JSON. Status: {resp.status}, Content: {await resp.text()}"
                )
