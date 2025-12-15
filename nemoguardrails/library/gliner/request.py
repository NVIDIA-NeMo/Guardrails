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

"""Module for handling GLiNER detection requests."""

import logging
from typing import Any, Dict, List, Optional

import aiohttp

log = logging.getLogger(__name__)


async def gliner_request(
    text: str,
    server_endpoint: str,
    enabled_entities: Optional[List[str]] = None,
    threshold: float = 0.5,
    chunk_length: int = 384,
    overlap: int = 128,
    flat_ner: bool = False,
) -> Dict[str, Any]:
    """Send a PII detection request to the GLiNER API.

    Args:
        text: The text to analyze.
        server_endpoint: The API endpoint URL (e.g., http://localhost:1235/v1/extract).
        enabled_entities: List of entity types to detect. If None, uses server defaults.
        threshold: Confidence threshold for entity detection (0.0 to 1.0).
        chunk_length: Length of text chunks for processing.
        overlap: Overlap between chunks.
        flat_ner: Whether to use flat NER mode.

    Returns:
        The response from the GLiNER API containing:
        - entities: List of detected entities with value, label, positions, and score
        - total_entities: Count of entities found
        - tagged_text: Text with entities tagged as [entity](label)

    Raises:
        ValueError: If the API call fails or the response cannot be parsed.
    """
    payload: Dict[str, Any] = {
        "text": text,
        "threshold": threshold,
        "chunk_length": chunk_length,
        "overlap": overlap,
        "flat_ner": flat_ner,
    }

    if enabled_entities:
        payload["labels"] = enabled_entities

    headers: Dict[str, str] = {
        "Content-Type": "application/json",
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(server_endpoint, json=payload, headers=headers) as resp:
            if resp.status != 200:
                raise ValueError(f"GLiNER call failed with status code {resp.status}.\nDetails: {await resp.text()}")

            try:
                return await resp.json()
            except aiohttp.ContentTypeError:
                raise ValueError(
                    f"Failed to parse GLiNER response as JSON. Status: {resp.status}, Content: {await resp.text()}"
                )
