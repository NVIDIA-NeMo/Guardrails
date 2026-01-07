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

"""Module for handling GLiNER detection requests."""

import logging
from typing import Any, Dict, List, Optional

import aiohttp

from nemoguardrails.library.gliner.models import GLiNERRequest

log = logging.getLogger(__name__)


async def gliner_request(
    text: str,
    server_endpoint: str,
    enabled_entities: Optional[List[str]] = None,
    threshold: Optional[float] = None,
    chunk_length: Optional[int] = None,
    overlap: Optional[int] = None,
    flat_ner: Optional[bool] = None,
) -> Dict[str, Any]:
    """Send a PII detection request to the GLiNER API.

    Args:
        text: The text to analyze.
        server_endpoint: The API endpoint URL (e.g., http://localhost:1235/v1/extract).
        enabled_entities: List of entity types to detect. If None, uses server defaults.
        threshold: Confidence threshold for entity detection (0.0 to 1.0).
            If None, uses default from GLiNERRequest model.
        chunk_length: Length of text chunks for processing.
            If None, uses default from GLiNERRequest model.
        overlap: Overlap between chunks.
            If None, uses default from GLiNERRequest model.
        flat_ner: Whether to use flat NER mode.
            If None, uses default from GLiNERRequest model.

    Returns:
        The response from the GLiNER API containing:
        - entities: List of detected entities with value, label, positions, and score
        - total_entities: Count of entities found
        - tagged_text: Text with entities tagged as [entity](label)

    Raises:
        ValueError: If the API call fails or the response cannot be parsed.
    """
    # Build request using GLiNERRequest model to get defaults
    request_data: Dict[str, Any] = {"text": text}
    if enabled_entities is not None:
        request_data["labels"] = enabled_entities
    if threshold is not None:
        request_data["threshold"] = threshold
    if chunk_length is not None:
        request_data["chunk_length"] = chunk_length
    if overlap is not None:
        request_data["overlap"] = overlap
    if flat_ner is not None:
        request_data["flat_ner"] = flat_ner

    # Create GLiNERRequest to apply defaults
    request = GLiNERRequest(**request_data)

    payload: Dict[str, Any] = {
        "text": request.text,
        "threshold": request.threshold,
        "chunk_length": request.chunk_length,
        "overlap": request.overlap,
        "flat_ner": request.flat_ner,
    }

    if request.labels:
        payload["labels"] = request.labels

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
