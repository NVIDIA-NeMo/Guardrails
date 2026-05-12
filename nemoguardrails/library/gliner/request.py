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

import json
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
    api_key: Optional[str] = None,
    model: str = "nvidia/gliner-pii",
) -> Dict[str, Any]:
    """Send a PII detection request to the GLiNER API.

    Supports two server formats:
    - Custom server (/v1/extract): plain JSON request, no auth required.
    - NIM API (/v1/chat/completions): OpenAI-compatible chat completions format,
      used by both the locally-run NIM container and the NVIDIA-hosted endpoint.
      Requires an API key for the hosted endpoint.

    Args:
        text: The text to analyze.
        server_endpoint: The API endpoint URL.
        enabled_entities: List of entity types to detect. If None, uses server defaults.
        threshold: Confidence threshold for entity detection (0.0 to 1.0).
        chunk_length: Length of text chunks for processing.
        overlap: Overlap between chunks.
        flat_ner: Whether to use flat NER mode.
        api_key: Optional Bearer token for authenticated endpoints.

    Returns:
        Normalized response dict with keys:
        - entities: List of dicts with value, suggested_label, start_position, end_position, score
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

    use_chat_completions = server_endpoint.rstrip("/").endswith("/v1/chat/completions")

    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    if use_chat_completions:
        payload: Dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": request.text}],
            "threshold": request.threshold,
            "chunk_length": request.chunk_length,
            "overlap": request.overlap,
            "flat_ner": request.flat_ner,
        }
    else:
        payload = {
            "text": request.text,
            "threshold": request.threshold,
            "chunk_length": request.chunk_length,
            "overlap": request.overlap,
            "flat_ner": request.flat_ner,
        }

    if request.labels:
        payload["labels"] = request.labels

    async with aiohttp.ClientSession() as session:
        async with session.post(server_endpoint, json=payload, headers=headers) as resp:
            if resp.status != 200:
                raise ValueError(f"GLiNER call failed with status code {resp.status}.\nDetails: {await resp.text()}")

            try:
                raw = await resp.json()
            except aiohttp.ContentTypeError as err:
                raise ValueError(
                    f"Failed to parse GLiNER response as JSON. Status: {resp.status}, Content: {await resp.text()}"
                ) from err

            if not use_chat_completions:
                return raw

            # Unwrap chat completions envelope and normalize entity field names.
            # NIM uses: text, label, start, end
            # Custom server uses: value, suggested_label, start_position, end_position
            try:
                nim_data = json.loads(raw["choices"][0]["message"]["content"])
            except (KeyError, IndexError, json.JSONDecodeError, TypeError) as e:
                raise ValueError(f"Failed to parse NIM response content: {e}") from e

            if not isinstance(nim_data, dict):
                raise ValueError(f"Expected NIM response content to be a JSON object, got {type(nim_data).__name__}")

            normalized_entities = [
                {
                    "value": e.get("text", e.get("value", "")),
                    "suggested_label": e.get("label", e.get("suggested_label", "")),
                    "start_position": e.get("start", e.get("start_position", 0)),
                    "end_position": e.get("end", e.get("end_position", 0)),
                    "score": e.get("score", 0.0),
                }
                for e in (nim_data.get("entities") or [])
            ]
            return {
                "entities": normalized_entities,
                "total_entities": nim_data.get("total_entities", len(normalized_entities)),
                "tagged_text": nim_data.get("tagged_text", ""),
            }
