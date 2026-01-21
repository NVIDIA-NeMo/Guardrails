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

"""PII detection using GLiNER."""

import logging
from typing import List

from nemoguardrails import RailsConfig
from nemoguardrails.actions import action
from nemoguardrails.library.gliner.request import gliner_request
from nemoguardrails.rails.llm.config import GLiNERDetection

log = logging.getLogger(__name__)


def detect_pii_mapping(result: bool) -> bool:
    """
    Mapping for detect_pii.

    Since the function returns True when PII is detected,
    we block if result is True.
    """
    return result


def _mask_text_with_entities(text: str, entities: List[dict]) -> str:
    """
    Mask detected entities in text with their labels.

    Args:
        text: Original text
        entities: List of entity dictionaries with 'value', 'suggested_label',
                 'start_position', 'end_position' keys

    Returns:
        Text with entities replaced by [LABEL] placeholders
    """
    if not entities:
        return text

    # Sort entities by start position in reverse order to replace from end to start
    sorted_entities = sorted(entities, key=lambda x: x["start_position"], reverse=True)

    masked_text = text
    for entity in sorted_entities:
        start = entity["start_position"]
        end = entity["end_position"]
        label = entity["suggested_label"].upper()
        masked_text = masked_text[:start] + f"[{label}]" + masked_text[end:]

    return masked_text


@action(is_system_action=False, output_mapping=detect_pii_mapping)
async def gliner_detect_pii(
    source: str,
    text: str,
    config: RailsConfig,
    **kwargs,
):
    """Checks whether the provided text contains any PII using GLiNER.

    Args:
        source: The source for the text, i.e. "input", "output", "retrieval".
        text: The text to check.
        config: The rails configuration object.

    Returns:
        True if PII is detected, False otherwise.

    Raises:
        ValueError: If the response is invalid or source is not valid.
    """
    gliner_config: GLiNERDetection = getattr(config.rails.config, "gliner")
    server_endpoint = gliner_config.server_endpoint
    source_config = getattr(gliner_config, source, None)

    if source_config is None:
        valid_sources = ["input", "output", "retrieval"]
        raise ValueError(
            f"GLiNER can only be defined in the following flows: {valid_sources}. "
            f"The current flow, '{source}', is not allowed."
        )

    enabled_entities = source_config.entities if source_config.entities else None

    gliner_response = await gliner_request(
        text=text,
        server_endpoint=server_endpoint,
        enabled_entities=enabled_entities,
        threshold=gliner_config.threshold,
        chunk_length=gliner_config.chunk_length,
        overlap=gliner_config.overlap,
        flat_ner=gliner_config.flat_ner,
    )

    try:
        total_entities = gliner_response.get("total_entities", 0)
        return total_entities > 0
    except (KeyError, TypeError) as e:
        raise ValueError(f"Invalid response from GLiNER service: {str(e)}")


@action(is_system_action=False)
async def gliner_mask_pii(source: str, text: str, config: RailsConfig):
    """Masks any detected PII in the provided text using GLiNER.

    Args:
        source: The source for the text, i.e. "input", "output", "retrieval".
        text: The text to check.
        config: The rails configuration object.

    Returns:
        The altered text with PII masked.

    Raises:
        ValueError: If the response is invalid or source is not valid.
    """
    gliner_config: GLiNERDetection = getattr(config.rails.config, "gliner")
    server_endpoint = gliner_config.server_endpoint
    source_config = getattr(gliner_config, source, None)

    if source_config is None:
        valid_sources = ["input", "output", "retrieval"]
        raise ValueError(
            f"GLiNER can only be defined in the following flows: {valid_sources}. "
            f"The current flow, '{source}', is not allowed."
        )

    enabled_entities = source_config.entities if source_config.entities else None

    gliner_response = await gliner_request(
        text=text,
        server_endpoint=server_endpoint,
        enabled_entities=enabled_entities,
        threshold=gliner_config.threshold,
        chunk_length=gliner_config.chunk_length,
        overlap=gliner_config.overlap,
        flat_ner=gliner_config.flat_ner,
    )

    if not gliner_response or not isinstance(gliner_response, dict):
        raise ValueError("Invalid response received from GLiNER service.")

    try:
        entities = gliner_response.get("entities", [])
        return _mask_text_with_entities(text, entities)
    except (KeyError, TypeError) as e:
        raise ValueError(f"Invalid response from GLiNER service: {str(e)}")
