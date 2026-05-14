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

"""PII detection using Polygraf."""

import logging
import os

from nemoguardrails import RailsConfig
from nemoguardrails.actions.actions import action
from nemoguardrails.library.polygraf.request import polygraf_request
from nemoguardrails.rails.llm.config import PolygrafDetection

log = logging.getLogger(__name__)


def detect_pii_mapping(result: bool) -> bool:
    """
    Mapping for polygraf_detect_pii.

    Since the function returns True when PII is detected,
    we block if result is True.
    """
    return result


def _get_polygraf_api_key() -> str | None:
    api_key = os.environ.get("POLYGRAF_API_KEY")
    if not api_key:
        log.warning(
            "POLYGRAF_API_KEY environment variable is not set. "
            "Polygraf cloud endpoints may reject unauthenticated requests."
        )
    return api_key


@action(is_system_action=False, output_mapping=detect_pii_mapping)
async def polygraf_detect_pii(
    source: str,
    text: str,
    config: RailsConfig,
    **kwargs,
) -> bool:
    """Checks whether the provided text contains any PII using Polygraf.

    Args:
        source: The source for the text, i.e. "input", "output", "retrieval".
        text: The text to check.
        config: The rails configuration object.

    Returns:
        True if PII is detected, False otherwise.

    Raises:
        ValueError: If the response is invalid or source is not valid.
    """
    polygraf_config: PolygrafDetection = getattr(config.rails.config, "polygraf")
    server_endpoint = polygraf_config.server_endpoint
    source_config = getattr(polygraf_config, source, None)

    if source_config is None:
        valid_sources = ["input", "output", "retrieval"]
        raise ValueError(
            f"Polygraf can only be defined in the following flows: {valid_sources}. "
            f"The current flow, '{source}', is not allowed."
        )

    enabled_entities = source_config.entities if source_config.entities else None
    api_key = _get_polygraf_api_key()
    session = kwargs.get("session")

    entities = await polygraf_request(text, server_endpoint, api_key, session=session)

    if not entities:
        return False

    if enabled_entities:
        return any(e["entity_type"] in enabled_entities for e in entities)

    return True


@action(is_system_action=False)
async def polygraf_mask_pii(source: str, text: str, config: RailsConfig, **kwargs) -> str:
    """Masks any detected PII in the provided text using Polygraf.

    Args:
        source: The source for the text, i.e. "input", "output", "retrieval".
        text: The text to check.
        config: The rails configuration object.

    Returns:
        The altered text with PII masked.

    Raises:
        ValueError: If the response is invalid or source is not valid.
    """
    polygraf_config: PolygrafDetection = getattr(config.rails.config, "polygraf")
    server_endpoint = polygraf_config.server_endpoint
    source_config = getattr(polygraf_config, source, None)

    if source_config is None:
        valid_sources = ["input", "output", "retrieval"]
        raise ValueError(
            f"Polygraf can only be defined in the following flows: {valid_sources}. "
            f"The current flow, '{source}', is not allowed."
        )

    enabled_entities = source_config.entities if source_config.entities else None
    api_key = _get_polygraf_api_key()
    session = kwargs.get("session")

    entities = await polygraf_request(text, server_endpoint, api_key, session=session)

    if not entities:
        return text

    if enabled_entities:
        entities = [e for e in entities if e["entity_type"] in enabled_entities]

    masked_text = text
    for entity in sorted(entities, key=lambda x: x["start"], reverse=True):
        start = entity["start"]
        end = entity["end"]
        entity_type = entity["entity_type"]
        masked_text = masked_text[:start] + f"<{entity_type}>" + masked_text[end:]

    return masked_text
