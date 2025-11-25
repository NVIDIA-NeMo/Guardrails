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

"""PII detection using Private AI."""

import logging
import os
from urllib.parse import urlparse

from nemoguardrails import RailsConfig
from nemoguardrails.actions import action
from nemoguardrails.library.privateai.request import private_ai_request
from nemoguardrails.rails.llm.config import PrivateAIDetection

log = logging.getLogger(__name__)


def detect_pii_mapping(result: bool) -> bool:
    """
    Mapping for detect_pii.

    Since the function returns True when PII is detected,
    we block if result is True.
    """
    return result


@action(is_system_action=False, output_mapping=detect_pii_mapping)
async def detect_pii(
    source: str,
    text: str,
    config: RailsConfig,
    **kwargs,
):
    """Checks whether the provided text contains any PII.

    Args
        source: The source for the text, i.e. "input", "output", "retrieval".
        text: The text to check.
        config: The rails configuration object.

    Returns
        True if PII is detected, False otherwise.

    Raises:
        ValueError: If PAI_API_KEY is missing when using cloud API or if the response is invalid.
    """
    pai_config: PrivateAIDetection = getattr(config.rails.config, "privateai")
    pai_api_key = os.environ.get("PAI_API_KEY")
    server_endpoint = pai_config.server_endpoint
    enabled_entities = getattr(pai_config, source).entities

    parsed_url = urlparse(server_endpoint)
    if parsed_url.hostname == "api.private-ai.com" and not pai_api_key:
        raise ValueError("PAI_API_KEY environment variable required for Private AI cloud API.")

    valid_sources = ["input", "output", "retrieval"]
    if source not in valid_sources:
        raise ValueError(
            f"Private AI can only be defined in the following flows: {valid_sources}. "
            f"The current flow, '{source}', is not allowed."
        )

    private_ai_response = await private_ai_request(
        text,
        enabled_entities,
        server_endpoint,
        pai_api_key,
    )

    try:
        entity_detected = any(res["entities_present"] for res in private_ai_response)
    except (KeyError, TypeError) as e:
        raise ValueError(f"Invalid response from Private AI service: {str(e)}")
    return entity_detected


@action(is_system_action=False)
async def mask_pii(source: str, text: str, config: RailsConfig):
    """Masks any detected PII in the provided text.

    Args:
        source (str): The source for the text, i.e. "input", "output", "retrieval".
        text (str): The text to check.
        config (RailsConfig): The rails configuration object.

    Returns:
        str: The altered text with PII masked.

    Raises:
        ValueError: If PAI_API_KEY is missing when using cloud API or if the response is invalid.
    """
    pai_config: PrivateAIDetection = getattr(config.rails.config, "privateai")
    pai_api_key = os.environ.get("PAI_API_KEY")
    server_endpoint = pai_config.server_endpoint
    enabled_entities = getattr(pai_config, source).entities

    parsed_url = urlparse(server_endpoint)
    if parsed_url.hostname == "api.private-ai.com" and not pai_api_key:
        raise ValueError("PAI_API_KEY environment variable required for Private AI cloud API.")

    valid_sources = ["input", "output", "retrieval"]
    if source not in valid_sources:
        raise ValueError(
            f"Private AI can only be defined in the following flows: {valid_sources}. "
            f"The current flow, '{source}', is not allowed."
        )

    private_ai_response = await private_ai_request(
        text,
        enabled_entities,
        server_endpoint,
        pai_api_key,
    )

    if not private_ai_response or not isinstance(private_ai_response, list):
        raise ValueError("Invalid response received from Private AI service. The response is not a list.")

    try:
        return private_ai_response[0]["processed_text"]
    except (IndexError, KeyError) as e:
        raise ValueError(f"Invalid response from Private AI service: {str(e)}")
