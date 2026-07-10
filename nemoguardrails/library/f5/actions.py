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

import asyncio
import logging
import os
from typing import Any, Optional

import aiohttp
from typing_extensions import cast

from nemoguardrails.actions import action
from nemoguardrails.rails.llm.config import F5GuardrailsRailConfig, RailsConfig

log = logging.getLogger(__name__)


def f5_guardrails_scan_mapping(result: dict) -> bool:
    """
    Mapping for f5_guardrails_scan.
    Returns True (blocked) if the outcome is not 'cleared'.
    """
    # The result is the dictionary returned by the action
    outcome = result.get("result", {}).get("outcome")
    return outcome != "cleared"


def _get_f5_config(config: Optional[RailsConfig]) -> F5GuardrailsRailConfig:
    if config is None:
        return F5GuardrailsRailConfig()
    rails_config = getattr(config.rails, "config", None)
    f5_config = getattr(rails_config, "f5", None) if rails_config is not None else None
    if f5_config is None:
        return F5GuardrailsRailConfig()
    return cast(F5GuardrailsRailConfig, f5_config)


def _fail_open_response() -> dict:
    """Response returned when the API is unreachable and fail_open is enabled.

    The ``fail_open`` marker makes this distinguishable from a real cleared
    scan in logs and traces, while ``result.outcome`` remains ``cleared`` so
    the mapping function does not block the request.
    """
    return {"result": {"outcome": "cleared"}, "fail_open": True}


@action(
    name="f5_guardrails_scan",
    is_system_action=True,
    output_mapping=f5_guardrails_scan_mapping,
)
async def f5_guardrails_scan(
    text: str,
    config: Optional[RailsConfig] = None,
    **kwargs: Any,
) -> dict:
    """
    Scans the provided text using the F5 Guardrails API.

    Args:
        text: The text to scan.
        config: The active RailsConfig; used to resolve ``rails.config.f5``.

    Returns:
        The response from the F5 Guardrails API. When the API call fails and
        ``rails.config.f5.fail_open`` is enabled, returns a response of
        ``{"result": {"outcome": "cleared"}, "fail_open": True}``.

    Resolution order for the API base URL:
        1. ``F5_GUARDRAILS_API_URL`` environment variable.
        2. ``rails.config.f5.api_url`` if set.
        3. The built-in default ``https://us1.calypsoai.app``.
    """
    f5_config = _get_f5_config(config)
    api_url = os.getenv("F5_GUARDRAILS_API_URL") or f5_config.api_url
    fail_open = f5_config.fail_open
    api_key = os.getenv("F5_GUARDRAILS_API_KEY")

    if not api_key:
        raise ValueError("F5 Guardrails API key not found. Please set F5_GUARDRAILS_API_KEY.")

    endpoint = f"{api_url.rstrip('/')}/backend/v1/scans"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "input": text,
    }

    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        try:
            async with session.post(endpoint, headers=headers, json=payload) as response:
                if response.status != 200:
                    content_type = response.headers.get("Content-Type", "unknown")
                    body_length = response.content_length if response.content_length is not None else "unknown"
                    log.error(
                        "F5 Guardrails API call failed: status=%s content_type=%s body_length=%s",
                        response.status,
                        content_type,
                        body_length,
                    )

                    if fail_open:
                        log.warning("F5 Guardrails API call failed; fail_open is enabled, allowing content.")
                        return _fail_open_response()

                    raise RuntimeError(f"F5 Guardrails API error: {response.status}")

                result = await response.json()
                return result
        except asyncio.TimeoutError:
            log.error("F5 Guardrails API call timed out after 30 seconds")

            if fail_open:
                log.warning("F5 Guardrails API call timed out; fail_open is enabled, allowing content.")
                return _fail_open_response()

            raise RuntimeError("F5 Guardrails API request timed out") from None
        except aiohttp.ClientError as e:
            log.error("Error connecting to F5 Guardrails API: %s", type(e).__name__)

            if fail_open:
                log.warning("F5 Guardrails API call failed; fail_open is enabled, allowing content.")
                return _fail_open_response()

            raise RuntimeError(f"Connection error to F5 Guardrails API: {type(e).__name__}") from e
