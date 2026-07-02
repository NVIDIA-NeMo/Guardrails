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
from typing import Any

import aiohttp

from nemoguardrails.actions import action

log = logging.getLogger(__name__)


def f5_guardrails_scan_mapping(result: dict) -> bool:
    """
    Mapping for f5_guardrails_scan.
    Returns True (blocked) if the outcome is not 'cleared'.
    """
    # The result is the dictionary returned by the action
    outcome = result.get("result", {}).get("outcome")
    return outcome != "cleared"


@action(
    name="f5_guardrails_scan",
    is_system_action=True,
    output_mapping=f5_guardrails_scan_mapping,
)
async def f5_guardrails_scan(
    text: str,
    **kwargs: Any,
) -> dict:
    """
    Scans the provided text using the F5 Guardrails API.

    Args:
        text: The text to scan.

    Returns:
        The response from the F5 Guardrails API.
    """
    api_url = os.getenv("F5_GUARDRAILS_API_URL", "https://us1.calypsoai.app")
    api_key = os.getenv("F5_GUARDRAILS_API_KEY")
    fail_open = os.getenv("F5_GUARDRAILS_FAIL_OPEN", "false").lower() in {"true", "yes", "1"}

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
                    error_detail = await response.text()
                    log.error(f"F5 Guardrails API call failed: {response.status} - {error_detail[:200]}")

                    if fail_open:
                        log.warning(
                            "F5 Guardrails API call failed, but F5_GUARDRAILS_FAIL_OPEN is enabled, allowing content.")
                        return {"result": {"outcome": "cleared"}}

                    raise RuntimeError(f"F5 Guardrails API error: {response.status}")

                result = await response.json()
                return result
        except asyncio.TimeoutError:
            log.error("F5 Guardrails API call timed out after 30 seconds")

            if fail_open:
                log.warning(
                    "F5 Guardrails API call timed out, but F5_GUARDRAILS_FAIL_OPEN is enabled, allowing content."
                )
                return {"result": {"outcome": "cleared"}}

            raise RuntimeError("F5 Guardrails API request timed out") from None
        except aiohttp.ClientError as e:
            log.error(f"Error connecting to F5 Guardrails API: {str(e)}")

            if fail_open:
                log.warning("F5 Guardrails API call failed, but F5_GUARDRAILS_FAIL_OPEN is enabled, allowing content.")
                return {"result": {"outcome": "cleared"}}

            raise RuntimeError(f"Connection error to F5 Guardrails API: {str(e)}") from e
