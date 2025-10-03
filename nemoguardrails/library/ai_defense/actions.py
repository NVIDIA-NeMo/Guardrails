# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Prompt/Response protection using Cisco AI Defense."""

import logging
import os
from typing import Any, Dict, Optional

import httpx

from nemoguardrails.actions import action

log = logging.getLogger(__name__)


def ai_defense_text_mapping(result: dict) -> bool:
    """
    Mapping for inspect API response.
    Expects result to be a dict with:
      - "is_blocked": a boolean indicating if the prompt or response sent to AI Defense should be blocked.

    Returns:
        True if "is_blocked" is True (i.e., the response should be blocked),
        False otherwise.
    """
    # If the provider does not return "is_safe", default to safe (not blocked)
    is_blocked = result.get("is_blocked", True)
    return is_blocked


@action(is_system_action=True, output_mapping=ai_defense_text_mapping)
async def ai_defense_inspect(
    user_prompt: Optional[str] = None, bot_response: Optional[str] = None, **kwargs
):
    api_key = os.environ.get("AI_DEFENSE_API_KEY")
    if api_key is None:
        msg = "AI_DEFENSE_API_KEY environment variable not set."
        log.error(msg)
        raise ValueError(msg)

    api_endpoint = os.environ.get("AI_DEFENSE_API_ENDPOINT")
    if api_endpoint is None:
        msg = "AI_DEFENSE_API_ENDPOINT environment variable not set."
        log.error(msg)
        raise ValueError(msg)

    headers = {
        "X-Cisco-AI-Defense-API-Key": api_key,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    if bot_response is not None:
        role = "assistant"
        text = str(bot_response)
    elif user_prompt is not None:
        role = "user"
        text = str(user_prompt)
    else:
        msg = "Either user_prompt or bot_response must be provided"
        log.error(msg)
        raise ValueError(msg)

    messages = [{"role": role, "content": text}]

    metadata = None
    user = kwargs.get("user")
    if user is not None:
        metadata = {"user": user}

    payload: Dict[str, Any] = {"messages": messages}
    if metadata:
        payload["metadata"] = metadata

    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(api_endpoint, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as e:
            msg = f"Error calling AI Defense API: {e}"
            log.error(msg)
            raise ValueError(msg)

        # Compose a consistent return structure for flows
        is_safe = bool(data.get("is_safe", True))
        rules = data.get("rules") or []
        if not is_safe and rules:
            entries = [
                f"{r.get('rule_name')} ({r.get('classification')})"
                for r in rules
                if isinstance(r, dict)
            ]
            if entries:
                log.info("AI Defense matched rules: %s", ", ".join(entries))

        # Ensure flows can check explicit block flag
        result: Dict[str, Any] = {
            "is_blocked": (not is_safe),
            "is_safe": is_safe,
        }

        return result
