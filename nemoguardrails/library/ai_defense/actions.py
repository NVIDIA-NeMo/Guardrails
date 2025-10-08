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

from nemoguardrails import RailsConfig
from nemoguardrails.actions import action

log = logging.getLogger(__name__)

# Default timeout for AI Defense API calls in seconds
DEFAULT_TIMEOUT = 30.0


def is_ai_defense_text_blocked(result: Dict[str, Any]) -> bool:
    """
    Mapping for inspect API response.
    Expects result to be a dict with:
      - "is_blocked": a boolean indicating if the prompt or response sent to AI Defense should be blocked.

    Returns:
        bool: True if the text should be blocked, False otherwise.
    """
    # The fail_open behavior is handled in the main function but default to fail closed here if
    # result is None or the is_blocked key is missing somehow
    if result is None:
        return True  # Fail closed: block content if result is None
    return result.get("is_blocked", True)


@action(is_system_action=True, output_mapping=is_ai_defense_text_blocked)
async def ai_defense_inspect(
    config: RailsConfig,
    user_prompt: Optional[str] = None,
    bot_response: Optional[str] = None,
    **kwargs,
):
    # Get configuration with defaults
    ai_defense_config = getattr(config.rails.config, "ai_defense", None)
    timeout = ai_defense_config.timeout if ai_defense_config else DEFAULT_TIMEOUT
    fail_open = ai_defense_config.fail_open if ai_defense_config else False

    api_key = os.environ.get("AI_DEFENSE_API_KEY")
    if not api_key:
        msg = "AI_DEFENSE_API_KEY environment variable not set."
        log.error(msg)
        raise ValueError(msg)

    api_endpoint = os.environ.get("AI_DEFENSE_API_ENDPOINT")
    if not api_endpoint:
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
        msg = "Either user_prompt or bot_response must be provided."
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
            resp = await client.post(
                api_endpoint, headers=headers, json=payload, timeout=timeout
            )
            resp.raise_for_status()
            data = resp.json()
        except (httpx.HTTPStatusError, httpx.TimeoutException, httpx.RequestError) as e:
            msg = f"Error calling AI Defense API: {e}"
            log.error(msg)
            if fail_open:
                # Fail open: allow content when API call fails
                log.warning(
                    "AI Defense API call failed, but fail_open=True, allowing content."
                )
                result: Dict[str, Any] = {
                    "is_blocked": False,
                }
                return result
            else:
                # Fail closed: block content when API call fails
                log.warning(
                    "AI Defense API call failed, fail_open=False, blocking content."
                )
                result: Dict[str, Any] = {
                    "is_blocked": True,
                }
                return result

        # Compose a consistent return structure for flows
        # Handle malformed responses based on fail_open setting
        if "is_safe" not in data:
            # Malformed response - respect fail_open setting
            if fail_open:
                log.warning(
                    "AI Defense API returned malformed response (missing 'is_safe'), but fail_open=True, allowing content."
                )
                is_blocked = False
            else:
                log.warning(
                    "AI Defense API returned malformed response (missing 'is_safe'), fail_open=False, blocking content."
                )
                is_blocked = True
        else:
            is_blocked = not bool(data.get("is_safe", False))

        rules = data.get("rules") or []
        if is_blocked and rules:
            entries = [
                f"{r.get('rule_name')} ({r.get('classification')})"
                for r in rules
                if isinstance(r, dict)
            ]
            if entries:
                log.debug("AI Defense matched rules: %s", ", ".join(entries))

        # Return structure for flows
        result: Dict[str, Any] = {
            "is_blocked": is_blocked,
        }

        return result
