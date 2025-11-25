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

"""Prompt/Response protection using Prompt Security."""

import logging
import os
from typing import Optional

import httpx

from nemoguardrails.actions import action

log = logging.getLogger(__name__)


async def ps_protect_api_async(
    ps_protect_url: str,
    ps_app_id: str,
    prompt: Optional[str] = None,
    system_prompt: Optional[str] = None,
    response: Optional[str] = None,
    user: Optional[str] = None,
):
    """Calls Prompt Security Protect API asynchronously.

    Args:
        ps_protect_url: the URL of the protect endpoint given by Prompt Security.
        URL is https://[REGION].prompt.security/api/protect where REGION is eu, useast or apac

        ps_app_id: the application ID given by Prompt Security (similar to an API key).
        Get it from the admin portal at https://[REGION].prompt.security/ where REGION is eu, useast or apac

        prompt: the user message to protect.

        system_prompt: the system message for context.

        response: the bot message to protect.

        user: the user ID or username for context.

    Returns:
        A dictionary with the following items:
        - is_blocked: True if the text should be blocked, False otherwise.
        - is_modified: True if the text should be modified, False otherwise.
        - modified_text: The modified text if is_modified is True, None otherwise.
    """

    headers = {
        "APP-ID": ps_app_id,
        "Content-Type": "application/json",
    }
    payload = {
        "prompt": prompt,
        "system_prompt": system_prompt,
        "response": response,
        "user": user,
    }
    async with httpx.AsyncClient() as client:
        modified_text = None
        ps_action = "log"
        try:
            ret = await client.post(ps_protect_url, headers=headers, json=payload)
            res = ret.json()
            ps_action = res.get("result", {}).get("action", "log")
            if ps_action == "modify":
                key = "response" if response else "prompt"
                modified_text = res.get("result", {}).get(key, {}).get("modified_text")
        except Exception as e:
            log.error("Error calling Prompt Security Protect API: %s", e)
        return {
            "is_blocked": ps_action == "block",
            "is_modified": ps_action == "modify",
            "modified_text": modified_text,
        }


def protect_text_mapping(result: dict) -> bool:
    """
    Mapping for protect_text action.

    Expects result to be a dict with:
      - "is_blocked": a boolean indicating if the response passed to prompt security should be blocked.

    Returns:
        True if the response should be blocked (i.e. if "is_blocked" is True),
        False otherwise.
    """
    blocked = result.get("is_blocked", True)
    return blocked


@action(is_system_action=True, output_mapping=protect_text_mapping)
async def protect_text(user_prompt: Optional[str] = None, bot_response: Optional[str] = None, **kwargs):
    """Protects the given user_prompt or bot_response.
    Args:
        user_prompt: The user message to protect.
        bot_response: The bot message to protect.
    Returns:
        A dictionary with the following items:
        - is_blocked: True if the text should be blocked, False otherwise.
        - is_modified: True if the text should be modified, False otherwise.
        - modified_text: The modified text if is_modified is True, None otherwise.
    Raises:
        ValueError is returned in one of the following cases:
        1. If PS_PROTECT_URL env variable is not set.
        2. If PS_APP_ID env variable is not set.
        3. If no user_prompt and no bot_response is provided.
    """

    ps_protect_url = os.getenv("PS_PROTECT_URL")
    if not ps_protect_url:
        raise ValueError("PS_PROTECT_URL env variable is required for Prompt Security.")

    ps_app_id = os.getenv("PS_APP_ID")
    if not ps_app_id:
        raise ValueError("PS_APP_ID env variable is required for Prompt Security.")

    if bot_response:
        return await ps_protect_api_async(ps_protect_url, ps_app_id, None, None, bot_response)

    if user_prompt:
        return await ps_protect_api_async(ps_protect_url, ps_app_id, user_prompt)

    raise ValueError("Neither user_message nor bot_message was provided")
