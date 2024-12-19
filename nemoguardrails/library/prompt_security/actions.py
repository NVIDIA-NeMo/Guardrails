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
        ret = await client.post(ps_protect_url, headers=headers, json=payload)
        return ret.json()


@action(is_system_action=True)
async def protect_text(source: str, text: str):
    """Protects the given text.

    Args
        source: The source for the text, i.e. "input", "output".
        text: The text to check.

    Returns
        True if text should be blocked, False otherwise.
    """

    ps_protect_url = os.getenv("PS_PROTECT_URL")
    if not ps_protect_url:
        raise ValueError("PS_PROTECT_URL env variable required for Prompt Security.")

    ps_app_id = os.getenv("PS_APP_ID")
    if not ps_app_id:
        raise ValueError("PS_APP_ID env variable required for Prompt Security.")

    if source == "input":
        response = await ps_protect_api_async(ps_protect_url, ps_app_id, text)
    elif source == "output":
        response = await ps_protect_api_async(
            ps_protect_url, ps_app_id, None, None, text
        )
    else:
        raise ValueError(f"The flow, '{source}', is not supported by Prompt Security.")

    return response["result"]["action"] == "block"
