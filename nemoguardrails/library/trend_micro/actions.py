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

import logging
import os
from typing import Optional

import httpx
from pydantic import BaseModel
from pydantic_core import to_json

from nemoguardrails.actions import action

log = logging.getLogger(__name__)


class Guard(BaseModel):
    guard: str


class GuardResult(BaseModel):
    action: str
    reason: str


@action(is_system_action=True)
async def trend_ai_guard(text: Optional[str] = None):
    """
    Custom action to invoke the Trend Ai Guard
    """
    v1_url = os.environ.get(
        "V1_URL", "https://api.xdr.trendmicro.com/beta/aiSecurity/guard"
    )
    v1_api_key = os.environ.get("V1_API_KEY")
    if not v1_api_key:
        raise ValueError("V1_API_KEY environment variable is not set.")

    if text is None:
        raise ValueError("No prompt/response found in the last event.")

    async with httpx.AsyncClient() as client:
        data = Guard(guard=text).model_dump()

        response = await client.post(
            v1_url,
            content=to_json(data),
            headers={
                "Authorization": f"Bearer {v1_api_key}",
                "Content-Type": "application/json",
            },
        )

        try:
            response.raise_for_status()
            guard_result = GuardResult(**response.json())
            log.debug("Trend Micro AI Guard Result: %s", guard_result)
        except Exception as e:
            log.error("Error calling Trend Micro AI Guard API: %s", e)
            return GuardResult(
                action="allow",
                reason="An error occurred while calling the Trend Micro AI Guard API.",
            )
        return guard_result
