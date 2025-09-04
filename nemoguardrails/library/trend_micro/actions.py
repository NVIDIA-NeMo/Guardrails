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
from typing import Optional

import httpx
from pydantic import BaseModel
from pydantic_core import to_json
from typing_extensions import cast

from nemoguardrails.actions import action
from nemoguardrails.rails.llm.config import RailsConfig, TrendMicroRailConfig

log = logging.getLogger(__name__)


class Guard(BaseModel):
    guard: str


class GuardResult(BaseModel):
    action: str
    reason: str


def get_config(config: RailsConfig) -> TrendMicroRailConfig:
    if (
        not hasattr(config.rails.config, "trend_micro")
        or config.rails.config.trend_micro is None
    ):
        return TrendMicroRailConfig()

    return cast(TrendMicroRailConfig, config.rails.config.trend_micro)


@action(is_system_action=True)
async def trend_ai_guard(config: RailsConfig, text: Optional[str] = None):
    """
    Custom action to invoke the Trend Ai Guard
    """

    trend_config = get_config(config)

    # No checks required since default is set in TrendMicroRailConfig
    v1_url = trend_config.v1_url

    v1_api_key = trend_config.get_api_key()
    if not v1_api_key:
        raise ValueError("Trend Micro Vision One API Key not found")

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
