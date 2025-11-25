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

import logging
from typing import Literal, Optional

import httpx
from pydantic import BaseModel, Field, model_validator
from pydantic import field_validator as validator
from pydantic_core import to_json
from typing_extensions import cast

from nemoguardrails.actions import action
from nemoguardrails.rails.llm.config import RailsConfig, TrendMicroRailConfig

log = logging.getLogger(__name__)


class Guard(BaseModel):
    """
    Represents a guard entity with a single string attribute.

    Attributes:
        guard (str): The input text for guard analysis.
    """

    guard: str


class GuardResult(BaseModel):
    """
    Represents the result of a guard analysis, specifying the action to take and the reason.

    Attributes:
        action (Literal["Block", "Allow"]): The action to take based on guard analysis.
        Must be either "Block" or "Allow".
        reason (str): Explanation for the chosen action. Must be a non-empty string.
    """

    action: Literal["Block", "Allow"] = Field(..., description="Action to take based on guard analysis")
    reason: str = Field(..., min_length=1, description="Explanation for the action")
    blocked: bool = Field(default=False, description="True if action is 'Block', else False")

    @validator("action")
    def validate_action(cls, v):
        log.error(f"Validating action: {v}")
        if v not in ["Block", "Allow"]:
            return "Allow"
        return v

    @model_validator(mode="before")
    def set_blocked(cls, values):
        a = values.get("action")
        values["blocked"] = a.lower() == "block"
        return values


def get_config(config: RailsConfig) -> TrendMicroRailConfig:
    """
    Retrieves the TrendMicroRailConfig from the provided RailsConfig object.

    Args:
        config (RailsConfig): The Rails configuration object containing possible
        Trend Micro settings.

    Returns:
        TrendMicroRailConfig: The Trend Micro configuration, either from the provided
        config or a default instance.
    """
    if not hasattr(config.rails.config, "trend_micro") or config.rails.config.trend_micro is None:
        return TrendMicroRailConfig()

    return cast(TrendMicroRailConfig, config.rails.config.trend_micro)


def trend_ai_guard_mapping(result: GuardResult) -> bool:
    """Convert Trend Micro result to boolean for flow logic."""
    return result.action.lower() == "block"


@action(is_system_action=True, output_mapping=trend_ai_guard_mapping)
async def trend_ai_guard(config: RailsConfig, text: Optional[str] = None):
    """
    Custom action to invoke the Trend Ai Guard
    """

    trend_config = get_config(config)

    # No checks required since default is set in TrendMicroRailConfig
    v1_url = trend_config.v1_url

    v1_api_key = trend_config.get_api_key()
    if not v1_api_key:
        log.error("Trend Micro Vision One API Key not found")
        return GuardResult(
            action="Block",
            reason="Trend Micro Vision One API Key not found",
        )

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
        except httpx.HTTPStatusError as e:
            log.error("Error calling Trend Micro AI Guard API: %s", e)
            return GuardResult(
                action="Allow",
                reason="An error occurred while calling the Trend Micro AI Guard API.",
            )
        return guard_result
