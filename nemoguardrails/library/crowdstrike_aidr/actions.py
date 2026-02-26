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

import logging
import os
from collections.abc import Mapping
from typing import Any, Optional, cast

import httpx
from pydantic import BaseModel
from pydantic_core import to_json
from typing_extensions import Literal, TypedDict

from nemoguardrails.actions import action
from nemoguardrails.rails.llm.config import CrowdStrikeAIDRRailConfig, RailsConfig

log = logging.getLogger(__name__)


class Message(BaseModel):
    role: str
    content: str


class GuardOutput(TypedDict, total=False):
    messages: list[Message]


class GuardChatCompletionsResult(BaseModel):
    guard_output: Optional[GuardOutput] = None
    """Updated structured prompt, if applicable."""

    blocked: Optional[bool] = None
    """Whether or not the prompt triggered a block detection."""

    transformed: Optional[bool] = None
    """Whether or not the original input was transformed."""

    # Additions.
    bot_message: Optional[str] = None
    user_message: Optional[str] = None


class GuardChatCompletionsResponse(BaseModel):
    result: GuardChatCompletionsResult


def get_crowdstrike_aidr_config(config: RailsConfig) -> CrowdStrikeAIDRRailConfig:
    if not hasattr(config.rails.config, "crowdstrike_aidr") or config.rails.config.crowdstrike_aidr is None:
        return CrowdStrikeAIDRRailConfig()

    return cast(CrowdStrikeAIDRRailConfig, config.rails.config.crowdstrike_aidr)


@action(is_system_action=True)
async def crowdstrike_aidr_guard(
    mode: Literal["input", "output"],
    config: RailsConfig,
    context: Mapping[str, Any] = {},
    user_message: Optional[str] = None,
    bot_message: Optional[str] = None,
) -> GuardChatCompletionsResult:
    base_url_template = os.getenv("CS_AIDR_BASE_URL_TEMPLATE", "https://api.crowdstrike.com/aidr/{SERVICE_NAME}")
    api_token = os.getenv("CS_AIDR_TOKEN")

    if not api_token:
        raise ValueError("CS_AIDR_TOKEN environment variable is not set.")

    crowdstrike_aidr_config = get_crowdstrike_aidr_config(config)

    user_message = user_message or context.get("user_message")
    bot_message = bot_message or context.get("bot_message")

    if not any((user_message, bot_message)):
        raise ValueError("Either user_message or bot_message must be provided.")

    messages: list[Message] = []
    if config.instructions:
        messages.extend([Message(role="system", content=instruction.content) for instruction in config.instructions])
    if user_message:
        messages.append(Message(role="user", content=user_message))
    if mode == "output" and bot_message:
        messages.append(Message(role="assistant", content=bot_message))

    async with httpx.AsyncClient(base_url=base_url_template.format(SERVICE_NAME="aiguard")) as client:
        response = await client.post(
            "/v1/guard_chat_completions",
            content=to_json({"guard_input": {"messages": messages}}),
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {api_token}",
                "Content-Type": "application/json",
                "User-Agent": "NeMo Guardrails (https://github.com/NVIDIA-NeMo/Guardrails)",
            },
            timeout=crowdstrike_aidr_config.timeout,
        )
        try:
            response.raise_for_status()
            guard_response = GuardChatCompletionsResponse(**response.json())
        except httpx.HTTPStatusError as e:
            log.error("HTTP status error from CrowdStrike AIDR API: %s", e)
            return GuardChatCompletionsResult(
                guard_output={"messages": messages},
                blocked=False,
                transformed=False,
                bot_message=bot_message,
                user_message=user_message,
            )
        except Exception as e:
            log.error("Error calling CrowdStrike AIDR API: %s", e)
            return GuardChatCompletionsResult(
                guard_output={"messages": messages},
                blocked=False,
                transformed=False,
                bot_message=bot_message,
                user_message=user_message,
            )

        result = guard_response.result
        output_messages = result.guard_output.get("messages", []) if result.guard_output else []

        result.bot_message = next((m.content for m in output_messages if m.role == "assistant"), bot_message)
        result.user_message = next((m.content for m in output_messages if m.role == "user"), user_message)

        return result
