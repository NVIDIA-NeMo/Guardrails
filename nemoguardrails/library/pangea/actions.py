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
import os
from collections.abc import Mapping
from typing import Any, Optional

import httpx
from pydantic import BaseModel
from pydantic_core import to_json
from typing_extensions import Literal, cast

from nemoguardrails.actions import action
from nemoguardrails.rails.llm.config import PangeaRailConfig, RailsConfig

log = logging.getLogger(__name__)


class Message(BaseModel):
    role: str
    content: str


class TextGuardResult(BaseModel):
    prompt_messages: Optional[list[Message]] = None
    """Updated structured prompt, if applicable."""

    blocked: Optional[bool] = None
    """Whether or not the prompt triggered a block detection."""

    transformed: Optional[bool] = None
    """Whether or not the original input was transformed."""

    # Additions.
    bot_message: Optional[str] = None
    user_message: Optional[str] = None


class TextGuardResponse(BaseModel):
    result: TextGuardResult


def get_pangea_config(config: RailsConfig) -> PangeaRailConfig:
    if not hasattr(config.rails.config, "pangea") or config.rails.config.pangea is None:
        return PangeaRailConfig()

    return cast(PangeaRailConfig, config.rails.config.pangea)


@action(is_system_action=True)
async def pangea_ai_guard(
    mode: Literal["input", "output"],
    config: RailsConfig,
    context: Mapping[str, Any] = {},
    user_message: Optional[str] = None,
    bot_message: Optional[str] = None,
) -> TextGuardResult:
    pangea_base_url_template = os.getenv(
        "PANGEA_BASE_URL_TEMPLATE", "https://{SERVICE_NAME}.aws.us.pangea.cloud"
    )
    pangea_api_token = os.getenv("PANGEA_API_TOKEN")

    if not pangea_api_token:
        raise ValueError("PANGEA_API_TOKEN environment variable is not set.")

    pangea_config = get_pangea_config(config)

    user_message = user_message or context.get("user_message")
    bot_message = bot_message or context.get("bot_message")

    if not any([user_message, bot_message]):
        raise ValueError("Either user_message or bot_message must be provided.")

    messages: list[Message] = []
    if config.instructions:
        messages.extend(
            [
                Message(role="system", content=instruction.content)
                for instruction in config.instructions
            ]
        )
    if user_message:
        messages.append(Message(role="user", content=user_message))
    if mode == "output" and bot_message:
        messages.append(Message(role="assistant", content=bot_message))

    recipe = (
        pangea_config.input.recipe
        if mode == "input" and pangea_config.input
        else (
            pangea_config.output.recipe
            if mode == "output" and pangea_config.output
            else None
        )
    )

    async with httpx.AsyncClient(
        base_url=pangea_base_url_template.format(SERVICE_NAME="ai-guard")
    ) as client:
        data = {"messages": messages, "recipe": recipe}
        # Remove `None` values.
        data = {k: v for k, v in data.items() if v is not None}

        response = await client.post(
            "/v1/text/guard",
            content=to_json(data),
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {pangea_api_token}",
                "Content-Type": "application/json",
                "User-Agent": "NeMo Guardrails (https://github.com/NVIDIA/NeMo-Guardrails)",
            },
        )
        try:
            response.raise_for_status()
            text_guard_response = TextGuardResponse(**response.json())
        except Exception as e:
            log.error("Error calling Pangea AI Guard API: %s", e)
            return TextGuardResult(
                prompt_messages=messages,
                blocked=False,
                transformed=False,
                bot_message=bot_message,
                user_message=user_message,
            )

        result = text_guard_response.result
        prompt_messages = result.prompt_messages or []

        result.bot_message = next(
            (m.content for m in prompt_messages if m.role == "assistant"), bot_message
        )
        result.user_message = next(
            (m.content for m in prompt_messages if m.role == "user"), user_message
        )

        return result
