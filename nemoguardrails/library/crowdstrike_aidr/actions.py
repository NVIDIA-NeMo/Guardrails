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

from pydantic import BaseModel, ValidationError
from pydantic_core import to_json
from typing_extensions import Literal, TypedDict

from nemoguardrails.actions import action
from nemoguardrails.actions.rail_outcome import RailOutcome, TransformTarget
from nemoguardrails.http import HTTPClient, HTTPClientError, HTTPStatusError, http_call
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


def _crowdstrike_aidr_outcome(
    result: GuardChatCompletionsResult,
    mode: Literal["input", "output"],
) -> RailOutcome:
    """Convert a CrowdStrike result into a rail decision.

    Blocking takes precedence over transformation. A transformed input rewrites
    the user message, while a transformed output rewrites the bot message.
    """

    metadata = {
        "blocked": bool(result.blocked),
        "transformed": bool(result.transformed),
        "guard_output": result.guard_output,
        "user_message": result.user_message,
        "bot_message": result.bot_message,
    }
    if result.blocked:
        return RailOutcome.block(metadata=metadata)
    if result.transformed:
        target = TransformTarget.USER_MESSAGE if mode == "input" else TransformTarget.BOT_MESSAGE
        text = result.user_message if mode == "input" else result.bot_message
        return RailOutcome.transform(
            [(target, text or "")],
            metadata=metadata,
        )
    return RailOutcome.allow(metadata=metadata)


@action(is_system_action=True)
async def crowdstrike_aidr_guard(
    mode: Literal["input", "output"],
    config: RailsConfig,
    context: Mapping[str, Any] = {},
    user_message: Optional[str] = None,
    bot_message: Optional[str] = None,
    http_client: Optional[HTTPClient] = None,
) -> RailOutcome:
    """Evaluate conversation content with CrowdStrike AI Guard.

    The action evaluates the user message for input rails and includes the bot
    message for output rails. A successful provider response can allow, block,
    or transform the active message.

    Expected provider failures fail open: unsuccessful HTTP status responses,
    transport failures, malformed JSON, and response-schema validation errors
    are logged and return an allow outcome containing the original messages.
    Unexpected programming errors are not swallowed.

    Args:
        mode: Whether the action is evaluating an input or output rail.
        config: Active rails configuration.
        context: Conversation context used when explicit messages are omitted.
        user_message: Optional user message override.
        bot_message: Optional bot message override.
        http_client: Optional caller-owned HTTP client. The action does not
            close an injected client.

    Returns:
        The provider-derived rail outcome, or an allow outcome for an expected
        provider failure.

    Raises:
        ValueError: If the API token or required conversation content is
            missing.
    """

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

    endpoint = base_url_template.format(SERVICE_NAME="aiguard").rstrip("/") + "/v1/guard_chat_completions"
    fallback = GuardChatCompletionsResult(
        guard_output={"messages": messages},
        blocked=False,
        transformed=False,
        bot_message=bot_message,
        user_message=user_message,
    )
    try:
        response = await http_call(
            http_client,
            "POST",
            endpoint,
            content=to_json({"guard_input": {"messages": messages}}),
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {api_token}",
                "Content-Type": "application/json",
                "User-Agent": "NeMo Guardrails (https://github.com/NVIDIA-NeMo/Guardrails)",
            },
            timeout=crowdstrike_aidr_config.timeout,
            raise_for_status=False,
        )
        response.raise_for_status()
        guard_response = GuardChatCompletionsResponse.model_validate(response.json())
    except HTTPStatusError as e:
        log.error("HTTP status error from CrowdStrike AIDR API: %s", e)
    except (HTTPClientError, ValidationError) as e:
        log.error("Error calling CrowdStrike AIDR API: %s", e)
    else:
        result = guard_response.result
        output_messages = result.guard_output.get("messages", []) if result.guard_output else []

        result.bot_message = next((m.content for m in output_messages if m.role == "assistant"), bot_message)
        result.user_message = next((m.content for m in output_messages if m.role == "user"), user_message)

        return _crowdstrike_aidr_outcome(result, mode)

    return _crowdstrike_aidr_outcome(fallback, mode)
