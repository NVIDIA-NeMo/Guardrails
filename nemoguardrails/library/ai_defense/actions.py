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

"""Prompt/Response protection using Cisco AI Defense."""

import logging
import os
from collections.abc import Mapping
from typing import Any, Optional

from nemoguardrails import RailsConfig
from nemoguardrails.actions import action
from nemoguardrails.actions.rail_outcome import RailOutcome
from nemoguardrails.http import (
    HTTPClient,
    HTTPClientError,
    HTTPResponseDecodeError,
    http_call,
)

log = logging.getLogger(__name__)

# Default timeout for AI Defense API calls in seconds
DEFAULT_TIMEOUT = 30.0


def _ai_defense_outcome(is_blocked: bool) -> RailOutcome:
    if is_blocked:
        return RailOutcome.block(metadata={"is_blocked": is_blocked})
    return RailOutcome.allow(metadata={"is_blocked": is_blocked})


def _ai_defense_failure_outcome(fail_open: bool, failure: str) -> RailOutcome:
    if fail_open:
        log.warning("%s, but fail_open=True, allowing content.", failure)
        return _ai_defense_outcome(False)
    log.warning("%s, fail_open=False, blocking content.", failure)
    return _ai_defense_outcome(True)


@action(is_system_action=True)
async def ai_defense_inspect(
    config: RailsConfig,
    user_prompt: Optional[str] = None,
    bot_response: Optional[str] = None,
    http_client: Optional[HTTPClient] = None,
    **kwargs,
) -> RailOutcome:
    """Inspect a user prompt or bot response with Cisco AI Defense.

    A safe response is allowed and an unsafe response is blocked. Transport
    failures and malformed responses follow the configured ``fail_open``
    policy, which defaults to fail closed.

    Args:
        config: Rails configuration containing the AI Defense settings.
        user_prompt: User content to inspect.
        bot_response: Bot content to inspect. Takes precedence over
            ``user_prompt`` when both are provided.
        http_client: Optional caller-owned HTTP client.
        **kwargs: Additional action parameters. The ``user`` value, when
            present, is forwarded as request metadata.

    Returns:
        The allow or block outcome from AI Defense or the failure policy.

    Raises:
        ValueError: If the required environment variables or content are
            missing.
    """
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

    payload: dict[str, Any] = {"messages": messages}
    if metadata:
        payload["metadata"] = metadata

    try:
        response = await http_call(
            http_client,
            "POST",
            api_endpoint,
            headers=headers,
            json=payload,
            timeout=timeout,
        )
        data = response.json()
    except HTTPResponseDecodeError as e:
        log.error("AI Defense API returned malformed JSON: %s", e)
        return _ai_defense_failure_outcome(fail_open, "AI Defense API returned malformed JSON")
    except HTTPClientError as e:
        log.error("Error calling AI Defense API: %s", e)
        return _ai_defense_failure_outcome(fail_open, "AI Defense API call failed")

    if not isinstance(data, Mapping):
        return _ai_defense_failure_outcome(
            fail_open,
            "AI Defense API returned malformed response (expected an object)",
        )

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
        entries = [f"{r.get('rule_name')} ({r.get('classification')})" for r in rules if isinstance(r, dict)]
        if entries:
            log.debug("AI Defense matched rules: %s", ", ".join(entries))

    return _ai_defense_outcome(is_blocked)
