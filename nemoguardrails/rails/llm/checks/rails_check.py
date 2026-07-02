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

"""Message rail check helpers."""

import logging
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Optional

from nemoguardrails.rails.llm.options import (
    GenerationResponse,
    RailsResult,
    RailStatus,
    RailType,
)
from nemoguardrails.rails.llm.types import RailsCheckSurface

log = logging.getLogger(__name__)

__all__ = ["RailsCheckSurface", "check_messages"]


@dataclass
class RailsCheckPlan:
    messages: List[dict]
    options: dict
    original_content: str


async def check_messages(
    rails: RailsCheckSurface,
    messages: List[dict],
    rail_types: Optional[List[RailType]] = None,
) -> RailsResult:
    """Run input/output rails for a message list and return rail check status."""
    plan = _plan_rails_check(messages, rail_types)
    if plan is None:
        last_content = messages[-1].get("content", "") if messages else ""
        return RailsResult(status=RailStatus.PASSED, content=last_content)

    response = await rails.generate_async(messages=plan.messages, options=plan.options)

    if not isinstance(response, GenerationResponse):
        raise RuntimeError(f"Expected GenerationResponse, got {type(response).__name__}")

    return _classify_rails_response(response, plan.original_content)


def _plan_rails_check(
    messages: List[dict],
    rail_types: Optional[List[RailType]] = None,
) -> Optional[RailsCheckPlan]:
    if rail_types is not None:
        options: Optional[dict] = {"rails": [r.value for r in rail_types]}
    else:
        options = _determine_rails_from_messages(messages)

    if options is None:
        return None

    rails_to_run = options["rails"]
    if "output" in rails_to_run:
        original_content = _get_last_content_by_role(messages, "assistant")
    else:
        original_content = _get_last_content_by_role(messages, "user")

    messages = _normalize_messages_for_rails(deepcopy(messages), rails_to_run)
    options["log"] = {"activated_rails": True}

    return RailsCheckPlan(
        messages=messages,
        options=options,
        original_content=original_content,
    )


def _classify_rails_response(
    response: GenerationResponse,
    original_content: str,
) -> RailsResult:
    blocking_rail = _get_blocking_rail(response)
    result_content = _get_last_response_content(response)

    if blocking_rail:
        return RailsResult(status=RailStatus.BLOCKED, content=result_content, rail=blocking_rail)

    if result_content != original_content:
        return RailsResult(status=RailStatus.MODIFIED, content=result_content)
    return RailsResult(status=RailStatus.PASSED, content=result_content)


def _determine_rails_from_messages(messages: List[dict]) -> Optional[dict]:
    roles = {msg.get("role") for msg in reversed(messages)}
    has_user = "user" in roles
    has_assistant = "assistant" in roles

    if not has_user and not has_assistant:
        log.warning(
            "check() called with no user or assistant messages. "
            "Only system, context, or tool messages found. "
            "Returning passing result without running rails."
        )
        return None

    if has_user and has_assistant:
        return {"rails": ["input", "output"]}
    if has_user:
        return {"rails": ["input"]}
    return {"rails": ["output"]}


def _normalize_messages_for_rails(
    messages: List[dict],
    rails: List[str],
) -> List[dict]:
    if rails == ["output"]:
        has_user = any(msg.get("role") == "user" for msg in messages)
        if not has_user:
            return [{"role": "user", "content": ""}] + messages

    return messages


def _get_last_content_by_role(messages: List[dict], role: str) -> str:
    for msg in reversed(messages):
        if msg.get("role") == role:
            return msg.get("content", "")
    return ""


def _get_blocking_rail(response: GenerationResponse) -> Optional[str]:
    if response.log and response.log.activated_rails:
        for rail in response.log.activated_rails:
            if rail.stop:
                return rail.name
    return None


def _get_last_response_content(response: GenerationResponse) -> str:
    if isinstance(response.response, list) and response.response:
        return response.response[-1].get("content", "")
    if isinstance(response.response, str):
        return response.response
    return ""
