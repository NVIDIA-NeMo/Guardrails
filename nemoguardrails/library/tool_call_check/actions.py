# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import json
import logging
from typing import Optional

from nemoguardrails.actions import action
from nemoguardrails.actions.llm.utils import llm_call
from nemoguardrails.context import llm_call_info_var
from nemoguardrails.logging.explain import LLMCallInfo

log = logging.getLogger(__name__)


def _parse_verdict(text: str) -> bool:
    """Return True (block) if the last non-empty line is BLOCK; False (allow) otherwise.

    Scans from the end of the model response so reasoning text that mentions
    ALLOW or BLOCK mid-response does not influence the verdict.  Fails closed:
    if no clear verdict is found, the call is blocked.
    """
    for line in reversed(text.strip().splitlines()):
        normalized = line.strip().upper()
        if normalized == "BLOCK":
            return True
        if normalized == "ALLOW":
            return False
    return True


@action(name="check_tool_call")
async def check_tool_call(
    check: str,
    context: Optional[dict] = None,
    llm=None,
    llm_task_manager=None,
) -> dict:
    ctx = context or {}
    tool_name = ctx.get("tool_name") or ""

    log.debug("check_tool_call: check=%r tool=%r", check, tool_name)

    prompt = llm_task_manager.render_task_prompt(
        task=check,
        context={
            "tool_call": json.dumps(ctx.get("tool_call") or {}),
            "tool_name": tool_name,
            "tool_message": ctx.get("tool_message") or "",
            "user_message": ctx.get("user_message") or "",
        },
    )

    log.debug(
        "check_tool_call: rendered prompt (%d chars): %.300s",
        len(prompt) if isinstance(prompt, str) else -1,
        prompt,
    )

    llm_call_info_var.set(LLMCallInfo(task=check))

    response = await llm_call(llm, prompt)
    text = getattr(response, "content", None) or str(response)

    log.debug("check_tool_call: raw response: %r", text)

    is_blocked = _parse_verdict(text)
    log.debug("check_tool_call: verdict=%s", "BLOCK" if is_blocked else "ALLOW")

    if is_blocked:
        return {"is_blocked": True, "reason": text}
    return {"is_blocked": False, "reason": None}


@action(name="get_tool_call_name")
async def get_tool_call_name(tool_call: dict) -> Optional[str]:
    if not isinstance(tool_call, dict):
        return None
    name = tool_call.get("function", {}).get("name")
    if name:
        return name
    return tool_call.get("name")


@action(name="get_per_tool_output_flows")
async def get_per_tool_output_flows(tool_name: Optional[str], config=None) -> list:
    if not tool_name or config is None:
        return []
    return list(config.rails.tool_output.per_tool.get(tool_name, []))


@action(name="get_per_tool_input_flows")
async def get_per_tool_input_flows(tool_name: Optional[str], config=None) -> list:
    if not tool_name or config is None:
        return []
    return list(config.rails.tool_input.per_tool.get(tool_name, []))
