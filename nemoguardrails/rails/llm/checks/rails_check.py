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
from typing import Any, List, Optional, Protocol

from nemoguardrails.colang.v2_x.runtime.flows import FlowStatus, State
from nemoguardrails.colang.v2_x.runtime.statemachine import InternalEvent, initialize_state
from nemoguardrails.rails.llm.options import (
    GenerationResponse,
    RailsResult,
    RailStatus,
    RailType,
)
from nemoguardrails.utils import new_readable_uuid

log = logging.getLogger(__name__)

__all__ = ["RailsCheckRuntime", "check_messages"]


class RailsCheckRuntime(Protocol):
    config: Any
    runtime: Any

    async def generate_async(
        self,
        *,
        messages: List[dict],
        options: dict,
    ) -> object: ...


@dataclass
class RailsCheckPlan:
    messages: List[dict]
    options: dict
    original_content: str


@dataclass
class Colang2RailFlowCheck:
    content: str
    stopped: bool
    rail: Optional[str]
    state: State


async def check_messages(
    rails: RailsCheckRuntime,
    messages: List[dict],
    rail_types: Optional[List[RailType]] = None,
) -> RailsResult:
    """Run input/output rails for a message list and return rail check status."""
    colang_version = rails.config.colang_version
    plan = _plan_rails_check(
        messages,
        rail_types,
        include_activated_rails_log=colang_version != "2.x",
    )
    if plan is None:
        last_content = messages[-1].get("content", "") if messages else ""
        return RailsResult(status=RailStatus.PASSED, content=last_content)

    if colang_version == "2.x":
        return await _check_colang_2_messages(rails, plan)

    response = await rails.generate_async(messages=plan.messages, options=plan.options)

    if not isinstance(response, GenerationResponse):
        raise RuntimeError(f"Expected GenerationResponse, got {type(response).__name__}")

    return _classify_rails_response(response, plan.original_content)


def _plan_rails_check(
    messages: List[dict],
    rail_types: Optional[List[RailType]] = None,
    *,
    include_activated_rails_log: bool = True,
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
    if include_activated_rails_log:
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


async def _check_colang_2_messages(
    rails: RailsCheckRuntime,
    plan: RailsCheckPlan,
) -> RailsResult:
    rails_to_run = plan.options["rails"]
    result_content = plan.original_content
    state = _new_colang_2_check_state(rails)

    if "input" in rails_to_run:
        input_check = await _run_colang_2_rail_flow(
            rails=rails,
            state=state,
            flow_id="input rails",
            content=_get_last_content_by_role(plan.messages, "user"),
            content_context_key="user_message",
        )
        state = input_check.state
        if input_check.stopped:
            return RailsResult(
                status=RailStatus.BLOCKED,
                content=input_check.content,
                rail=input_check.rail,
            )
        if rails_to_run == ["input"]:
            result_content = input_check.content

    if "output" in rails_to_run:
        output_check = await _run_colang_2_rail_flow(
            rails=rails,
            state=state,
            flow_id="output rails",
            content=_get_last_content_by_role(plan.messages, "assistant"),
            content_context_key="bot_message",
        )
        if output_check.stopped:
            return RailsResult(
                status=RailStatus.BLOCKED,
                content=output_check.content,
                rail=output_check.rail,
            )
        result_content = output_check.content

    if result_content != plan.original_content:
        return RailsResult(status=RailStatus.MODIFIED, content=result_content)
    return RailsResult(status=RailStatus.PASSED, content=result_content)


def _new_colang_2_check_state(rails: RailsCheckRuntime) -> State:
    state = State(
        flow_states={},
        flow_configs=rails.runtime.flow_configs,
        rails_config=rails.runtime.config,
    )
    initialize_state(state)
    assert state.main_flow_state is not None

    # check_async starts rail flows directly; it must not run the config's main
    # flow just to obtain a parent flow for the rail-flow state.
    state.main_flow_state.status = FlowStatus.STARTED
    return state


async def _run_colang_2_rail_flow(
    *,
    rails: RailsCheckRuntime,
    state: State,
    flow_id: str,
    content: str,
    content_context_key: str,
) -> Colang2RailFlowCheck:
    if flow_id not in state.flow_configs:
        return Colang2RailFlowCheck(
            content=content,
            stopped=False,
            rail=None,
            state=state,
        )

    main_flow = state.main_flow_state
    assert main_flow is not None

    flow_instance_uid = new_readable_uuid(flow_id)
    start_event = InternalEvent(
        name="StartFlow",
        arguments={
            "flow_id": flow_id,
            "flow_instance_uid": flow_instance_uid,
            "flow_hierarchy_position": "0.1",
            "source_flow_instance_uid": main_flow.uid,
            "source_head_uid": list(main_flow.heads.values())[0].uid,
            "$0": content,
        },
    )
    output_events, state = await rails.runtime.process_events(
        [start_event],
        state=state,
        blocking=True,
        instant_actions=_colang_2_instant_actions(rails),
    )
    flow_state = state.flow_states[flow_instance_uid]
    stopped = flow_state.status == FlowStatus.STOPPED

    return Colang2RailFlowCheck(
        content=_colang_2_checked_content(
            state=state,
            output_events=output_events,
            original_content=content,
            content_context_key=content_context_key,
        ),
        stopped=stopped,
        # Colang 2.x check runs the wrapper rail flow directly and does not have
        # v1-style activated-rail logs, so blocked results can only name the
        # rail category wrapper here.
        rail=flow_id if stopped else None,
        state=state,
    )


def _colang_2_instant_actions(rails: RailsCheckRuntime) -> List[str]:
    if rails.config.rails.actions.instant_actions is not None:
        return rails.config.rails.actions.instant_actions
    return ["UtteranceBotAction"]


def _colang_2_checked_content(
    *,
    state: State,
    output_events: List[dict],
    original_content: str,
    content_context_key: str,
) -> str:
    checked_content = state.context.get(content_context_key)
    if checked_content is not None:
        return checked_content

    bot_message = state.context.get("bot_message")
    if bot_message is not None:
        return bot_message

    bot_action_content = _last_colang_2_bot_action_content(output_events)
    if bot_action_content is not None:
        return bot_action_content

    return original_content


def _last_colang_2_bot_action_content(output_events: List[dict]) -> Optional[str]:
    for event in reversed(output_events):
        if event.get("type") == "UtteranceBotActionFinished":
            return event.get("final_script", "")
        if event.get("type") == "StartUtteranceBotAction":
            return event.get("script", "")
    return None


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
