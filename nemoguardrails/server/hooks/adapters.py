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

"""Each agent's hook interface: the event its path handles, the tool call that event asks to check, and its answer."""

from dataclasses import dataclass
from typing import Any, Callable, Literal, Union

from pydantic import BaseModel

from nemoguardrails.guardrails.guardrails_types import quoted_identity
from nemoguardrails.server.hooks.checking import HookOutcome, ToolInvocation
from nemoguardrails.server.hooks.schemas import (
    ClaudeCodePreToolUseRequest,
    CodexPreToolUseRequest,
    HookBody,
    NemoRelayToolConditionalExecutionGuardrailRequest,
)

HookAgent = Literal["claude-code", "codex", "nemo-relay"]

# Claude Code and Codex show the deny reason to the model, so it says who denied the call.
DENY_REASON_PREFIX = "NeMo Guardrails: "


class UnsupportedHookEventError(Exception):
    """A `hook_event_name` that the agent's path doesn't handle."""

    def __init__(self, agent: HookAgent, event: str, handled_event: str) -> None:
        self.agent = agent
        self.event = event
        super().__init__(
            f"hook_event_name {quoted_identity(event)!r} is not handled on /v1/hooks/{agent}; send {handled_event}"
        )


@dataclass(frozen=True, slots=True)
class HookAdapter:
    """One agent's path: the event it handles, the tool call that event's body carries, and each outcome's response."""

    agent: HookAgent
    event: str
    request_model: type[BaseModel]
    invocation: Callable[[Any], ToolInvocation]
    response_body: Callable[[HookOutcome], dict]

    def parse(self, body: object) -> ToolInvocation:
        """The tool call *body* asks to check, as this path's event; error locations are relative to the body."""
        event = HookBody.model_validate(body).hook_event_name
        if event != self.event:
            raise UnsupportedHookEventError(self.agent, event, self.event)
        return self.invocation(self.request_model.model_validate(body))


def _pre_tool_use_invocation(request: Union[ClaudeCodePreToolUseRequest, CodexPreToolUseRequest]) -> ToolInvocation:
    """The tool call a Claude Code or Codex `PreToolUse` hook asks to check."""
    return ToolInvocation(name=request.tool_name, arguments=request.tool_input, call_id=request.tool_use_id)


def _relay_invocation(request: NemoRelayToolConditionalExecutionGuardrailRequest) -> ToolInvocation:
    """The tool call a NeMo Relay guardrail asks to check, with no id: Relay passes its guardrail callback none."""
    return ToolInvocation(name=request.tool_name, arguments=request.tool_input)


def _pre_tool_use_output(outcome: HookOutcome) -> dict:
    """Claude Code's and Codex's `PreToolUse` output: no decision for an allow, the deny shape for a deny."""
    _require_decision(outcome)
    if outcome.kind == "allow":
        # No decision rather than "allow", which would skip Claude Code's permission prompt.
        return {}
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": f"{DENY_REASON_PREFIX}{outcome.reason}",
        }
    }


def _relay_decision(outcome: HookOutcome) -> dict:
    """The verdict the Relay plugin turns into its guardrail callback's `Ok(None)` or `Ok(Some(reason))`."""
    _require_decision(outcome)
    if outcome.kind == "allow":
        return {"decision": "allow"}
    return {"decision": "deny", "reason": outcome.reason}


def _require_decision(outcome: HookOutcome) -> None:
    """Raise for an error outcome, which is answered with an error status, not a `200` body."""
    if outcome.kind == "error":
        raise ValueError("an error outcome is answered with an error status, not a 200 body")


ADAPTERS: dict[HookAgent, HookAdapter] = {
    adapter.agent: adapter
    for adapter in (
        HookAdapter(
            agent="claude-code",
            event="PreToolUse",
            request_model=ClaudeCodePreToolUseRequest,
            invocation=_pre_tool_use_invocation,
            response_body=_pre_tool_use_output,
        ),
        HookAdapter(
            agent="codex",
            event="PreToolUse",
            request_model=CodexPreToolUseRequest,
            invocation=_pre_tool_use_invocation,
            response_body=_pre_tool_use_output,
        ),
        HookAdapter(
            agent="nemo-relay",
            event="NemoRelayToolConditionalExecutionGuardrail",
            request_model=NemoRelayToolConditionalExecutionGuardrailRequest,
            invocation=_relay_invocation,
            response_body=_relay_decision,
        ),
    )
}
