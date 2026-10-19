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

"""Request models for the agent hooks the hooks server checks, one interface per agent path."""

from enum import Enum
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic_core import PydanticCustomError

from nemoguardrails.guardrails.guardrails_types import quoted_identity

HookAgent = Literal["claude-code", "codex", "nemo-relay"]


class HookErrorCode(str, Enum):
    """The `error.code` of a hooks request that ends in an error rather than an allow or a deny."""

    UNSUPPORTED_HOOK_EVENT = "unsupported_hook_event"
    RAIL_FAILED = "rail_failed"
    QUEUE_FULL = "queue_full"


class _ClaudeCodeEffort(BaseModel):
    """The effort level Claude Code reports, when the model supports one."""

    model_config = ConfigDict(extra="allow")

    level: Optional[str] = None


class _ClaudeCodeMcpServer(BaseModel):
    """The MCP server Claude Code reports for an MCP tool call."""

    model_config = ConfigDict(extra="allow")

    name: Optional[str] = None
    source: Optional[str] = None


class ClaudeCodePreToolUseRequest(BaseModel):
    """Claude Code's `PreToolUse` hook input, requiring what the Agent SDK's `PreToolUseHookInput` requires."""

    model_config = ConfigDict(extra="allow")

    # Mandatory Codex-only fields. Declared first, so a Codex payload sent to this path reports them as its first error.
    # Excluded from dumps, since a dumped `None` would be rejected when the dump is validated again.
    # Codex-only Common input field (https://learn.chatgpt.com/docs/hooks#common-input-fields)
    model: Any = Field(default=None, exclude=True)
    # Codex-only PreToolUse field (https://learn.chatgpt.com/docs/hooks#pretooluse)
    turn_id: Any = Field(default=None, exclude=True)

    # Claude Code Common input fields (https://code.claude.com/docs/en/hooks#common-input-fields)
    session_id: str
    prompt_id: Optional[str] = None
    transcript_path: str
    cwd: str
    scratchpad_dir: Optional[str] = None
    permission_mode: Optional[str] = None
    effort: Optional[_ClaudeCodeEffort] = None
    hook_event_name: Literal["PreToolUse"]
    agent_id: Optional[str] = None
    agent_type: Optional[str] = None

    # Claude Code PreToolUse input fields (https://code.claude.com/docs/en/hooks#pretooluse-input)
    tool_name: str = Field(min_length=1)
    tool_input: Any
    tool_use_id: str
    # MCP-tool only
    mcp_server: Optional[_ClaudeCodeMcpServer] = None

    @field_validator("model", "turn_id", mode="before")
    @classmethod
    def _reject_codex_fields(cls, value: Any) -> Any:
        """Reject a field only Codex sends, which marks a Codex payload sent to this path."""
        raise PydanticCustomError("foreign_hook_field", "Codex field not accepted on /v1/hooks/claude-code")


class CodexPreToolUseRequest(BaseModel):
    """Codex's `PreToolUse` hook input, requiring exactly what Codex's generated input schema requires."""

    model_config = ConfigDict(extra="allow")

    # Codex common input fields: https://learn.chatgpt.com/docs/hooks#common-input-fields
    session_id: str
    transcript_path: Optional[str]
    cwd: str
    hook_event_name: Literal["PreToolUse"]
    model: str  # Declared after the fields Claude Code also sends, so a Claude Code payload fails first here.
    permission_mode: str

    # Codex PreToolUse fields: https://learn.chatgpt.com/docs/hooks#pretooluse
    turn_id: str
    tool_name: str = Field(min_length=1)
    tool_use_id: str
    tool_input: Any

    # Optional, found in Codex Github:
    # https://github.com/openai/codex/blob/e444aa99d76beeb8b5b0610b793c0cdefe58da58/codex-rs/hooks/src/schema.rs#L282-L285
    agent_id: Optional[str] = None
    agent_type: Optional[str] = None

    # Optional Claude-Code fields rejected by validator, and excluded from dumps for the same reason as Claude Code's.
    prompt_id: Any = Field(default=None, exclude=True)
    scratchpad_dir: Any = Field(default=None, exclude=True)
    effort: Any = Field(default=None, exclude=True)
    mcp_server: Any = Field(default=None, exclude=True)

    @field_validator("prompt_id", "scratchpad_dir", "effort", "mcp_server", mode="before")
    @classmethod
    def _reject_claude_code_fields(cls, value: Any) -> Any:
        """Reject a field only Claude Code sends, which marks a Claude Code payload sent to this path."""
        raise PydanticCustomError("foreign_hook_field", "Claude Code field not accepted on /v1/hooks/codex")


class NemoRelayToolConditionalExecutionGuardrailRequest(BaseModel):
    """NeMo Relay tool conditional-execution guardrail callback receives: the tool name and its arguments."""

    model_config = ConfigDict(extra="forbid")

    hook_event_name: Literal["NemoRelayToolConditionalExecutionGuardrail"]
    tool_name: str = Field(min_length=1)
    tool_input: Any


HookRequest = Union[
    ClaudeCodePreToolUseRequest,
    CodexPreToolUseRequest,
    NemoRelayToolConditionalExecutionGuardrailRequest,
]

HOOK_HANDLERS: dict[HookAgent, dict[str, type[HookRequest]]] = {
    "claude-code": {"PreToolUse": ClaudeCodePreToolUseRequest},
    "codex": {"PreToolUse": CodexPreToolUseRequest},
    "nemo-relay": {"NemoRelayToolConditionalExecutionGuardrail": NemoRelayToolConditionalExecutionGuardrailRequest},
}


class UnsupportedHookEventError(Exception):
    """A `hook_event_name` that the agent's path doesn't handle."""

    def __init__(self, agent: HookAgent, event: str) -> None:
        self.agent = agent
        self.event = event
        handled = ", ".join(HOOK_HANDLERS[agent])
        super().__init__(
            f"hook_event_name {quoted_identity(event)!r} is not handled on /v1/hooks/{agent}; send {handled}"
        )


# Not underscored: pydantic quotes the class name in the error for a body that isn't an object.
class HookBody(BaseModel):
    """The field every hook body carries, read before the body is validated as its event."""

    hook_event_name: str


def parse_hook_request(agent: HookAgent, body: object) -> HookRequest:
    """Validate *body* as the event it names, on *agent*'s path; error locations are relative to the body."""
    event = HookBody.model_validate(body).hook_event_name
    request_model = HOOK_HANDLERS[agent].get(event)
    if request_model is None:
        raise UnsupportedHookEventError(agent, event)
    return request_model.model_validate(body)
