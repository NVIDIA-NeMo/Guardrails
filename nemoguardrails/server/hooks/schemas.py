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
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, StrictBool, model_validator


class HookErrorCode(str, Enum):
    """The `error.code` of a hooks request that ends in an error rather than an allow or a deny."""

    UNSUPPORTED_HOOK_EVENT = "unsupported_hook_event"
    RAIL_FAILED = "rail_failed"
    QUEUE_FULL = "queue_full"
    CHECK_NOT_CONFIGURED = "check_not_configured"
    NOT_QUARANTINED = "not_quarantined"
    QUARANTINE_STORE_FAILED = "quarantine_store_failed"
    DECISION_RECORD_FAILED = "decision_record_failed"


class ClaudeCodePreToolUseRequest(BaseModel):
    """Claude Code's `PreToolUse` hook input, requiring only the fields the server reads."""

    # Undeclared fields are kept as extras, unchecked, so an agent release can add or change them.
    model_config = ConfigDict(extra="allow")

    # Claude Code Common input fields (https://code.claude.com/docs/en/hooks#common-input-fields)
    hook_event_name: Literal["PreToolUse"]

    # Claude Code PreToolUse input fields (https://code.claude.com/docs/en/hooks#pretooluse-input)
    tool_name: str = Field(min_length=1)
    tool_input: Any
    tool_use_id: str


class CodexPreToolUseRequest(BaseModel):
    """Codex's `PreToolUse` hook input, requiring only the fields the server reads."""

    # Undeclared fields are kept as extras, unchecked, so an agent release can add or change them.
    model_config = ConfigDict(extra="allow")

    # Codex common input fields: https://learn.chatgpt.com/docs/hooks#common-input-fields
    hook_event_name: Literal["PreToolUse"]

    # Codex PreToolUse fields: https://learn.chatgpt.com/docs/hooks#pretooluse
    tool_name: str = Field(min_length=1)
    tool_use_id: str
    tool_input: Any


class NemoRelayToolConditionalExecutionGuardrailRequest(BaseModel):
    """NeMo Relay tool conditional-execution guardrail callback receives: the tool name and its arguments."""

    model_config = ConfigDict(extra="forbid")

    hook_event_name: Literal["NemoRelayToolConditionalExecutionGuardrail"]
    tool_name: str = Field(min_length=1)
    tool_input: Any


class AgentPart(BaseModel):
    """One part an agent starts with, as plain facts the caller gathered; the server reads no package format."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, description="What a deny reason names the part by, such as its path in the package.")
    type: Literal["skill", "plugin", "connector", "hook", "package", "policy", "agent"] = Field(
        description="The kind of part; `agent` is an agent this agent calls."
    )
    source: str = Field(min_length=1, description="Where the part came from, as the caller states it.")
    version: Optional[str] = None
    digest: Optional[str] = Field(default=None, description="A digest of the part's bytes, such as `sha256:...`.")
    # Strict, so a text such as "no" is refused rather than read as a boolean.
    local: StrictBool = Field(description="Whether the source is a folder on the machine that starts the agent.")


class AgentPolicy(BaseModel):
    """A policy the agent brings, and the enforcers that must apply it where the agent runs."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    required: StrictBool = True
    enforcers: list[str] = Field(description="The enforcers the policy needs, such as `openshell`.")


class AgentStartRequest(BaseModel):
    """The facts about an agent that is about to start: its parts, the models it names, and its policies."""

    model_config = ConfigDict(extra="forbid")

    hook_event_name: Literal["AgentStart"]
    agent_name: str = Field(min_length=1)
    agent_digest: Optional[str] = Field(
        default=None,
        min_length=1,
        description="The digest of the agent's package, such as the sha256 of its lock. With the name, it is the "
        "identity a quarantine holds. Without it, a quarantine of any version of the agent denies the start.",
    )
    parts: list[AgentPart]
    models: list[str] = Field(
        description="Every model the run may use: the models the agent and the agents it calls declare, and the model "
        "the caller binds the run to. The rules judge each."
    )
    declared_models: Optional[list[str]] = Field(
        default=None, description="Of `models`, the ones the agent's package declares, when the caller says."
    )
    bound_model: Optional[str] = Field(
        default=None,
        description="Of `models`, the one the caller's config binds the run to, which is the model that will run.",
    )
    policies: list[AgentPolicy] = Field(default_factory=list)
    enforcers_present: list[str] = Field(
        default_factory=list, description="The enforcers the caller says run where the agent starts."
    )

    @model_validator(mode="after")
    def judge_every_model(self) -> "AgentStartRequest":
        """A declared or bound model the caller left out of `models` is added, so the rules judge it too."""
        for model in [*(self.declared_models or []), self.bound_model]:
            if model and model not in self.models:
                self.models.append(model)
        return self


class ModelCallRequest(BaseModel):
    """One model call an agent is about to make: which agent, and which model it calls."""

    model_config = ConfigDict(extra="forbid")

    hook_event_name: Literal["ModelCall"]
    agent_name: str = Field(min_length=1)
    agent_digest: Optional[str] = Field(default=None, min_length=1)
    model: str = Field(min_length=1, description="The model, named as the agent-start facts name models.")
    id: Optional[str] = Field(default=None, description="The caller's id for this call, kept in the record.")


class QuarantineRequest(BaseModel):
    """A person's or the platform's request to quarantine an agent, with no violation."""

    model_config = ConfigDict(extra="forbid")

    agent_name: str = Field(min_length=1)
    agent_digest: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    requested_by: str = Field(min_length=1)


class QuarantineReleaseRequest(BaseModel):
    """A release of a quarantined agent: who releases it and why."""

    model_config = ConfigDict(extra="forbid")

    agent_name: str = Field(min_length=1)
    agent_digest: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    released_by: str = Field(min_length=1)


# Not underscored: pydantic quotes the class name in the error for a body that isn't an object.
class HookBody(BaseModel):
    """The field every hook body carries, read before the body is validated as its event."""

    hook_event_name: str
