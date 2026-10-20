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
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class HookErrorCode(str, Enum):
    """The `error.code` of a hooks request that ends in an error rather than an allow or a deny."""

    UNSUPPORTED_HOOK_EVENT = "unsupported_hook_event"
    RAIL_FAILED = "rail_failed"
    QUEUE_FULL = "queue_full"


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


# Not underscored: pydantic quotes the class name in the error for a body that isn't an object.
class HookBody(BaseModel):
    """The field every hook body carries, read before the body is validated as its event."""

    hook_event_name: str
