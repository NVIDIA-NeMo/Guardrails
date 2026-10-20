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

"""The bridge to the engine: the check a hook's tool call asks for, and the verdict the check's result gives."""

import json
from dataclasses import dataclass
from typing import Any, Literal, Optional

from nemoguardrails.rails.llm.options import RailsResult, RailStatus, ToolViolation, ToolViolationType
from nemoguardrails.server.hooks.schemas import HookErrorCode

_UNSPECIFIED_DENY_REASON = "tool call blocked"

# Violations that mean the check couldn't finish, rather than a decision on the call. A malformed call is a decision.
_CHECK_FAILURES = frozenset(
    {
        ToolViolationType.RAIL_FAILED,
        ToolViolationType.EXTRACTION_FAILED,
        ToolViolationType.INVALID_TOOLSET,
        ToolViolationType.INVALID_TOOL_SCHEMA,
    }
)


@dataclass(frozen=True, slots=True)
class ToolInvocation:
    """The tool call a hook asks to check, whichever agent sent it; internal to the hooks server."""

    name: str
    arguments: Any
    call_id: Optional[str] = None


def check_messages(call: ToolInvocation) -> list[dict]:
    """The one assistant message carrying *call*, as `check_async()` takes it; with no id when the call has none."""
    tool_call: dict[str, Any] = {
        "type": "function",
        "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
    }
    if call.call_id is not None:
        tool_call["id"] = call.call_id
    return [{"role": "assistant", "content": None, "tool_calls": [tool_call]}]


@dataclass(frozen=True, slots=True)
class HookOutcome:
    """A check's verdict on a tool call, before it becomes the agent's response."""

    kind: Literal["allow", "deny", "error"]
    reason: Optional[str] = None
    code: Optional[HookErrorCode] = None
    message: Optional[str] = None

    def __post_init__(self) -> None:
        """Refuse any field the outcome's kind doesn't have, so only a valid allow, deny or error exists."""
        if self.code is not None:
            object.__setattr__(self, "code", HookErrorCode(self.code))
        if self.kind == "allow":
            self._require_unset("reason", "code", "message")
        elif self.kind == "deny":
            self._require_text("reason")
            self._require_unset("code", "message")
        elif self.kind == "error":
            self._require_text("message")
            self._require_unset("reason")
        else:
            raise ValueError(f"kind must be 'allow', 'deny' or 'error', got {self.kind!r}")

    def _require_text(self, name: str) -> None:
        """Raise unless the field *name* holds a non-empty string."""
        value = getattr(self, name)
        if value is None or value == "":
            raise ValueError(f"a {self.kind} outcome needs a {name}")
        if not isinstance(value, str):
            raise TypeError(f"a {self.kind} outcome's {name} must be a string")

    def _require_unset(self, *names: str) -> None:
        """Raise when any of the fields *names* is set, since this kind of outcome has none of them."""
        for name in names:
            if getattr(self, name) is not None:
                raise ValueError(f"a {self.kind} outcome has no {name}")

    @classmethod
    def allow(cls) -> "HookOutcome":
        return cls(kind="allow")

    @classmethod
    def deny(cls, reason: str) -> "HookOutcome":
        return cls(kind="deny", reason=reason)

    @classmethod
    def error(cls, message: str, code: Optional[HookErrorCode] = None) -> "HookOutcome":
        return cls(kind="error", code=code, message=message)


def outcome_from_result(result: RailsResult) -> HookOutcome:
    """The verdict a tool-call check's result gives: allow, deny, or an error when no rail decided."""
    if result.status == RailStatus.PASSED:
        return HookOutcome.allow()
    if result.status == RailStatus.BLOCKED:
        return _blocked_outcome(result)
    # A tool-call check rewrites nothing, so any other status breaks the engine's contract.
    return HookOutcome.error(f"unexpected check status {result.status.value!r}")


def _blocked_outcome(result: RailsResult) -> HookOutcome:
    """A deny when any violation is a decision on the call, or an error when every violation is a check failure."""
    violations = result.tool_violations or []
    # A block that reports no violations still blocks, so it is a deny rather than an error.
    if violations and all(violation.violation_type in _CHECK_FAILURES for violation in violations):
        return _check_failed_outcome(violations)
    return HookOutcome.deny(_deny_reason(result))


def _check_failed_outcome(violations: list[ToolViolation]) -> HookOutcome:
    """A `rail_failed` error when every failure is a rail's, otherwise an error with no code naming the first other."""
    other_failures = [
        violation for violation in violations if violation.violation_type != ToolViolationType.RAIL_FAILED
    ]
    if not other_failures:
        return HookOutcome.error(_rail_failed_message(violations[0]), code=HookErrorCode.RAIL_FAILED)
    return HookOutcome.error(f"the tool-call check failed: {other_failures[0].reason}")


def _deny_reason(result: RailsResult) -> str:
    """The first non-empty reason among the decided violations, then the result's reason, then the rail that blocked."""
    for violation in result.tool_violations or []:
        if violation.violation_type not in _CHECK_FAILURES and violation.reason:
            return violation.reason
    return result.reason or result.rail or _UNSPECIFIED_DENY_REASON


def _rail_failed_message(violation: ToolViolation) -> str:
    """The error message for a failed rail, naming the rail when the violation does."""
    if violation.rail is None:
        return f"a tool-call rail failed: {violation.reason}"
    return f"rail '{violation.rail}' failed: {violation.reason}"
