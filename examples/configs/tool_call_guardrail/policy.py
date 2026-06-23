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

"""Runtime-agnostic core for agent tool-call authorization.

This module has NO dependency on NeMo Guardrails (or any agent framework). It
decides whether a proposed tool call should be allowed *before* the tool runs.
The Guardrails integration in `config.py` is a thin wrapper around this; lifting
this file into a standalone package is the planned path to the runtime-agnostic
library.

The model is deliberately small:
  - A `Principal` is whoever the agent is acting for (id + roles + attributes).
  - A `ToolCall` is a proposed (tool name, arguments) pair.
  - A `ToolPolicy` says which roles may call a tool and which argument-level
    `Rule`s must pass.
  - `ToolCallGuard.authorize` returns an allow/deny `PolicyDecision` and is
    default-deny: a tool with no registered policy is refused.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional, Sequence


@dataclass(frozen=True)
class Principal:
    """The identity the agent is acting on behalf of."""

    id: str
    roles: frozenset = field(default_factory=frozenset)
    attributes: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolCall:
    """A tool the agent proposes to invoke, with its arguments."""

    tool: str
    args: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class PolicyDecision:
    """The guard's verdict for a single tool call."""

    allowed: bool
    reason: str


# A Rule inspects a call + principal and returns None to pass, or a string
# explaining why the call must be blocked. Rules are pure functions so they are
# trivial to unit-test and reuse outside any agent runtime.
Rule = Callable[[ToolCall, Principal], Optional[str]]


@dataclass(frozen=True)
class ToolPolicy:
    """Authorization policy for one tool: who may call it, under what argument
    constraints."""

    allowed_roles: frozenset
    rules: Sequence[Rule] = ()


class ToolCallGuard:
    """Evaluates tool calls against per-tool policies. Default-deny."""

    def __init__(self, policies: Mapping[str, ToolPolicy]):
        self._policies = dict(policies)

    def registered_tools(self) -> frozenset:
        """Tools that have a policy. A tool not in this set is default-denied."""
        return frozenset(self._policies)

    def policy_for(self, tool: str) -> Optional[ToolPolicy]:
        """The policy governing `tool`, or None if none is registered."""
        return self._policies.get(tool)

    def authorize(self, call: ToolCall, principal: Principal) -> PolicyDecision:
        policy = self._policies.get(call.tool)
        if policy is None:
            return PolicyDecision(
                False,
                f"no policy registered for tool '{call.tool}' (default-deny)",
            )

        if not (principal.roles & policy.allowed_roles):
            need = ", ".join(sorted(policy.allowed_roles)) or "(none)"
            return PolicyDecision(
                False,
                f"principal '{principal.id}' lacks a role permitting '{call.tool}' (requires one of: {need})",
            )

        for rule in policy.rules:
            reason = rule(call, principal)
            if reason is not None:
                return PolicyDecision(False, reason)

        return PolicyDecision(True, "authorized")


# --- Reusable argument-level rules -----------------------------------------
# These are example rule factories. Each returns a `Rule`. They illustrate the
# two argument checks an agent guardrail most often needs: ownership (the
# principal may only act on resources it owns) and bounds (a numeric argument
# must stay within a ceiling). New rules are just functions with the same shape.


def require_owns_arg(arg_name: str, owned_attr: str = "owned_accounts") -> Rule:
    """Block unless `call.args[arg_name]` is in `principal.attributes[owned_attr]`."""

    def rule(call: ToolCall, principal: Principal) -> Optional[str]:
        target = call.args.get(arg_name)
        owned = principal.attributes.get(owned_attr, ())
        if target not in owned:
            return f"principal '{principal.id}' does not own {arg_name}={target!r}"
        return None

    return rule


def max_numeric_arg(arg_name: str, ceiling: float) -> Rule:
    """Block when a numeric argument exceeds `ceiling`."""

    def rule(call: ToolCall, principal: Principal) -> Optional[str]:
        value = call.args.get(arg_name)
        if value is None:
            return f"missing required argument '{arg_name}'"
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return f"argument '{arg_name}'={value!r} is not numeric"
        if numeric > ceiling:
            return f"{arg_name}={numeric:g} exceeds the allowed ceiling of {ceiling:g}"
        return None

    return rule


def deny_arg_values(arg_name: str, denied) -> Rule:
    """Block when `call.args[arg_name]` is one of the `denied` values."""

    denied_set = frozenset(denied)

    def rule(call: ToolCall, principal: Principal) -> Optional[str]:
        value = call.args.get(arg_name)
        if value in denied_set:
            return f"{arg_name}={value!r} is a denied target"
        return None

    return rule


def arg_matches_pattern(arg_name: str, pattern: str) -> Rule:
    """Block when `call.args[arg_name]` is missing or does not fully match
    `pattern` — e.g. to reject path traversal, injection, or malformed ids."""

    regex = re.compile(pattern)

    def rule(call: ToolCall, principal: Principal) -> Optional[str]:
        value = call.args.get(arg_name)
        if value is None:
            return f"missing required argument '{arg_name}'"
        if not isinstance(value, str) or regex.fullmatch(value) is None:
            return f"{arg_name}={value!r} does not match required pattern {pattern!r}"
        return None

    return rule


def require_principal_attr(attr_name: str, expected: object = True) -> Rule:
    """Block unless `principal.attributes[attr_name]` equals `expected` — e.g. a
    step-up clearance like `mfa_verified` for a sensitive operation."""

    def rule(call: ToolCall, principal: Principal) -> Optional[str]:
        actual = principal.attributes.get(attr_name)
        if actual != expected:
            return f"principal '{principal.id}' lacks required clearance {attr_name}={expected!r} (has {actual!r})"
        return None

    return rule
