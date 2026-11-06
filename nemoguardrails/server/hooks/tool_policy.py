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

"""The hooks server's tool policy: which tools an agent's call may use, and which of their arguments it may pass."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Optional, Union

from nemoguardrails.guardrails.guardrails_types import quoted_identity
from nemoguardrails.rails.llm.config import HooksServerConfig, ToolEntry

# A tool's entry in one list: None for a bare name, else the argument names the entry lists, in config order.
ToolArguments = Optional[tuple[str, ...]]


def _entries(tool_list: Iterable[Union[str, ToolEntry]]) -> Mapping[str, ToolArguments]:
    """Each tool a list names, with its entry's arguments, or None for a bare name."""
    entries: dict[str, ToolArguments] = {}
    for entry in tool_list:
        if isinstance(entry, str):
            entries[entry] = None
        else:
            entries[entry.name] = tuple(entry.arguments)
    return MappingProxyType(entries)


@dataclass(frozen=True)
class ToolPolicy:
    """`hooks_server.allowed_tools`, `blocked_tools` and `allow_unlisted_tools`, read once when the server starts."""

    allowed: Mapping[str, ToolArguments] = field(default_factory=lambda: MappingProxyType({}))
    blocked: Mapping[str, ToolArguments] = field(default_factory=lambda: MappingProxyType({}))
    # A tool is unlisted when `allowed_tools` doesn't name it, whatever `blocked_tools` says about its arguments.
    allow_unlisted: bool = True

    @classmethod
    def from_config(cls, hooks_server: HooksServerConfig) -> "ToolPolicy":
        """The policy a config's hooks_server section sets."""
        return cls(
            allowed=_entries(hooks_server.allowed_tools),
            blocked=_entries(hooks_server.blocked_tools),
            allow_unlisted=hooks_server.allow_unlisted_tools,
        )

    def blocked_by_name(self, name: str) -> bool:
        """Whether `blocked_tools` names *name* bare, which blocks every call to it."""
        return name in self.blocked and self.blocked[name] is None

    def denied_by_name(self, name: str) -> bool:
        """Whether every call to *name* is denied: it's blocked by name, or unlisted while unlisted tools are denied."""
        if self.blocked_by_name(name):
            return True
        return name not in self.allowed and not self.allow_unlisted

    def allowed_arguments(self, name: str) -> ToolArguments:
        """The arguments *name*'s `allowed_tools` entry lists, or None when any argument is allowed."""
        return self.allowed.get(name)

    def decide(self, name: str, arguments: Mapping[str, Any]) -> Optional[str]:
        """Why the policy denies a call to *name* with *arguments*, or None when it lets the call through."""
        passed = arguments.keys()
        if self.blocked_by_name(name):
            return f"tool call '{quoted_identity(name)}' is a blocked tool"
        blocked = next((argument for argument in self.blocked.get(name) or () if argument in passed), None)
        if blocked is not None:
            # The argument's name comes from the config, so the reason can state it.
            return f"tool call '{quoted_identity(name)}' passes argument '{blocked}', which is blocked"
        if name not in self.allowed:
            if self.allow_unlisted:
                return None
            return f"tool call '{quoted_identity(name)}' is not an allowed tool"
        allowed = self.allowed[name]
        unlisted = [] if allowed is None else [argument for argument in passed if argument not in allowed]
        if not unlisted:
            return None
        # A count, not the names: the agent chose them, so they can carry data.
        noun = "argument" if len(unlisted) == 1 else "arguments"
        return f"tool call '{quoted_identity(name)}' passes {len(unlisted)} {noun} its allowed_tools entry doesn't list"
