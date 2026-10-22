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

"""The hooks server's tool policy, decided on a hook's call before any rail runs."""

from typing import Any

import pytest

from nemoguardrails.rails.llm.config import RailsConfig
from nemoguardrails.server.hooks.tool_policy import ToolPolicy


def _policy(**hooks: Any) -> ToolPolicy:
    """The policy a config with *hooks* as its hooks_server section sets."""
    return ToolPolicy.from_config(RailsConfig.from_content(config={"models": [], "hooks_server": hooks}).hooks_server)


class TestDecide:
    def test_tool_blocked_by_name_is_denied(self):
        """A call to a tool hooks_server.blocked_tools names bare is denied as a blocked tool."""
        reason = _policy(blocked_tools=["Bash"]).decide("Bash", {"command": "ls"})

        assert reason == "tool call 'Bash' is a blocked tool"

    def test_call_passing_a_blocked_argument_is_denied_naming_it(self):
        """A call that passes an argument a hooks_server.blocked_tools entry lists is denied, naming the argument."""
        policy = _policy(blocked_tools=[{"name": "Bash", "arguments": ["sandbox", "timeout"]}])

        reason = policy.decide("Bash", {"command": "ls", "timeout": 5})

        assert reason == "tool call 'Bash' passes argument 'timeout', which is blocked"

    def test_call_without_a_blocked_argument_is_allowed(self):
        """A call to a tool with a blocked_tools entry is allowed when it passes none of the entry's arguments."""
        assert (
            _policy(blocked_tools=[{"name": "Bash", "arguments": ["sandbox"]}]).decide("Bash", {"command": "ls"})
            is None
        )

    @pytest.mark.parametrize(
        ("arguments", "reason"),
        [
            ({"query": "q", "limit": 3}, "tool call 'search' passes 1 argument its allowed_tools entry doesn't list"),
            (
                {"query": "q", "token": "secret", "limit": 3},
                "tool call 'search' passes 2 arguments its allowed_tools entry doesn't list",
            ),
        ],
        ids=["one", "two"],
    )
    def test_arguments_outside_an_allowed_entry_are_denied_by_count(self, arguments, reason):
        """Arguments a hooks_server.allowed_tools entry doesn't list are denied, counted rather than named."""
        assert _policy(allowed_tools=[{"name": "search", "arguments": ["query"]}]).decide("search", arguments) == reason

    def test_tool_allowed_by_name_takes_any_arguments(self):
        """A call to a tool hooks_server.allowed_tools names bare is allowed with any arguments."""
        policy = _policy(allowed_tools=["Bash"], allow_unlisted_tools=False)

        assert policy.decide("Bash", {"command": "ls", "anything": [1]}) is None

    def test_unlisted_tool_is_allowed_by_default(self):
        """A call to a tool allowed_tools doesn't name is allowed while allow_unlisted_tools keeps its default."""
        assert _policy(allowed_tools=["Read"]).decide("Bash", {"command": "ls"}) is None

    def test_unlisted_tool_is_denied_when_unlisted_tools_are_denied(self):
        """With allow_unlisted_tools false, a call to a tool hooks_server.allowed_tools doesn't name is not allowed."""
        policy = _policy(allowed_tools=["Read"], allow_unlisted_tools=False)

        assert (
            policy.decide("WebFetch", {"url": "https://example.com"}) == "tool call 'WebFetch' is not an allowed tool"
        )

    def test_blocked_argument_entry_leaves_the_tool_unlisted(self):
        """A tool only blocked_tools names, by an argument entry, is denied while unlisted tools are denied."""
        policy = _policy(
            allowed_tools=["Read"],
            blocked_tools=[{"name": "Bash", "arguments": ["sandbox"]}],
            allow_unlisted_tools=False,
        )

        assert policy.decide("Bash", {"command": "ls"}) == "tool call 'Bash' is not an allowed tool"

    def test_blocked_argument_is_reported_before_one_outside_the_allowed_entry(self):
        """An argument both blocked and outside the tool's allowed_tools entry is reported as blocked."""
        policy = _policy(
            allowed_tools=[{"name": "search", "arguments": ["query"]}],
            blocked_tools=[{"name": "search", "arguments": ["limit"]}],
        )

        reason = policy.decide("search", {"query": "q", "limit": 3})

        assert reason == "tool call 'search' passes argument 'limit', which is blocked"

    def test_default_policy_allows_every_call(self):
        """With no hooks_server section, the policy allows a call to any tool with any arguments."""
        assert _policy().decide("mcp__any__tool", {"anything": True}) is None


class TestNames:
    @pytest.mark.parametrize(
        ("hooks", "blocked", "denied"),
        [
            ({"blocked_tools": ["Bash"]}, True, True),
            ({"blocked_tools": [{"name": "Bash", "arguments": ["sandbox"]}]}, False, False),
            ({"allowed_tools": ["Read"], "allow_unlisted_tools": False}, False, True),
            ({"allowed_tools": ["Bash"], "allow_unlisted_tools": False}, False, False),
        ],
        ids=["blocked-by-name", "blocked-by-argument", "unlisted-denied", "allowed"],
    )
    def test_whether_every_call_to_a_tool_is_denied(self, hooks, blocked, denied):
        """A tool is blocked by name only by a bare blocked_tools entry, and denied by name when blocked or unlisted."""
        policy = _policy(**hooks)

        assert (policy.blocked_by_name("Bash"), policy.denied_by_name("Bash")) == (blocked, denied)

    @pytest.mark.parametrize(
        ("hooks", "arguments"),
        [
            ({"allowed_tools": [{"name": "Bash", "arguments": ["command"]}]}, ("command",)),
            ({"allowed_tools": ["Bash"]}, None),
            ({}, None),
        ],
        ids=["entry", "bare-name", "unlisted"],
    )
    def test_allowed_arguments_are_an_entrys_list_or_none(self, hooks, arguments):
        """A tool's allowed arguments are its allowed_tools entry's list, or None when any argument is allowed."""
        assert _policy(**hooks).allowed_arguments("Bash") == arguments
