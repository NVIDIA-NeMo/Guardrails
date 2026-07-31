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

"""Guard configuration for the Hermes agent tool surface.

Drop-in replacement for example_policies.py targeting the real Hermes tool
registry (terminal, execute_code, writefile, cronjob, skillmanage, etc.)
instead of the coding-agent placeholder. Pass --policies hermes to
propose_guardrails.py and apply_guardrails.py to scan against this surface.

Usage:
    python3 propose_guardrails.py --docs advisories_hermes/ --policies hermes
    # edit review_queue.json, flip approved candidates to true
    python3 apply_guardrails.py --queue review_queue.json --policies hermes
"""

from __future__ import annotations

from policy import (
    Principal,
    ToolCallGuard,
    ToolPolicy,
    arg_matches_pattern,
    deny_arg_matching,
    deny_arg_values,
    max_numeric_arg,
    require_arg_prefix,
    require_owns_arg,
    require_principal_attr,
)
from scanner.scan import ArgSpec

# ---------------------------------------------------------------------------
# VULNERABLE_GUARD — intentional gaps for the scanner to surface.
#
# terminal: loose timeout ceiling, no command denylist.
# readfile / search_files: no path constraints.
# execute_code: elevated not required.
# writefile / patch: unpoliced — flagged as coverage gaps.
# cronjob / skillmanage / delegate_task / memory / send_message:
#   all unpoliced — the highest-risk new tools in the Hermes surface.
# ---------------------------------------------------------------------------
VULNERABLE_GUARD = ToolCallGuard(
    {
        "terminal": ToolPolicy(
            allowed_roles=frozenset({"developer", "admin"}),
            rules=[max_numeric_arg("timeout_seconds", ceiling=3600)],
        ),
        "execute_code": ToolPolicy(
            allowed_roles=frozenset({"developer", "admin"}),
            rules=[max_numeric_arg("timeout_seconds", ceiling=3600)],
        ),
        "readfile": ToolPolicy(
            allowed_roles=frozenset({"developer", "admin", "analyst"}),
        ),
        "search_files": ToolPolicy(
            allowed_roles=frozenset({"developer", "admin", "analyst"}),
        ),
        "process": ToolPolicy(
            allowed_roles=frozenset({"developer", "admin"}),
        ),
    }
)

# ---------------------------------------------------------------------------
# HARDENED_GUARD — VULNERABLE_GUARD with scanner-derived rules folded in.
#
# Rules trace to the Hermes-specific advisories in advisories_hermes/:
#   terminal <- deny_arg_matching (command-injection-terminal)
#             + max_numeric_arg   (unbounded-shell-timeout)
#   execute_code <- require_principal_attr (privilege-escalation-execute-code)
#                 + max_numeric_arg        (unbounded-execute-timeout)
#   writefile / patch <- require_arg_prefix + require_principal_attr
#                        (prefix-ownership-bypass / privilege-escalation)
#   cronjob <- require_principal_attr (privilege-escalation-cronjob)
#   skillmanage <- require_principal_attr (privilege-escalation-skillmanage)
#               + deny_arg_values         (disallowed-target-skillmanage-delete)
#   delegate_task <- max_numeric_arg      (unbounded-arg-delegate-task)
#                  + require_principal_attr (privilege-escalation-delegate)
#   memory <- require_principal_attr (privilege-escalation-memory)
#   send_message <- require_owns_arg (ownership-bypass-send-message)
#   readfile <- arg_matches_pattern  (argument-injection-readfile)
# ---------------------------------------------------------------------------
HARDENED_GUARD = ToolCallGuard(
    {
        "terminal": ToolPolicy(
            allowed_roles=frozenset({"developer", "admin"}),
            rules=[
                # unbounded-arg: tighten loose 3600s ceiling to 300s.
                max_numeric_arg("timeout_seconds", ceiling=300),
                # disallowed-pattern: block sensitive-file reads, destructive rm,
                # and shell-chaining metacharacters surfaced by AgentBreaker audits.
                deny_arg_matching(
                    "command",
                    r"/etc/(passwd|shadow|sudoers)|\brm\s+-rf\b|[;&|`]|\$\(|\bsudo\b",
                ),
            ],
        ),
        "execute_code": ToolPolicy(
            allowed_roles=frozenset({"developer", "admin"}),
            rules=[
                # privilege-escalation: arbitrary code execution requires step-up.
                require_principal_attr("elevated"),
                # unbounded-arg: cap execution time.
                max_numeric_arg("timeout_seconds", ceiling=120),
            ],
        ),
        "readfile": ToolPolicy(
            allowed_roles=frozenset({"developer", "admin", "analyst"}),
            # argument-injection: workspace-relative paths only — reject traversal.
            rules=[arg_matches_pattern("path", r"(?!(?:.*/)?\.\.(?:/|$))[\w.-][\w./-]*")],
        ),
        "search_files": ToolPolicy(
            allowed_roles=frozenset({"developer", "admin", "analyst"}),
            rules=[arg_matches_pattern("path", r"(?!(?:.*/)?\.\.(?:/|$))[\w.-][\w./-]*")],
        ),
        "writefile": ToolPolicy(
            allowed_roles=frozenset({"developer", "admin"}),
            rules=[
                # prefix-ownership-bypass: writes must stay in the principal's prefix.
                require_arg_prefix("path", "owned_paths"),
                # privilege-escalation: write operations require step-up.
                require_principal_attr("elevated"),
            ],
        ),
        "patch": ToolPolicy(
            allowed_roles=frozenset({"developer", "admin"}),
            rules=[
                require_arg_prefix("path", "owned_paths"),
                require_principal_attr("elevated"),
            ],
        ),
        "process": ToolPolicy(
            allowed_roles=frozenset({"developer", "admin"}),
        ),
        "cronjob": ToolPolicy(
            allowed_roles=frozenset({"admin"}),
            # privilege-escalation: creating persistent scheduled tasks is a
            # write to a shared resource; requires both admin role and step-up.
            rules=[require_principal_attr("elevated")],
        ),
        "skillmanage": ToolPolicy(
            allowed_roles=frozenset({"admin"}),
            rules=[
                # privilege-escalation: modifying agent skills (RT-6 rug-pull risk)
                # requires step-up.
                require_principal_attr("elevated"),
                # disallowed-target: delete is irreversible; block it explicitly.
                deny_arg_values("action", ["delete"]),
            ],
        ),
        "delegate_task": ToolPolicy(
            allowed_roles=frozenset({"developer", "admin"}),
            rules=[
                # unbounded-arg: cap sub-agent proliferation.
                max_numeric_arg("max_agents", ceiling=3),
                # privilege-escalation: spawning sub-agents that could act with
                # the principal's full authority requires step-up.
                require_principal_attr("elevated"),
            ],
        ),
        "memory": ToolPolicy(
            allowed_roles=frozenset({"developer", "admin"}),
            # privilege-escalation: persistent memory modifications (RT-13
            # agent memory compromise) require step-up.
            rules=[require_principal_attr("elevated")],
        ),
        "send_message": ToolPolicy(
            allowed_roles=frozenset({"developer", "admin", "analyst"}),
            # ownership-bypass: can only message recipients the principal
            # is authorized to contact — blocks exfiltration via cross-platform
            # messaging (RT-4 data exfiltration).
            rules=[require_owns_arg("recipient", "approved_recipients")],
        ),
    }
)

# ---------------------------------------------------------------------------
# Tool registry, schemas, and principal configuration
# ---------------------------------------------------------------------------

TOOL_REGISTRY = {
    "terminal": "Execute a Linux shell command in the foreground or background",
    "execute_code": "Execute a self-contained Python script",
    "readfile": "Read the contents of a file",
    "writefile": "Create or completely replace the contents of a file",
    "search_files": "Search for patterns inside files or find files by name",
    "patch": "Apply a targeted find-and-replace edit to a file",
    "process": "Inspect, poll, write to, or control a background process",
    "cronjob": "Create, list, update, pause, resume, or remove a scheduled cron job",
    "skillmanage": "Load, create, patch, or delete a reusable agent skill",
    "delegate_task": "Spawn one or more sub-agents to reason about or execute a task",
    "memory": "Add, replace, or remove a durable fact that persists across sessions",
    "send_message": "Send a message to a user or channel on a supported platform",
}

TOOL_SCHEMAS = {
    "terminal": [
        ArgSpec("command", "string", "shell command to execute"),
        ArgSpec("background", "boolean", "run in the background (true/false)"),
        ArgSpec("timeout_seconds", "number", "max wall-clock seconds before kill"),
    ],
    "execute_code": [
        ArgSpec("code", "string", "Python source code to execute"),
        ArgSpec("timeout_seconds", "number", "max execution time in seconds"),
    ],
    "readfile": [
        ArgSpec("path", "string", "path to the file to read"),
    ],
    "writefile": [
        ArgSpec("path", "string", "path to the file to write"),
        ArgSpec("content", "string", "full file contents to write"),
    ],
    "search_files": [
        ArgSpec("pattern", "string", "search pattern or glob"),
        ArgSpec("path", "string", "directory or file to search within"),
        ArgSpec("mode", "string", "content or filename search mode"),
    ],
    "patch": [
        ArgSpec("path", "string", "path to the file to edit"),
        ArgSpec("old_str", "string", "exact string to find"),
        ArgSpec("new_str", "string", "replacement string"),
    ],
    "process": [
        ArgSpec("action", "string", "inspect, poll, write, or kill"),
        ArgSpec("pid", "string", "process identifier"),
        ArgSpec("input", "string", "input to write to the process stdin"),
    ],
    "cronjob": [
        ArgSpec("action", "string", "create, list, update, pause, resume, or remove"),
        ArgSpec("name", "string", "human-readable job name"),
        ArgSpec("schedule", "string", "cron expression (e.g. 0 2 * * *)"),
        ArgSpec("command", "string", "command or skill to run on schedule"),
    ],
    "skillmanage": [
        ArgSpec("action", "string", "load, create, patch, or delete"),
        ArgSpec("name", "string", "skill name"),
        ArgSpec("content", "string", "skill definition content (for create/patch)"),
    ],
    "delegate_task": [
        ArgSpec("task", "string", "task description for the sub-agent"),
        ArgSpec("context", "string", "context or constraints for the sub-agent"),
        ArgSpec("max_agents", "number", "maximum number of sub-agents to spawn"),
    ],
    "memory": [
        ArgSpec("action", "string", "add, replace, or remove"),
        ArgSpec("key", "string", "memory key"),
        ArgSpec("value", "string", "memory value (for add or replace)"),
    ],
    "send_message": [
        ArgSpec("recipient", "string", "target user ID or channel name"),
        ArgSpec("message", "string", "message content"),
        ArgSpec("channel", "string", "platform: slack, email, discord, telegram"),
    ],
}

PRINCIPAL_ATTRS = [
    "elevated",
    "mfa_verified",
    "owned_paths",
    "approved_recipients",
]

PRINCIPALS = {
    "sven-admin": Principal(
        "sven-admin",
        roles=frozenset({"developer", "admin", "analyst"}),
        attributes={
            "owned_paths": frozenset({"/sandbox/workspace/sven/"}),
            "approved_recipients": frozenset({"team-guardrails", "shaona_ghosh"}),
            "elevated": True,
            "mfa_verified": True,
        },
    ),
    "agent-standard": Principal(
        "agent-standard",
        roles=frozenset({"developer"}),
        attributes={
            "owned_paths": frozenset({"/sandbox/workspace/agent/"}),
            "approved_recipients": frozenset(),
            "elevated": False,
        },
    ),
    "agent-elevated": Principal(
        "agent-elevated",
        roles=frozenset({"developer", "admin"}),
        attributes={
            "owned_paths": frozenset({"/sandbox/workspace/agent/", "/sandbox/workspace/shared/"}),
            "approved_recipients": frozenset({"team-guardrails"}),
            "elevated": True,
            "mfa_verified": True,
        },
    ),
    "anon": Principal("anon", roles=frozenset()),
}
