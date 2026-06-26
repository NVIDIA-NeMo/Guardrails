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

"""Example guard configuration, shared by the Guardrails wiring and the offline
demo. Like `policy.py`, this has no Guardrails dependency, so `demo.py` can run
it without installing the framework."""

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

# Example agent: a coding/dev assistant. The guard sits between its decision to
# call a tool and the call's execution.

# The "before" guard. Who may call each tool, under which baseline argument
# constraints. Deliberately thin: git_push has no ownership check and run_shell's
# timeout ceiling is loose — exactly the gaps the field scanner is meant to
# surface and tighten. write_file and http_request are unpoliced, so the coverage
# analyzer flags them. demo.py and demo_bridge.py run against this guard to show
# the gaps; the rails serve HARDENED_GUARD (below).
VULNERABLE_GUARD = ToolCallGuard(
    {
        "read_file": ToolPolicy(
            allowed_roles=frozenset({"developer", "ci", "reviewer"}),
        ),
        "run_shell": ToolPolicy(
            allowed_roles=frozenset({"developer"}),
            rules=[max_numeric_arg("timeout_seconds", ceiling=3600)],
        ),
        "git_push": ToolPolicy(
            allowed_roles=frozenset({"developer"}),
        ),
        "install_package": ToolPolicy(
            allowed_roles=frozenset({"developer"}),
        ),
    }
)

# The "after" guard: VULNERABLE_GUARD with the human-approved output of the
# scanner→synthesis pipeline folded in (see demo_bridge.py for the derivation).
# Each rule below traces to one finding over scanner/sample_docs; the two
# previously-unpoliced tools (write_file, http_request) also gain an explicit
# role grant, since an approved rule can never open role access on its own
# (synthesis/review.apply fail-closes — that human step is deliberate). This is
# the guard the Guardrails runtime serves (config.py).
HARDENED_GUARD = ToolCallGuard(
    {
        "read_file": ToolPolicy(
            allowed_roles=frozenset({"developer", "ci", "reviewer"}),
            # argument-injection: workspace-relative paths only — reject parent-dir
            # traversal ('..' components), absolute paths, and shell metacharacters.
            rules=[arg_matches_pattern("path", r"(?!(?:.*/)?\.\.(?:/|$))[\w.-][\w./-]*")],
        ),
        "run_shell": ToolPolicy(
            allowed_roles=frozenset({"developer"}),
            # unbounded-arg: tighten the loose 3600s ceiling to 300s.
            rules=[max_numeric_arg("timeout_seconds", ceiling=300)],
        ),
        "git_push": ToolPolicy(
            allowed_roles=frozenset({"developer"}),
            # ownership-bypass: the principal must own the target remote.
            rules=[require_owns_arg("remote", "owned_repos")],
        ),
        "install_package": ToolPolicy(
            allowed_roles=frozenset({"developer"}),
            # disallowed-target: block known-malicious package names.
            rules=[deny_arg_values("name", ["leftpad-evil", "reqwest-utils"])],
        ),
        "write_file": ToolPolicy(
            allowed_roles=frozenset({"developer"}),
            rules=[
                # prefix-ownership-bypass: writes must stay in the principal's
                # owned workspace prefix, and (privilege-escalation) require step-up.
                require_arg_prefix("path", "owned_paths"),
                require_principal_attr("elevated"),
            ],
        ),
        "http_request": ToolPolicy(
            allowed_roles=frozenset({"developer"}),
            # disallowed-pattern: block egress to cloud-metadata / internal hosts.
            rules=[deny_arg_matching("url", r"(169\.254\.169\.254|^https?://(localhost|127\.|10\.|192\.168\.))")],
        ),
    }
)

# The tools the agent can call (name -> description). Single source of truth for
# both the scanner (which grounds findings against it) and the coverage analyzer
# (which flags tools with no policy). It is a superset of the tools VULNERABLE_GUARD has
# policies for: write_file and http_request are unpoliced so gaps are surfaced.
TOOL_REGISTRY = {
    "read_file": "Read a file from the workspace",
    "write_file": "Create or overwrite a file in the workspace",
    "run_shell": "Execute a shell command in the sandbox",
    "http_request": "Make an outbound HTTP request",
    "git_push": "Push commits to a git remote",
    "install_package": "Install a dependency from a package index",
}

# Each tool's argument schema, handed to the LLM extractor so it grounds a
# proposed `arg_name` against real argument names instead of guessing from prose.
TOOL_SCHEMAS = {
    "read_file": [ArgSpec("path", "string", "workspace-relative file path")],
    "write_file": [
        ArgSpec("path", "string", "workspace-relative file path"),
        ArgSpec("content", "string", "file contents to write"),
    ],
    "run_shell": [
        ArgSpec("command", "string", "shell command to execute"),
        ArgSpec("timeout_seconds", "number", "max wall-clock seconds before kill"),
    ],
    "http_request": [
        ArgSpec("url", "string", "target URL"),
        ArgSpec("method", "string", "HTTP method"),
    ],
    "git_push": [
        ArgSpec("remote", "string", "git remote the principal owns"),
        ArgSpec("branch", "string", "branch to push"),
    ],
    "install_package": [
        ArgSpec("name", "string", "package name"),
        ArgSpec("version", "string", "version spec"),
    ],
}

# Principal attributes the guard recognizes — the values an `attr_name` param
# (e.g. on a privilege-escalation finding) or an `owned_attr` may reference.
PRINCIPAL_ATTRS = ["elevated", "mfa_verified", "owned_repos", "owned_paths", "approved_deploy"]

# Principals the agent might be acting for. `owned_paths` holds the workspace
# prefixes a principal is scoped to — the set a prefix-ownership rule checks a
# path argument against (cf. `owned_repos`, which is exact-membership).
PRINCIPALS = {
    "dev-alice": Principal(
        "dev-alice",
        roles=frozenset({"developer"}),
        attributes={
            "owned_repos": frozenset({"origin"}),
            "owned_paths": frozenset({"/workspace/alice/"}),
            "elevated": True,
        },
    ),
    "dev-bob": Principal(
        "dev-bob",
        roles=frozenset({"developer"}),
        attributes={
            "owned_repos": frozenset({"fork-bob"}),
            "owned_paths": frozenset({"/workspace/bob/"}),
            "elevated": False,
        },
    ),
    "ci-bot": Principal("ci-bot", roles=frozenset({"ci"})),
    "anon": Principal("anon", roles=frozenset()),
}
