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
    max_numeric_arg,
    require_owns_arg,
)
from scanner.scan import ArgSpec

# Who may call each tool, under which argument constraints.
GUARD = ToolCallGuard(
    {
        "read_account": ToolPolicy(
            allowed_roles=frozenset({"customer", "teller"}),
            rules=[require_owns_arg("account_id")],
        ),
        "transfer_funds": ToolPolicy(
            allowed_roles=frozenset({"customer"}),
            rules=[
                require_owns_arg("from_account"),
                max_numeric_arg("amount", ceiling=10_000),
            ],
        ),
    }
)

# The tools the agent can call (name -> description). Single source of truth for
# both the scanner (which grounds findings against it) and the coverage analyzer
# (which flags tools with no policy). It is a superset of the tools GUARD has
# policies for: close_account is intentionally unpoliced so a gap is surfaced.
TOOL_REGISTRY = {
    "read_account": "Read an account's balance",
    "transfer_funds": "Move money between accounts",
    "close_account": "Permanently close an account",
}

# Each tool's argument schema, handed to the LLM extractor so it grounds a
# proposed `arg_name` against real argument names instead of guessing from prose.
TOOL_SCHEMAS = {
    "read_account": [ArgSpec("account_id", "string", "the account being read")],
    "transfer_funds": [
        ArgSpec("from_account", "string", "source account; must be owned by the principal"),
        ArgSpec("to_account", "string", "destination account"),
        ArgSpec("amount", "number", "amount of money to move"),
    ],
    "close_account": [ArgSpec("account_id", "string", "the account to close")],
}

# Principal attributes the guard recognizes — the values an `attr_name` param
# (e.g. on a privilege-escalation finding) may legitimately reference.
PRINCIPAL_ATTRS = ["mfa_verified", "elevated", "owned_accounts"]

# Principals the agent might be acting for.
PRINCIPALS = {
    "cust-alice": Principal(
        "cust-alice",
        roles=frozenset({"customer"}),
        attributes={"owned_accounts": frozenset({"acct-1001"})},
    ),
    "cust-bob": Principal(
        "cust-bob",
        roles=frozenset({"customer"}),
        attributes={"owned_accounts": frozenset({"acct-1002"})},
    ),
    "anon": Principal("anon", roles=frozenset()),
}
