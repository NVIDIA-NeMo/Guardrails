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

"""NeMo Guardrails integration for the agent tool-call guardrail.

This is the *thin* runtime wrapper. The per-call authorization logic lives in
`policy.py` and the session-aware egress backstop in `egress.py`, neither of
which knows anything about Guardrails. Here we build a guard from example
policies, register a `safe_tool_call` action that authorizes then dispatches,
and wire it into the rails via `init`.

The stateless per-call guard always runs. The session-aware egress layer is a
prototype and is **off by default**: set the environment variable ``TCG_EGRESS=1``
to additionally route calls through the `EgressMonitor`.
"""

from __future__ import annotations

import os
import sys

# The Guardrails loader imports this file by path, so make its siblings
# (policy.py, tools.py, egress.py) importable as top-level modules.
sys.path.insert(0, os.path.dirname(__file__))

from egress import EgressLimits, EgressMonitor, authorize_with_egress  # noqa: E402
from example_policies import HARDENED_GUARD, PRINCIPALS  # noqa: E402
from policy import Principal, ToolCall  # noqa: E402
from tools import TOOLS  # noqa: E402

from nemoguardrails import LLMRails  # noqa: E402
from nemoguardrails.actions import action  # noqa: E402

# Opt-in session-aware egress backstop (still a prototype), off unless TCG_EGRESS=1.
# When enabled, limits are tuned LOW so the layer is observable in a short demo; a
# real deployment would set realistic ceilings and back the state with a shared
# store keyed by the conversation id.
USE_EGRESS = os.getenv("TCG_EGRESS", "0") == "1"
EGRESS_MONITOR = (
    EgressMonitor(
        EgressLimits(
            max_requests=5,
            max_distinct_hosts=3,
            max_cumulative_bytes=50_000,
            max_requests_per_window=5,
            window_seconds=300,
        )
    )
    if USE_EGRESS
    else None
)


@action(name="safe_tool_call")
async def safe_tool_call(tool: str, principal_id: str = "anon", session_id: str = "default", **args) -> str:
    """Authorize a proposed tool call, then dispatch only if allowed. The stateless
    per-call guard (`HARDENED_GUARD`) always runs; when egress monitoring is enabled
    (``TCG_EGRESS=1``) the call is additionally checked against the session-aware
    `EgressMonitor`. `session_id` should be the conversation id; it defaults so the
    canned flows run without threading it through every call."""
    principal = PRINCIPALS.get(principal_id, Principal(principal_id))
    call = ToolCall(tool, args)
    if EGRESS_MONITOR is not None:
        decision = authorize_with_egress(HARDENED_GUARD, EGRESS_MONITOR, session_id, call, principal)
    else:
        decision = HARDENED_GUARD.authorize(call, principal)
    if not decision.allowed:
        return f"BLOCKED ({tool}): {decision.reason}"

    impl = TOOLS.get(tool)
    if impl is None:
        return f"BLOCKED ({tool}): no implementation registered"
    return impl(**args)


def init(app: LLMRails):
    app.register_action(safe_tool_call, "safe_tool_call")
