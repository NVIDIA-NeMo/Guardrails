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

This is the *thin* runtime wrapper. All authorization logic lives in
`policy.py`, which knows nothing about Guardrails. Here we only: build a guard
from example policies, register a `safe_tool_call` action that authorizes then
dispatches, and wire it into the rails via `init`.
"""

from __future__ import annotations

import os
import sys

# The Guardrails loader imports this file by path, so make its siblings
# (policy.py, tools.py) importable as top-level modules.
sys.path.insert(0, os.path.dirname(__file__))

from example_policies import HARDENED_GUARD, PRINCIPALS  # noqa: E402
from policy import Principal, ToolCall  # noqa: E402
from tools import TOOLS  # noqa: E402

from nemoguardrails import LLMRails  # noqa: E402
from nemoguardrails.actions import action  # noqa: E402


@action(name="safe_tool_call")
async def safe_tool_call(tool: str, principal_id: str = "anon", **args) -> str:
    """Authorize a proposed tool call, then dispatch it only if allowed."""
    principal = PRINCIPALS.get(principal_id, Principal(principal_id))
    decision = HARDENED_GUARD.authorize(ToolCall(tool, args), principal)
    if not decision.allowed:
        return f"BLOCKED ({tool}): {decision.reason}"

    impl = TOOLS.get(tool)
    if impl is None:
        return f"BLOCKED ({tool}): no implementation registered"
    return impl(**args)


def init(app: LLMRails):
    app.register_action(safe_tool_call, "safe_tool_call")
