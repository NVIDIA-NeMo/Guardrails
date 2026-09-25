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

"""Conduct guard rail action.

Wraps ``conduct_nemo_guard`` so any Colang flow can gate a message
through Conduct Guard's runtime policy engine. Returns a
:class:`RailOutcome` shaped for the standard NeMo Guardrails
`if $result.is_blocked` idiom.
"""

import logging
import os
from typing import Any, Literal, Optional

from nemoguardrails.actions import action
from nemoguardrails.actions.rail_outcome import RailOutcome

from nemoguardrails.library.conduct.errors import (
    ConductPluginConfigurationError,
    ConductPluginImportError,
)

log = logging.getLogger(__name__)

RailDirection = Literal["input", "output"]


_conduct_action = None


def _load_action():
    """Import the plugin lazily so the module still parses when the
    ``conduct-nemo-guard`` package isn't installed. Raises a friendly
    error at rail-run time if the caller forgot to ``pip install``."""
    global _conduct_action
    if _conduct_action is not None:
        return _conduct_action
    try:
        from conduct_nemo_guard.actions import conduct_guard_check
    except ImportError as e:  # pragma: no cover — exercised in tests
        raise ConductPluginImportError(
            "The Conduct guard rail requires the 'conduct-nemo-guard' package. "
            "Install it with:\n"
            "    pip install conduct-nemo-guard"
        ) from e
    _conduct_action = conduct_guard_check
    return _conduct_action


@action(name="ConductCheckAction")
async def conduct_check(
    text: str = "",
    rail: RailDirection = "input",
    api_url: Optional[str] = None,
    agent_token: Optional[str] = None,
    workspace_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> RailOutcome:
    """Evaluate ``text`` against the caller's Conduct Guard policy.

    Parameters
    ----------
    text:
        The user turn (input rail) or bot response (output rail) to
        check. Bound by ``RAIL.spec.surfaces``.
    rail:
        Which rail is calling — ``"input"`` or ``"output"``. Used for
        audit attribution and to keep the action symmetric.
    api_url:
        Conduct API endpoint. Defaults to https://api.conductai.ai or
        the ``CONDUCT_API_URL`` env var.
    agent_token:
        A ``cond_agt_*`` token minted at conductai.ai. Falls back to
        ``CONDUCT_AGENT_TOKEN`` if unset.
    workspace_id:
        Optional workspace pin. Falls back to ``CONDUCT_WORKSPACE_ID``.
    session_id:
        Optional session id — echoed in audit rows for correlation
        with HITL approval receipts.
    """
    # Env fall-throughs mirror what conduct_nemo_guard does internally,
    # but exposing them here means a workspace can drop the config block
    # into config.yml with either literal values or `os.environ/VAR`
    # references and both cases work identically.
    if api_url is None:
        api_url = os.environ.get("CONDUCT_API_URL", "https://api.conductai.ai")
    if agent_token is None:
        agent_token = os.environ.get("CONDUCT_AGENT_TOKEN")
    if workspace_id is None:
        workspace_id = os.environ.get("CONDUCT_WORKSPACE_ID")

    if not agent_token:
        raise ConductPluginConfigurationError(
            "Conduct guard rail requires an agent_token. Set "
            "CONDUCT_AGENT_TOKEN or supply agent_token in the rails.conduct "
            "config block. Mint one at https://conductai.ai."
        )

    check = _load_action()
    result = await check(prompt=text, session_id=session_id)

    verdict = str(result.get("verdict", "unknown"))
    rule_id = result.get("rule_id")
    message = result.get("message") or f"Blocked by Conduct rule {rule_id}" if rule_id else "Blocked by Conduct policy"

    is_blocked = verdict in ("block", "approval")

    # The full raw response is stashed under `metadata` so downstream
    # flows can pattern-match on verdict === "warning" / "advisory"
    # etc without another action call. Property-9 in Conduct means the
    # payload is already redacted at the audit boundary — nothing
    # sensitive lands here.
    return RailOutcome(
        is_blocked=is_blocked,
        rail=rail,
        metadata={
            "verdict": verdict,
            "rule_id": rule_id,
            "message": message,
            "surface": "nemo",
        },
    )
