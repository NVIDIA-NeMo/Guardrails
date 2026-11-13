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

"""Agent Threat Rules (ATR) detection rail.

Evaluates the input against Agent Threat Rules -- an open, community-maintained
detection standard for AI-agent attacks (like Sigma, but for prompt injection,
jailbreak, tool poisoning, MCP attacks, and skill compromise) -- via the
``pyatr`` package. As an input rail it operates on the user message, so it is
most effective against input-borne attacks such as prompt injection and
jailbreak. Rules are bundled inside ``pyatr``; no rule files or API keys needed.
"""

import logging
from typing import Optional, Set

from nemoguardrails import RailsConfig
from nemoguardrails.actions import action
from nemoguardrails.actions.rail_outcome import RailOutcome
from nemoguardrails.library.atr.rail_config import DEFAULT_BLOCK_SEVERITIES

log = logging.getLogger(__name__)


def _allow(rules: Optional[list] = None) -> RailOutcome:
    """Allow, recording any sub-threshold matches for traces and monitoring."""
    return RailOutcome.allow(metadata={"rules": rules or [], "max_severity": None})


def _block_severities(config: Optional[RailsConfig]) -> Set[str]:
    """Read block severities from the rail's ``atr`` config section.

    The section is declared by ``rail_config.build_config_spec`` and is absent
    when a config does not mention the rail. An explicitly configured
    ``block_severities: []`` is honored — it flags nothing, giving a
    monitor-only rail — so only an absent section falls back to the default.
    """
    atr_config = getattr(getattr(config, "rails", None), "config", None)
    atr_config = getattr(atr_config, "atr", None)
    if atr_config is None:
        return {s.lower() for s in DEFAULT_BLOCK_SEVERITIES}
    return {str(s).lower() for s in atr_config.block_severities}


@action()
async def atr_detection(text: str, config: RailsConfig) -> RailOutcome:
    """Detect AI-agent threats in *text* using Agent Threat Rules.

    Args:
        text: The text to evaluate (typically the user message for an input rail).
        config: The Rails configuration; ``rails.config.atr.block_severities``
            overrides the default ``["critical", "high"]`` block list.

    Returns:
        A blocking RailOutcome when a rule at or above a block severity matched,
        otherwise an allowing one. Both carry ``rules`` (matched ATR rule IDs)
        and ``max_severity`` in their metadata.

    Raises:
        ImportError: If the ``pyatr`` package is not installed.
    """
    try:
        from pyatr import scan
    except ImportError as exc:
        raise ImportError(
            "The `pyatr` package is required for the ATR rail. Install it with: pip install pyatr"
        ) from exc

    if not text:
        return _allow()

    block = _block_severities(config)
    matches = scan(text)  # bundled ATR rules; returns matches sorted by severity
    blocking = [match for match in matches if match.severity.lower() in block]
    if not blocking:
        return _allow([match.rule_id for match in matches])

    rule_ids = [match.rule_id for match in blocking]
    log.info("ATR rail flagged input on rule(s): %s", ", ".join(rule_ids))
    return RailOutcome.block(metadata={"rules": rule_ids, "max_severity": blocking[0].severity})
