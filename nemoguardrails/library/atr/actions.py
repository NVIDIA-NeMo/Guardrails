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
from typing import List, Optional, Set, TypedDict

from nemoguardrails import RailsConfig
from nemoguardrails.actions import action

log = logging.getLogger(__name__)

# ATR severities that flag the content by default. Lower severities
# ("medium", "low") match but do not flag, keeping false positives low.
DEFAULT_BLOCK_SEVERITIES = ("critical", "high")


class ATRDetectionResult(TypedDict):
    """Result of evaluating text against Agent Threat Rules.

    Attributes:
        flagged: True if a rule at or above the block severity matched.
        rules: Matched ATR rule IDs (e.g. ``["ATR-2026-00001"]``).
        max_severity: Highest matched severity, or None when nothing flagged.
    """

    flagged: bool
    rules: List[str]
    max_severity: Optional[str]


def _block_severities(config: Optional[RailsConfig]) -> Set[str]:
    """Read block severities from ``rails.config.atr``, falling back to default.

    An explicitly configured ``block_severities: []`` is honored (it flags
    nothing — useful for monitor-only mode); only an absent/None value falls
    back to ``DEFAULT_BLOCK_SEVERITIES``.
    """
    try:
        atr_config = config.rails.config.atr  # type: ignore[union-attr]
        severities = getattr(atr_config, "block_severities", None)
        if severities is None and hasattr(atr_config, "get"):
            severities = atr_config.get("block_severities")
        if severities is not None:
            return {str(s).lower() for s in severities}
    except (AttributeError, TypeError):
        pass
    return {s.lower() for s in DEFAULT_BLOCK_SEVERITIES}


@action()
async def atr_detection(text: str, config: RailsConfig) -> ATRDetectionResult:
    """Detect AI-agent threats in *text* using Agent Threat Rules.

    Args:
        text: The text to evaluate (typically the user message for an input rail).
        config: The Rails configuration; ``rails.config.atr.block_severities``
            overrides the default ``["critical", "high"]`` block list.

    Returns:
        ATRDetectionResult with the flag, matched rule IDs, and max severity.

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
        return ATRDetectionResult(flagged=False, rules=[], max_severity=None)

    block = _block_severities(config)
    matches = scan(text)  # bundled ATR rules; returns matches sorted by severity
    blocking = [match for match in matches if match.severity.lower() in block]
    if not blocking:
        return ATRDetectionResult(flagged=False, rules=[], max_severity=None)

    rule_ids = [match.rule_id for match in blocking]
    log.info("ATR rail flagged input on rule(s): %s", ", ".join(rule_ids))
    return ATRDetectionResult(flagged=True, rules=rule_ids, max_severity=blocking[0].severity)
