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

Evaluates user messages against the bundled ATR rule set to detect
agent-specific threats: prompt injection, jailbreak, tool poisoning,
MCP attacks, and skill compromise.

ATR is an open, MIT-licensed detection standard.  The rule set is
distributed as the ``pyatr`` PyPI package and runs locally with no API
key or network call required.

.. code:: bash

    pip install pyatr

When the package is not installed the action raises ``ImportError``
with a helpful hint.
"""

import logging
from typing import Any, FrozenSet, List, Optional, Set, TypedDict

from nemoguardrails import RailsConfig
from nemoguardrails.actions import action

log = logging.getLogger(__name__)

# Lazy import — pyatr is an optional dependency.
_pyatr_module = None
_ATREngine = None
_AgentEvent = None
try:
    import pyatr as _pyatr_module

    _ATREngine = _pyatr_module.ATREngine
    _AgentEvent = _pyatr_module.AgentEvent
except ImportError:
    _pyatr_module = None
except AttributeError:
    log.warning(
        "pyatr is installed but does not expose the expected API "
        "(ATREngine, AgentEvent). ATR detection will be unavailable."
    )
    _pyatr_module = None

# Module-level cache for ATREngine — the rule bundle is loaded once
# per process lifetime.
_cached_engine = None

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_SEVERITIES: FrozenSet[str] = frozenset({"critical", "high"})
"""Default severities that are blocked when no explicit list is configured."""

VALID_SEVERITIES: FrozenSet[str] = frozenset({"critical", "high", "medium", "low"})
"""All valid ATR severity levels (lowercase)."""


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


class ATRDetectionResult(TypedDict):
    """Result returned by the ``atr_detection`` action."""

    is_threat: bool
    """``True`` when at least one ATR rule matched at or above the configured
    severity threshold."""

    text: str
    """The original text that was evaluated (unchanged)."""

    detections: List[str]
    """Matched ATR rule IDs, e.g. ``['ATR-2026-001', 'ATR-2026-042']``."""


# ---------------------------------------------------------------------------
# Helpers — config reading
# ---------------------------------------------------------------------------

# Depending on whether ``atr_detection`` is registered as a proper Pydantic
# model field or comes in as an ``extra="allow"`` field, the object may be a
# model instance or a plain dict.  The helpers below normalise access so
# that both cases work transparently.


def _get_attr(obj: Any, name: str, default: Any = None) -> Any:
    """Return *name* from *obj*, supporting both dicts and objects."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _atr_config_raw(config: RailsConfig) -> Optional[Any]:
    """Return the raw ``atr_detection`` config value, or ``None``."""
    return getattr(config.rails.config, "atr_detection", None)


# ---------------------------------------------------------------------------
# Validation & extraction
# ---------------------------------------------------------------------------


def _check_pyatr_available() -> None:
    """Raise ``ImportError`` with an install hint when *pyatr* is absent."""
    if _ATREngine is None:
        raise ImportError("The pyatr module is required for ATR detection. Please install it using: pip install pyatr")


def _validate_atr_config(config: RailsConfig) -> None:
    """Validate the ``atr_detection`` section of the rails config.

    Args:
        config: The active rails configuration object.

    Raises:
        ValueError: If configured severity values are invalid.
    """
    atr_config = _atr_config_raw(config)

    if atr_config is None:
        return

    severities = _get_attr(atr_config, "severities", None)
    if severities is None:
        return  # defaults will be applied by the extractor

    if not isinstance(severities, (list, tuple)):
        msg = f"atr_detection.severities must be a list, got {type(severities)!r}"
        log.error(msg)
        raise ValueError(msg)

    for sev in severities:
        if not isinstance(sev, str):
            msg = f"Invalid severity entry {sev!r} in atr_detection config: expected a string."
            log.error(msg)
            raise ValueError(msg)
        if sev.lower() not in VALID_SEVERITIES:
            msg = (
                f"Invalid severity '{sev}' in atr_detection config. "
                f"Valid severities: {', '.join(sorted(VALID_SEVERITIES))}."
            )
            log.error(msg)
            raise ValueError(msg)


def _extract_atr_config(config: RailsConfig) -> FrozenSet[str]:
    """Extract the set of severity levels that should be flagged.

    Args:
        config: The active rails configuration object.

    Returns:
        Lowercase severity strings from the config, defaulting to
        ``{"critical", "high"}`` when none are specified.
    """
    atr_config = _atr_config_raw(config)

    if atr_config is None:
        return DEFAULT_SEVERITIES

    severities = _get_attr(atr_config, "severities", None)

    if severities is None:
        return DEFAULT_SEVERITIES

    return frozenset({s.lower() for s in severities})


# ---------------------------------------------------------------------------
# Core detection
# ---------------------------------------------------------------------------


def _evaluate_atr(
    text: str,
    atr_engine,
    severities: Set[str],
) -> ATRDetectionResult:
    """Evaluate *text* against a loaded ATR engine.

    Args:
        text: The user-message text to scan.
        atr_engine: An ``ATREngine`` instance with rules already loaded.
        severities: Severity levels to alert on.

    Returns:
        An ``ATRDetectionResult`` containing the threat flag and match details.
    """
    if not text:
        return ATRDetectionResult(is_threat=False, text=text, detections=[])

    event = _AgentEvent(content=text, event_type="llm_input")
    matches = atr_engine.evaluate(event)

    detections: List[str] = []
    for match in matches:
        match_severity = match.severity.lower()
        if match_severity in severities:
            detections.append(match.rule_id)
            log.info(
                "ATR match: rule=%s severity=%s title=%s",
                match.rule_id,
                match_severity,
                match.title,
            )

    if detections:
        log.info(
            "ATR detection flagged %d rule(s): %s",
            len(detections),
            ", ".join(detections),
        )
        return ATRDetectionResult(is_threat=True, text=text, detections=detections)

    return ATRDetectionResult(is_threat=False, text=text, detections=[])


# ---------------------------------------------------------------------------
# Public action
# ---------------------------------------------------------------------------


@action()
async def atr_detection(
    text: str,
    config: RailsConfig,
) -> ATRDetectionResult:
    """Evaluate *text* against the bundled Agent Threat Rules (ATR).

    Uses the local ``pyatr`` engine (no API key required) to scan for
    prompt injection, jailbreak attempts, tool poisoning, MCP attacks,
    skill compromise, and other agent-specific threats.

    The severity threshold is read from
    ``config.rails.config.atr_detection.severities`` and defaults to
    ``["critical", "high"]``.

    The ``ATREngine`` instance is cached at module level so the rule
    bundle is only loaded once per process lifetime.

    Args:
        text: The user message to evaluate.
        config: The rails configuration object.

    Returns:
        ``ATRDetectionResult`` containing:
          - ``is_threat``: ``True`` if a threat was detected.
          - ``text``: The original text (unchanged).
          - ``detections``: Matched ATR rule IDs.

    Raises:
        ImportError: If ``pyatr`` is not installed.
        ValueError: If configured severity values are invalid.
    """
    global _cached_engine

    _check_pyatr_available()

    _validate_atr_config(config)

    severities = _extract_atr_config(config)

    if _cached_engine is None:
        _cached_engine = _ATREngine()

    return _evaluate_atr(text, _cached_engine, severities)
