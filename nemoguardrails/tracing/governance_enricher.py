# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Governance trace enricher for NeMo Guardrails.

This module provides :class:`GovernanceTraceEnricher`, a helper that adds
governance-relevant metadata to OpenTelemetry spans produced by NeMo Guardrails.

Typical usage in an instrumented application::

    from opentelemetry import trace

    from nemoguardrails.tracing.governance_enricher import GovernanceTraceEnricher
    from nemoguardrails.tracing.governance_conventions import (
        GovernanceDecisions,
        SecurityThreatTypes,
        SecuritySeverity,
    )

    enricher = GovernanceTraceEnricher(domain="insurance")

    # After a rail produces a decision, enrich the active span:
    span = trace.get_current_span()
    enricher.record_governance_decision(
        span=span,
        decision=GovernanceDecisions.DENY,
        rule_id="no_competitor_mention",
        reason="Competitor product detected",
        category="brand_protection",
        position="input",
        confidence=0.97,
    )

    # If the rail also detected a security threat:
    enricher.record_security_threat(
        span=span,
        threat_type=SecurityThreatTypes.PROMPT_INJECTION,
        severity=SecuritySeverity.HIGH,
        detection_method="llm_classifier",
        detection_confidence=0.91,
    )

    # On the root interaction span, finalise the compliance summary:
    enricher.finalise_compliance_summary(
        span=root_span,
        rails_evaluated=5,
        rails_passed=4,
        rails_failed=1,
        failed_rail_names=["check_jailbreak"],
    )
"""

from __future__ import annotations

import logging
import time
from typing import List, Optional, Sequence

logger = logging.getLogger(__name__)

try:
    from opentelemetry import trace  # type: ignore
    from opentelemetry.trace import Span  # type: ignore
except ImportError:
    raise ImportError(
        "OpenTelemetry API is not installed. "
        "Install NeMo Guardrails with tracing support: "
        "`pip install nemoguardrails[tracing]` "
        "or install the API directly: `pip install opentelemetry-api`."
    )

from nemoguardrails.tracing.governance_conventions import (
    ComplianceAttributes,
    EscalationAttributes,
    EscalationPriority,
    GovernanceAttributes,
    GovernanceDecisions,
    GovernanceEventNames,
    PIIAttributes,
    RiskAttributes,
    SecurityAttributes,
    SecuritySeverity,
    SecurityThreatTypes,
)

# Map severity strings to a numeric weight for comparison
_SEVERITY_ORDER = {
    SecuritySeverity.LOW: 1,
    SecuritySeverity.MEDIUM: 2,
    SecuritySeverity.HIGH: 3,
    SecuritySeverity.CRITICAL: 4,
}


def _max_severity(a: Optional[str], b: str) -> str:
    """Return the higher of two severity strings."""
    if a is None:
        return b
    return a if _SEVERITY_ORDER.get(a, 0) >= _SEVERITY_ORDER.get(b, 0) else b


class GovernanceTraceEnricher:
    """Enriches OpenTelemetry spans with governance metadata.

    An enricher instance is typically created once per ``LLMRails`` instance
    (or per request) and reused for all spans produced during that interaction.

    Parameters
    ----------
    domain:
        The regulatory / industry domain for this deployment
        (e.g. ``"insurance"``, ``"finance"``, ``"healthcare"``).
        Written to every compliance summary via
        :attr:`~governance_conventions.ComplianceAttributes.DOMAIN`.
    """

    def __init__(self, domain: str = "general") -> None:
        self._domain = domain

        # Internal counters — updated by the record_* methods and
        # flushed to the root span by finalise_compliance_summary().
        self._violation_count: int = 0
        self._highest_severity: Optional[str] = None

    # ------------------------------------------------------------------
    # Public enrichment methods
    # ------------------------------------------------------------------

    def record_governance_decision(
        self,
        span: Span,
        decision: str,
        rule_id: Optional[str] = None,
        reason: Optional[str] = None,
        category: Optional[str] = None,
        position: Optional[str] = None,
        confidence: Optional[float] = None,
    ) -> None:
        """Record a policy enforcement decision on *span*.

        Sets span attributes and adds a ``guardrails.governance.decision``
        event so that the decision is visible both as filterable metadata
        and as a time-stamped event in the trace waterfall.

        Parameters
        ----------
        span:
            The OTel span to enrich.  Usually the span for the rail that
            made the decision.
        decision:
            The enforcement outcome.  Use a constant from
            :class:`~governance_conventions.GovernanceDecisions`
            (``"allow"``, ``"deny"``, ``"modify"``, or ``"escalate"``).
        rule_id:
            Identifier of the policy rule that fired
            (e.g. ``"no_competitor_mention"``).
        reason:
            Human-readable explanation of why this decision was made.
        category:
            High-level policy category
            (e.g. ``"content_safety"``, ``"data_privacy"``).
        position:
            Rail position: ``"input"``, ``"output"``, or ``"dialog"``.
        confidence:
            Decision confidence in *[0.0, 1.0]*.
        """
        if not _span_is_recording(span):
            return

        span.set_attribute(GovernanceAttributes.DECISION, decision)

        if rule_id is not None:
            span.set_attribute(GovernanceAttributes.RULE_ID, rule_id)
        if reason is not None:
            span.set_attribute(GovernanceAttributes.REASON, reason)
        if category is not None:
            span.set_attribute(GovernanceAttributes.CATEGORY, category)
        if position is not None:
            span.set_attribute(GovernanceAttributes.POSITION, position)
        if confidence is not None:
            span.set_attribute(GovernanceAttributes.CONFIDENCE, confidence)

        # Add a time-stamped event for waterfall visibility
        event_attrs: dict = {"decision": decision}
        if rule_id is not None:
            event_attrs["rule_id"] = rule_id
        if reason is not None:
            event_attrs["reason"] = reason
        if category is not None:
            event_attrs["category"] = category

        span.add_event(
            name=GovernanceEventNames.GOVERNANCE_DECISION,
            attributes=event_attrs,
            timestamp=_now_ns(),
        )

        # Track violations for the compliance summary
        if decision in (
            GovernanceDecisions.DENY,
            GovernanceDecisions.MODIFY,
            GovernanceDecisions.ESCALATE,
        ):
            self._violation_count += 1

    def record_security_threat(
        self,
        span: Span,
        threat_type: str,
        severity: str,
        detection_method: Optional[str] = None,
        detection_confidence: Optional[float] = None,
    ) -> None:
        """Record a security threat detection as a span event.

        Sets high-level security attributes on *span* and adds a
        ``guardrails.security.threat_detected`` event with full detail.

        Parameters
        ----------
        span:
            The OTel span to enrich.
        threat_type:
            Type of threat.  Use a constant from
            :class:`~governance_conventions.SecurityThreatTypes`
            (e.g. ``"prompt_injection"``, ``"jailbreak"``, ``"pii"``).
        severity:
            Severity level.  Use a constant from
            :class:`~governance_conventions.SecuritySeverity`
            (``"low"``, ``"medium"``, ``"high"``, or ``"critical"``).
        detection_method:
            The method used to detect the threat
            (e.g. ``"llm_classifier"``, ``"regex"``, ``"yara"``).
        detection_confidence:
            Detection confidence in *[0.0, 1.0]*.
        """
        if not _span_is_recording(span):
            return

        # Span-level summary attributes
        span.set_attribute(SecurityAttributes.THREAT_DETECTED, True)
        span.set_attribute(SecurityAttributes.THREAT_TYPE, threat_type)
        span.set_attribute(SecurityAttributes.SEVERITY, severity)

        if detection_method is not None:
            span.set_attribute(SecurityAttributes.DETECTION_METHOD, detection_method)
        if detection_confidence is not None:
            span.set_attribute(
                SecurityAttributes.DETECTION_CONFIDENCE, detection_confidence
            )

        # Detailed event for the trace waterfall
        event_attrs: dict = {
            "threat_type": threat_type,
            "severity": severity,
        }
        if detection_method is not None:
            event_attrs["detection_method"] = detection_method
        if detection_confidence is not None:
            event_attrs["detection_confidence"] = detection_confidence

        span.add_event(
            name=GovernanceEventNames.SECURITY_THREAT_DETECTED,
            attributes=event_attrs,
            timestamp=_now_ns(),
        )

        # Update internal running totals
        self._violation_count += 1
        self._highest_severity = _max_severity(self._highest_severity, severity)

    def record_escalation(
        self,
        span: Span,
        reason: str,
        priority: str = EscalationPriority.MEDIUM,
        queue: Optional[str] = None,
    ) -> None:
        """Record a human-in-the-loop escalation event on *span*.

        Sets escalation attributes on *span* and emits a
        ``guardrails.escalation.triggered`` event.

        Parameters
        ----------
        span:
            The OTel span to enrich (usually the root interaction span).
        reason:
            Human-readable explanation of why the escalation was triggered.
        priority:
            Escalation urgency.  Use a constant from
            :class:`~governance_conventions.EscalationPriority`
            (``"low"``, ``"medium"``, ``"high"``, or ``"urgent"``).
        queue:
            Optional routing queue or team name
            (e.g. ``"compliance-review"``, ``"security-ops"``).
        """
        if not _span_is_recording(span):
            return

        span.set_attribute(EscalationAttributes.TRIGGERED, True)
        span.set_attribute(EscalationAttributes.REASON, reason)
        span.set_attribute(EscalationAttributes.PRIORITY, priority)

        if queue is not None:
            span.set_attribute(EscalationAttributes.QUEUE, queue)

        event_attrs: dict = {
            "reason": reason,
            "priority": priority,
        }
        if queue is not None:
            event_attrs["queue"] = queue

        span.add_event(
            name=GovernanceEventNames.ESCALATION_TRIGGERED,
            attributes=event_attrs,
            timestamp=_now_ns(),
        )

    def record_pii_detection(
        self,
        span: Span,
        entity_types: Sequence[str],
        source: str = "user_message",
        anonymised: bool = False,
        detection_engine: Optional[str] = None,
    ) -> None:
        """Record a PII detection event on *span*.

        Emits a ``guardrails.pii.detected`` span event.  Does *not* log
        the actual PII values — only the entity types detected.

        Parameters
        ----------
        span:
            The OTel span to enrich.
        entity_types:
            List of PII entity type labels that were detected
            (e.g. ``["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER"]``).
        source:
            Where the PII was found: ``"user_message"`` or ``"bot_response"``.
        anonymised:
            Whether the PII was redacted / anonymised before forwarding.
        detection_engine:
            The engine that performed detection
            (e.g. ``"presidio"``, ``"regex"``, ``"gliner"``).
        """
        if not _span_is_recording(span):
            return

        entity_types_str = ",".join(entity_types)

        event_attrs: dict = {
            PIIAttributes.ENTITY_TYPES: entity_types_str,
            PIIAttributes.SOURCE: source,
            PIIAttributes.ANONYMISED: anonymised,
        }
        if detection_engine is not None:
            event_attrs[PIIAttributes.DETECTION_ENGINE] = detection_engine

        span.add_event(
            name=GovernanceEventNames.PII_DETECTED,
            attributes=event_attrs,
            timestamp=_now_ns(),
        )

        # PII counts as a violation
        self._violation_count += 1

    def finalise_compliance_summary(
        self,
        span: Span,
        rails_evaluated: int,
        rails_passed: int,
        rails_failed: int,
        failed_rail_names: Optional[List[str]] = None,
    ) -> None:
        """Write the compliance audit summary to the root interaction *span*.

        This should be called once, after all rails have run, on the
        *root* (``guardrails.request``) span.  It writes aggregate counts
        and the domain, and sets risk attributes derived from the violations
        accumulated since this enricher was created.

        Parameters
        ----------
        span:
            The root OTel span for the interaction.
        rails_evaluated:
            Total number of rails that were evaluated.
        rails_passed:
            Number of rails that completed without stopping execution.
        rails_failed:
            Number of rails that stopped or modified execution.
        failed_rail_names:
            Optional list of rail names that failed, for drill-down queries.
        """
        if not _span_is_recording(span):
            return

        # Compliance attributes
        span.set_attribute(ComplianceAttributes.DOMAIN, self._domain)
        span.set_attribute(ComplianceAttributes.AUDIT_COMPLETE, True)
        span.set_attribute(ComplianceAttributes.RAILS_EVALUATED, rails_evaluated)
        span.set_attribute(ComplianceAttributes.RAILS_PASSED, rails_passed)
        span.set_attribute(ComplianceAttributes.RAILS_FAILED, rails_failed)

        if failed_rail_names:
            span.set_attribute(
                ComplianceAttributes.FAILED_RAIL_NAMES, ",".join(failed_rail_names)
            )

        # Risk attributes derived from accumulated state
        risk_score = _compute_risk_score(
            violation_count=self._violation_count,
            highest_severity=self._highest_severity,
            rails_evaluated=rails_evaluated,
        )
        span.set_attribute(RiskAttributes.RISK_SCORE, risk_score)
        span.set_attribute(RiskAttributes.VIOLATION_COUNT, self._violation_count)

        if self._highest_severity is not None:
            span.set_attribute(
                RiskAttributes.VIOLATION_SEVERITY, self._highest_severity
            )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _span_is_recording(span: Span) -> bool:
    """Return ``True`` if *span* is an active, recording span."""
    return span is not None and span.is_recording()


def _now_ns() -> int:
    """Return the current time in nanoseconds (monotonic wall clock)."""
    return time.time_ns()


def _compute_risk_score(
    violation_count: int,
    highest_severity: Optional[str],
    rails_evaluated: int,
) -> float:
    """Compute a normalised risk score in *[0.0, 1.0]*.

    The score blends two signals:

    * **Violation density** — fraction of evaluated rails that fired.
    * **Severity weight** — the numeric weight of the highest severity seen.

    Parameters
    ----------
    violation_count:
        Total accumulated violations across all ``record_*`` calls.
    highest_severity:
        The highest severity string seen, or ``None`` if no threats detected.
    rails_evaluated:
        Total rails evaluated (denominator for density).

    Returns
    -------
    float
        Risk score in *[0.0, 1.0]*, rounded to 4 decimal places.
    """
    if rails_evaluated == 0 and violation_count == 0:
        return 0.0

    # Violation density (capped at 1.0)
    denominator = max(rails_evaluated, 1)
    density = min(violation_count / denominator, 1.0)

    # Severity weight mapped to [0.0, 1.0]
    max_weight = max(_SEVERITY_ORDER.values())  # 4 for CRITICAL
    severity_weight = _SEVERITY_ORDER.get(highest_severity or "", 0) / max_weight

    # Weighted blend: 60 % density + 40 % severity
    score = 0.6 * density + 0.4 * severity_weight
    return round(min(score, 1.0), 4)
