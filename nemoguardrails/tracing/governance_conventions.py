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

"""Custom OpenTelemetry semantic conventions for governance metadata in NeMo Guardrails.

These conventions extend the existing ``guardrails.*`` OTel namespace with
sub-namespaces for governance decision tracking, security threat detection,
escalation events, and compliance auditing.

They are designed for use in regulated industries (insurance, finance, healthcare)
where compliance teams need to monitor policy enforcement using standard
observability tools such as Datadog, Grafana, and Jaeger.

Namespace layout
----------------
``guardrails.governance.*``  – policy enforcement decisions
``guardrails.security.*``    – threat detection (prompt injection, jailbreak, PII)
``guardrails.compliance.*``  – audit summary metrics
``guardrails.risk.*``        – risk and confidence scoring

Event names
-----------
``guardrails.security.threat_detected``  – span event for a detected threat
``guardrails.escalation.triggered``      – span event when human-in-loop fires
``guardrails.pii.detected``             – span event for PII detection
"""


class GovernanceAttributes:
    """Span attributes for policy enforcement decisions.

    Set these on the span that represents the rail or action that
    produced a policy decision.

    Example::

        span.set_attribute(GovernanceAttributes.DECISION, "deny")
        span.set_attribute(GovernanceAttributes.RULE_ID, "no_competitor_mention")
        span.set_attribute(GovernanceAttributes.REASON, "Competitor product detected in user message")
    """

    # The enforcement outcome.  One of: allow | deny | modify | escalate
    DECISION = "guardrails.governance.decision"

    # Identifier of the policy rule that fired (e.g. "no_competitor_mention")
    RULE_ID = "guardrails.governance.rule_id"

    # Human-readable reason for the decision
    REASON = "guardrails.governance.reason"

    # High-level policy category (e.g. "content_safety", "data_privacy",
    # "compliance", "brand_protection")
    CATEGORY = "guardrails.governance.category"

    # Rail position in the pipeline: "input" | "output" | "dialog"
    POSITION = "guardrails.governance.position"

    # Confidence score for the decision in [0.0, 1.0]
    CONFIDENCE = "guardrails.governance.confidence"


class GovernanceDecisions:
    """Allowed values for :attr:`GovernanceAttributes.DECISION`."""

    ALLOW = "allow"
    DENY = "deny"
    MODIFY = "modify"
    ESCALATE = "escalate"


class SecurityAttributes:
    """Span attributes for security threat detection.

    Attach these to the root interaction span or the span for the rail
    that detected the threat.  Detailed per-event data should be added
    via :meth:`GovernanceTraceEnricher.record_security_threat`.

    Example::

        span.set_attribute(SecurityAttributes.THREAT_DETECTED, True)
        span.set_attribute(SecurityAttributes.THREAT_TYPE, SecurityThreatTypes.PROMPT_INJECTION)
        span.set_attribute(SecurityAttributes.SEVERITY, SecuritySeverity.HIGH)
    """

    # Whether any security threat was detected during this interaction
    THREAT_DETECTED = "guardrails.security.threat_detected"

    # The type of threat detected (see :class:`SecurityThreatTypes`)
    THREAT_TYPE = "guardrails.security.threat_type"

    # Severity level (see :class:`SecuritySeverity`)
    SEVERITY = "guardrails.security.severity"

    # Detection method used (e.g. "regex", "llm_classifier", "yara", "presidio")
    DETECTION_METHOD = "guardrails.security.detection_method"

    # Confidence score for the threat detection in [0.0, 1.0]
    DETECTION_CONFIDENCE = "guardrails.security.detection_confidence"


class SecurityThreatTypes:
    """Allowed values for :attr:`SecurityAttributes.THREAT_TYPE`."""

    PROMPT_INJECTION = "prompt_injection"
    JAILBREAK = "jailbreak"
    PII = "pii"
    TOXIC_CONTENT = "toxic_content"
    COMPETITOR_MENTION = "competitor_mention"
    OFF_TOPIC = "off_topic"
    HALLUCINATION = "hallucination"


class SecuritySeverity:
    """Ordered severity levels for :attr:`SecurityAttributes.SEVERITY`.

    Values are intentionally strings so they appear as human-readable
    labels in observability dashboards.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class EscalationAttributes:
    """Span attributes for human-in-the-loop escalation events.

    These are set on the root interaction span when an escalation is
    triggered.  A corresponding span *event* is also added via
    :meth:`GovernanceTraceEnricher.record_escalation`.

    Example::

        span.set_attribute(EscalationAttributes.TRIGGERED, True)
        span.set_attribute(EscalationAttributes.REASON, "High-severity jailbreak attempt")
        span.set_attribute(EscalationAttributes.PRIORITY, EscalationPriority.HIGH)
    """

    # Whether this interaction was escalated to a human reviewer
    TRIGGERED = "guardrails.escalation.triggered"

    # Human-readable reason for the escalation
    REASON = "guardrails.escalation.reason"

    # Escalation priority level (see :class:`EscalationPriority`)
    PRIORITY = "guardrails.escalation.priority"

    # Queue or team the escalation was routed to
    QUEUE = "guardrails.escalation.queue"


class EscalationPriority:
    """Allowed values for :attr:`EscalationAttributes.PRIORITY`."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class ComplianceAttributes:
    """Span attributes for the compliance audit summary.

    These are set on the root interaction span by
    :meth:`GovernanceTraceEnricher.finalise_compliance_summary` after all
    rails have run.

    Example::

        span.set_attribute(ComplianceAttributes.DOMAIN, "insurance")
        span.set_attribute(ComplianceAttributes.RAILS_EVALUATED, 5)
        span.set_attribute(ComplianceAttributes.RAILS_PASSED, 4)
        span.set_attribute(ComplianceAttributes.RAILS_FAILED, 1)
        span.set_attribute(ComplianceAttributes.AUDIT_COMPLETE, True)
    """

    # Industry / regulatory domain (e.g. "insurance", "finance", "healthcare")
    DOMAIN = "guardrails.compliance.domain"

    # Whether the audit record for this interaction is complete
    AUDIT_COMPLETE = "guardrails.compliance.audit_complete"

    # Total number of rails evaluated
    RAILS_EVALUATED = "guardrails.compliance.rails_evaluated"

    # Number of rails that passed (did not stop execution)
    RAILS_PASSED = "guardrails.compliance.rails_passed"

    # Number of rails that failed (stopped or modified execution)
    RAILS_FAILED = "guardrails.compliance.rails_failed"

    # Comma-separated list of failed rail names (low cardinality in practice)
    FAILED_RAIL_NAMES = "guardrails.compliance.failed_rail_names"


class RiskAttributes:
    """Span attributes for risk and confidence scoring on the root span.

    Example::

        span.set_attribute(RiskAttributes.RISK_SCORE, 0.82)
        span.set_attribute(RiskAttributes.VIOLATION_COUNT, 3)
        span.set_attribute(RiskAttributes.VIOLATION_SEVERITY, SecuritySeverity.HIGH)
    """

    # Aggregate risk score for the interaction in [0.0, 1.0]
    RISK_SCORE = "guardrails.risk.risk_score"

    # Total number of policy / security violations detected
    VIOLATION_COUNT = "guardrails.risk.violation_count"

    # Highest severity level among all violations detected
    VIOLATION_SEVERITY = "guardrails.risk.violation_severity"


# ---------------------------------------------------------------------------
# Governance event names
# ---------------------------------------------------------------------------


class GovernanceEventNames:
    """OTel span event names emitted by :class:`GovernanceTraceEnricher`.

    Each event carries structured attributes describing the specific
    incident.  Use these names to filter events in your observability
    backend.
    """

    # Emitted once per detected security threat
    SECURITY_THREAT_DETECTED = "guardrails.security.threat_detected"

    # Emitted when human-in-the-loop escalation is triggered
    ESCALATION_TRIGGERED = "guardrails.escalation.triggered"

    # Emitted when PII is detected in the user message or bot response
    PII_DETECTED = "guardrails.pii.detected"

    # Emitted when a governance decision (allow/deny/modify/escalate) is recorded
    GOVERNANCE_DECISION = "guardrails.governance.decision"


class PIIAttributes:
    """Attribute keys used inside ``guardrails.pii.detected`` span events."""

    # Comma-separated list of PII entity types detected
    # (e.g. "PERSON,EMAIL_ADDRESS,PHONE_NUMBER")
    ENTITY_TYPES = "pii.entity_types"

    # Source of the PII: "user_message" | "bot_response"
    SOURCE = "pii.source"

    # Whether the PII was anonymised / redacted before forwarding
    ANONYMISED = "pii.anonymised"

    # Detection engine used (e.g. "presidio", "regex", "gliner")
    DETECTION_ENGINE = "pii.detection_engine"
