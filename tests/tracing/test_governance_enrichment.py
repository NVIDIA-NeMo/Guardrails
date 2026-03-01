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

"""Unit tests for governance trace enrichment.

Tests cover:
- :class:`~nemoguardrails.tracing.governance_enricher.GovernanceTraceEnricher`
  (all public methods)
- :func:`~nemoguardrails.tracing.governance_enricher._compute_risk_score`
- Semantic conventions defined in
  :mod:`~nemoguardrails.tracing.governance_conventions`
"""

import unittest
from unittest.mock import MagicMock

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
from nemoguardrails.tracing.governance_enricher import (
    GovernanceTraceEnricher,
    _compute_risk_score,
    _max_severity,
    _span_is_recording,
)


def _mock_span(is_recording: bool = True) -> MagicMock:
    """Return a MagicMock that behaves like a recording OTel span."""
    span = MagicMock()
    span.is_recording.return_value = is_recording
    return span


class TestGovernanceConventions(unittest.TestCase):
    """Smoke-tests for the constants in governance_conventions.py."""

    def test_governance_decision_values(self):
        self.assertEqual(GovernanceDecisions.ALLOW, "allow")
        self.assertEqual(GovernanceDecisions.DENY, "deny")
        self.assertEqual(GovernanceDecisions.MODIFY, "modify")
        self.assertEqual(GovernanceDecisions.ESCALATE, "escalate")

    def test_security_threat_type_values(self):
        self.assertEqual(SecurityThreatTypes.PROMPT_INJECTION, "prompt_injection")
        self.assertEqual(SecurityThreatTypes.JAILBREAK, "jailbreak")
        self.assertEqual(SecurityThreatTypes.PII, "pii")

    def test_security_severity_ordering(self):
        """Severity constants must be distinct strings."""
        severities = [
            SecuritySeverity.LOW,
            SecuritySeverity.MEDIUM,
            SecuritySeverity.HIGH,
            SecuritySeverity.CRITICAL,
        ]
        self.assertEqual(len(set(severities)), 4)

    def test_escalation_priority_values(self):
        self.assertEqual(EscalationPriority.LOW, "low")
        self.assertEqual(EscalationPriority.URGENT, "urgent")

    def test_attribute_namespaces(self):
        """All attribute keys must use the expected namespace prefix."""
        for key, attr in vars(GovernanceAttributes).items():
            if isinstance(attr, str) and not key.startswith("_"):
                self.assertTrue(attr.startswith("guardrails.governance."), attr)

        for key, attr in vars(SecurityAttributes).items():
            if isinstance(attr, str) and not key.startswith("_"):
                self.assertTrue(attr.startswith("guardrails.security."), attr)

        for key, attr in vars(ComplianceAttributes).items():
            if isinstance(attr, str) and not key.startswith("_"):
                self.assertTrue(attr.startswith("guardrails.compliance."), attr)

        for key, attr in vars(RiskAttributes).items():
            if isinstance(attr, str) and not key.startswith("_"):
                self.assertTrue(attr.startswith("guardrails.risk."), attr)

    def test_event_names_namespace(self):
        for key, attr in vars(GovernanceEventNames).items():
            if isinstance(attr, str) and not key.startswith("_"):
                self.assertTrue(attr.startswith("guardrails."), attr)

    def test_pii_attribute_keys_present(self):
        self.assertTrue(hasattr(PIIAttributes, "ENTITY_TYPES"))
        self.assertTrue(hasattr(PIIAttributes, "SOURCE"))
        self.assertTrue(hasattr(PIIAttributes, "ANONYMISED"))
        self.assertTrue(hasattr(PIIAttributes, "DETECTION_ENGINE"))


class TestHelpers(unittest.TestCase):
    """Unit tests for internal helper functions."""

    def test_span_is_recording_true(self):
        span = _mock_span(is_recording=True)
        self.assertTrue(_span_is_recording(span))

    def test_span_is_recording_false(self):
        span = _mock_span(is_recording=False)
        self.assertFalse(_span_is_recording(span))

    def test_span_is_recording_none(self):
        self.assertFalse(_span_is_recording(None))  # type: ignore[arg-type]

    def test_max_severity_both_none_like(self):
        self.assertEqual(_max_severity(None, SecuritySeverity.LOW), SecuritySeverity.LOW)

    def test_max_severity_a_wins(self):
        self.assertEqual(
            _max_severity(SecuritySeverity.HIGH, SecuritySeverity.LOW),
            SecuritySeverity.HIGH,
        )

    def test_max_severity_b_wins(self):
        self.assertEqual(
            _max_severity(SecuritySeverity.LOW, SecuritySeverity.CRITICAL),
            SecuritySeverity.CRITICAL,
        )

    def test_max_severity_equal(self):
        self.assertEqual(
            _max_severity(SecuritySeverity.MEDIUM, SecuritySeverity.MEDIUM),
            SecuritySeverity.MEDIUM,
        )


class TestComputeRiskScore(unittest.TestCase):
    """Unit tests for the risk score computation."""

    def test_zero_violations_zero_rails(self):
        score = _compute_risk_score(violation_count=0, highest_severity=None, rails_evaluated=0)
        self.assertEqual(score, 0.0)

    def test_zero_violations_with_rails(self):
        score = _compute_risk_score(violation_count=0, highest_severity=None, rails_evaluated=5)
        self.assertEqual(score, 0.0)

    def test_all_rails_failed_low_severity(self):
        # density = 1.0, severity_weight = 0.25 (LOW = 1/4)
        score = _compute_risk_score(
            violation_count=5,
            highest_severity=SecuritySeverity.LOW,
            rails_evaluated=5,
        )
        expected = round(0.6 * 1.0 + 0.4 * 0.25, 4)  # 0.7
        self.assertAlmostEqual(score, expected, places=4)

    def test_critical_severity_one_violation(self):
        # density = 0.2, severity_weight = 1.0 (CRITICAL = 4/4)
        score = _compute_risk_score(
            violation_count=1,
            highest_severity=SecuritySeverity.CRITICAL,
            rails_evaluated=5,
        )
        expected = round(0.6 * 0.2 + 0.4 * 1.0, 4)  # 0.52
        self.assertAlmostEqual(score, expected, places=4)

    def test_score_capped_at_one(self):
        score = _compute_risk_score(
            violation_count=100,
            highest_severity=SecuritySeverity.CRITICAL,
            rails_evaluated=1,
        )
        self.assertLessEqual(score, 1.0)

    def test_score_in_range(self):
        for v in range(6):
            for rails in range(1, 6):
                score = _compute_risk_score(v, SecuritySeverity.HIGH, rails)
                self.assertGreaterEqual(score, 0.0)
                self.assertLessEqual(score, 1.0)

    def test_unknown_severity_treated_as_zero(self):
        score_unknown = _compute_risk_score(
            violation_count=1,
            highest_severity="unknown_severity",
            rails_evaluated=5,
        )
        score_none = _compute_risk_score(
            violation_count=1,
            highest_severity=None,
            rails_evaluated=5,
        )
        # Both should treat severity weight as 0
        self.assertAlmostEqual(score_unknown, score_none, places=4)


class TestRecordGovernanceDecision(unittest.TestCase):
    """Tests for GovernanceTraceEnricher.record_governance_decision."""

    def setUp(self):
        self.enricher = GovernanceTraceEnricher(domain="insurance")

    def test_deny_sets_attributes_and_event(self):
        span = _mock_span()
        self.enricher.record_governance_decision(
            span=span,
            decision=GovernanceDecisions.DENY,
            rule_id="no_competitor_mention",
            reason="Competitor detected",
            category="brand_protection",
            position="input",
            confidence=0.95,
        )

        span.set_attribute.assert_any_call(GovernanceAttributes.DECISION, GovernanceDecisions.DENY)
        span.set_attribute.assert_any_call(GovernanceAttributes.RULE_ID, "no_competitor_mention")
        span.set_attribute.assert_any_call(GovernanceAttributes.REASON, "Competitor detected")
        span.set_attribute.assert_any_call(GovernanceAttributes.CATEGORY, "brand_protection")
        span.set_attribute.assert_any_call(GovernanceAttributes.POSITION, "input")
        span.set_attribute.assert_any_call(GovernanceAttributes.CONFIDENCE, 0.95)

        span.add_event.assert_called_once()
        event_call = span.add_event.call_args
        self.assertEqual(event_call.kwargs["name"], GovernanceEventNames.GOVERNANCE_DECISION)
        self.assertEqual(event_call.kwargs["attributes"]["decision"], GovernanceDecisions.DENY)
        self.assertEqual(event_call.kwargs["attributes"]["rule_id"], "no_competitor_mention")

    def test_allow_does_not_increment_violation_count(self):
        span = _mock_span()
        self.enricher.record_governance_decision(span=span, decision=GovernanceDecisions.ALLOW)
        self.assertEqual(self.enricher._violation_count, 0)

    def test_deny_increments_violation_count(self):
        span = _mock_span()
        self.enricher.record_governance_decision(span=span, decision=GovernanceDecisions.DENY)
        self.assertEqual(self.enricher._violation_count, 1)

    def test_modify_increments_violation_count(self):
        span = _mock_span()
        self.enricher.record_governance_decision(span=span, decision=GovernanceDecisions.MODIFY)
        self.assertEqual(self.enricher._violation_count, 1)

    def test_escalate_increments_violation_count(self):
        span = _mock_span()
        self.enricher.record_governance_decision(span=span, decision=GovernanceDecisions.ESCALATE)
        self.assertEqual(self.enricher._violation_count, 1)

    def test_optional_fields_omitted_when_none(self):
        span = _mock_span()
        self.enricher.record_governance_decision(span=span, decision=GovernanceDecisions.ALLOW)

        set_keys = [c.args[0] for c in span.set_attribute.call_args_list]
        self.assertNotIn(GovernanceAttributes.RULE_ID, set_keys)
        self.assertNotIn(GovernanceAttributes.REASON, set_keys)
        self.assertNotIn(GovernanceAttributes.CATEGORY, set_keys)
        self.assertNotIn(GovernanceAttributes.POSITION, set_keys)
        self.assertNotIn(GovernanceAttributes.CONFIDENCE, set_keys)

    def test_non_recording_span_is_skipped(self):
        span = _mock_span(is_recording=False)
        self.enricher.record_governance_decision(span=span, decision=GovernanceDecisions.DENY)
        span.set_attribute.assert_not_called()
        span.add_event.assert_not_called()

    def test_multiple_decisions_accumulate_violations(self):
        span = _mock_span()
        self.enricher.record_governance_decision(span=span, decision=GovernanceDecisions.DENY)
        self.enricher.record_governance_decision(span=span, decision=GovernanceDecisions.DENY)
        self.enricher.record_governance_decision(span=span, decision=GovernanceDecisions.ALLOW)
        self.assertEqual(self.enricher._violation_count, 2)


class TestRecordSecurityThreat(unittest.TestCase):
    """Tests for GovernanceTraceEnricher.record_security_threat."""

    def setUp(self):
        self.enricher = GovernanceTraceEnricher(domain="finance")

    def test_sets_attributes_and_event(self):
        span = _mock_span()
        self.enricher.record_security_threat(
            span=span,
            threat_type=SecurityThreatTypes.JAILBREAK,
            severity=SecuritySeverity.HIGH,
            detection_method="llm_classifier",
            detection_confidence=0.88,
        )

        span.set_attribute.assert_any_call(SecurityAttributes.THREAT_DETECTED, True)
        span.set_attribute.assert_any_call(SecurityAttributes.THREAT_TYPE, SecurityThreatTypes.JAILBREAK)
        span.set_attribute.assert_any_call(SecurityAttributes.SEVERITY, SecuritySeverity.HIGH)
        span.set_attribute.assert_any_call(SecurityAttributes.DETECTION_METHOD, "llm_classifier")
        span.set_attribute.assert_any_call(SecurityAttributes.DETECTION_CONFIDENCE, 0.88)

        span.add_event.assert_called_once()
        event_attrs = span.add_event.call_args.kwargs["attributes"]
        self.assertEqual(event_attrs["threat_type"], SecurityThreatTypes.JAILBREAK)
        self.assertEqual(event_attrs["severity"], SecuritySeverity.HIGH)
        self.assertEqual(event_attrs["detection_method"], "llm_classifier")
        self.assertAlmostEqual(event_attrs["detection_confidence"], 0.88)

    def test_increments_violation_count(self):
        span = _mock_span()
        self.enricher.record_security_threat(
            span=span,
            threat_type=SecurityThreatTypes.PROMPT_INJECTION,
            severity=SecuritySeverity.MEDIUM,
        )
        self.assertEqual(self.enricher._violation_count, 1)

    def test_tracks_highest_severity(self):
        span = _mock_span()
        self.enricher.record_security_threat(span=span, threat_type="pii", severity=SecuritySeverity.LOW)
        self.assertEqual(self.enricher._highest_severity, SecuritySeverity.LOW)

        self.enricher.record_security_threat(span=span, threat_type="jailbreak", severity=SecuritySeverity.CRITICAL)
        self.assertEqual(self.enricher._highest_severity, SecuritySeverity.CRITICAL)

        # A lower severity should NOT replace the highest
        self.enricher.record_security_threat(span=span, threat_type="pii", severity=SecuritySeverity.MEDIUM)
        self.assertEqual(self.enricher._highest_severity, SecuritySeverity.CRITICAL)

    def test_optional_fields_omitted_when_none(self):
        span = _mock_span()
        self.enricher.record_security_threat(
            span=span,
            threat_type=SecurityThreatTypes.PII,
            severity=SecuritySeverity.LOW,
        )
        set_keys = [c.args[0] for c in span.set_attribute.call_args_list]
        self.assertNotIn(SecurityAttributes.DETECTION_METHOD, set_keys)
        self.assertNotIn(SecurityAttributes.DETECTION_CONFIDENCE, set_keys)

    def test_non_recording_span_is_skipped(self):
        span = _mock_span(is_recording=False)
        self.enricher.record_security_threat(
            span=span,
            threat_type=SecurityThreatTypes.JAILBREAK,
            severity=SecuritySeverity.HIGH,
        )
        span.set_attribute.assert_not_called()
        span.add_event.assert_not_called()
        self.assertEqual(self.enricher._violation_count, 0)


class TestRecordEscalation(unittest.TestCase):
    """Tests for GovernanceTraceEnricher.record_escalation."""

    def setUp(self):
        self.enricher = GovernanceTraceEnricher(domain="healthcare")

    def test_sets_attributes_and_event_with_queue(self):
        span = _mock_span()
        self.enricher.record_escalation(
            span=span,
            reason="High-severity jailbreak attempt",
            priority=EscalationPriority.URGENT,
            queue="security-ops",
        )

        span.set_attribute.assert_any_call(EscalationAttributes.TRIGGERED, True)
        span.set_attribute.assert_any_call(EscalationAttributes.REASON, "High-severity jailbreak attempt")
        span.set_attribute.assert_any_call(EscalationAttributes.PRIORITY, EscalationPriority.URGENT)
        span.set_attribute.assert_any_call(EscalationAttributes.QUEUE, "security-ops")

        span.add_event.assert_called_once()
        event_call = span.add_event.call_args
        self.assertEqual(event_call.kwargs["name"], GovernanceEventNames.ESCALATION_TRIGGERED)
        self.assertEqual(event_call.kwargs["attributes"]["reason"], "High-severity jailbreak attempt")
        self.assertEqual(event_call.kwargs["attributes"]["priority"], EscalationPriority.URGENT)
        self.assertEqual(event_call.kwargs["attributes"]["queue"], "security-ops")

    def test_default_priority_is_medium(self):
        span = _mock_span()
        self.enricher.record_escalation(span=span, reason="Uncertain intent")

        span.set_attribute.assert_any_call(EscalationAttributes.PRIORITY, EscalationPriority.MEDIUM)

    def test_queue_omitted_when_not_provided(self):
        span = _mock_span()
        self.enricher.record_escalation(span=span, reason="Review needed")
        set_keys = [c.args[0] for c in span.set_attribute.call_args_list]
        self.assertNotIn(EscalationAttributes.QUEUE, set_keys)

    def test_non_recording_span_is_skipped(self):
        span = _mock_span(is_recording=False)
        self.enricher.record_escalation(span=span, reason="Test")
        span.set_attribute.assert_not_called()
        span.add_event.assert_not_called()


class TestRecordPiiDetection(unittest.TestCase):
    """Tests for GovernanceTraceEnricher.record_pii_detection."""

    def setUp(self):
        self.enricher = GovernanceTraceEnricher(domain="insurance")

    def test_emits_event_with_entity_types(self):
        span = _mock_span()
        self.enricher.record_pii_detection(
            span=span,
            entity_types=["PERSON", "EMAIL_ADDRESS"],
            source="user_message",
            anonymised=True,
            detection_engine="presidio",
        )

        span.add_event.assert_called_once()
        event_call = span.add_event.call_args
        self.assertEqual(event_call.kwargs["name"], GovernanceEventNames.PII_DETECTED)
        attrs = event_call.kwargs["attributes"]
        self.assertEqual(attrs[PIIAttributes.ENTITY_TYPES], "PERSON,EMAIL_ADDRESS")
        self.assertEqual(attrs[PIIAttributes.SOURCE], "user_message")
        self.assertTrue(attrs[PIIAttributes.ANONYMISED])
        self.assertEqual(attrs[PIIAttributes.DETECTION_ENGINE], "presidio")

    def test_entity_types_joined_as_csv(self):
        span = _mock_span()
        self.enricher.record_pii_detection(
            span=span,
            entity_types=["PHONE_NUMBER", "CREDIT_CARD", "SSN"],
        )
        attrs = span.add_event.call_args.kwargs["attributes"]
        self.assertEqual(attrs[PIIAttributes.ENTITY_TYPES], "PHONE_NUMBER,CREDIT_CARD,SSN")

    def test_increments_violation_count(self):
        span = _mock_span()
        self.enricher.record_pii_detection(span=span, entity_types=["PERSON"])
        self.assertEqual(self.enricher._violation_count, 1)

    def test_detection_engine_omitted_when_not_provided(self):
        span = _mock_span()
        self.enricher.record_pii_detection(span=span, entity_types=["EMAIL_ADDRESS"])
        attrs = span.add_event.call_args.kwargs["attributes"]
        self.assertNotIn(PIIAttributes.DETECTION_ENGINE, attrs)

    def test_non_recording_span_is_skipped(self):
        span = _mock_span(is_recording=False)
        self.enricher.record_pii_detection(span=span, entity_types=["PERSON"])
        span.add_event.assert_not_called()
        self.assertEqual(self.enricher._violation_count, 0)

    def test_default_source_is_user_message(self):
        span = _mock_span()
        self.enricher.record_pii_detection(span=span, entity_types=["PERSON"])
        attrs = span.add_event.call_args.kwargs["attributes"]
        self.assertEqual(attrs[PIIAttributes.SOURCE], "user_message")

    def test_default_anonymised_is_false(self):
        span = _mock_span()
        self.enricher.record_pii_detection(span=span, entity_types=["EMAIL_ADDRESS"])
        attrs = span.add_event.call_args.kwargs["attributes"]
        self.assertFalse(attrs[PIIAttributes.ANONYMISED])


class TestFinaliseComplianceSummary(unittest.TestCase):
    """Tests for GovernanceTraceEnricher.finalise_compliance_summary."""

    def setUp(self):
        self.enricher = GovernanceTraceEnricher(domain="finance")

    def test_writes_all_compliance_attributes(self):
        span = _mock_span()
        self.enricher.finalise_compliance_summary(
            span=span,
            rails_evaluated=6,
            rails_passed=5,
            rails_failed=1,
            failed_rail_names=["check_jailbreak"],
        )

        span.set_attribute.assert_any_call(ComplianceAttributes.DOMAIN, "finance")
        span.set_attribute.assert_any_call(ComplianceAttributes.AUDIT_COMPLETE, True)
        span.set_attribute.assert_any_call(ComplianceAttributes.RAILS_EVALUATED, 6)
        span.set_attribute.assert_any_call(ComplianceAttributes.RAILS_PASSED, 5)
        span.set_attribute.assert_any_call(ComplianceAttributes.RAILS_FAILED, 1)
        span.set_attribute.assert_any_call(ComplianceAttributes.FAILED_RAIL_NAMES, "check_jailbreak")

    def test_risk_score_set_on_span(self):
        span = _mock_span()
        # Seed a security violation first to get a non-zero score
        threat_span = _mock_span()
        self.enricher.record_security_threat(
            span=threat_span,
            threat_type=SecurityThreatTypes.JAILBREAK,
            severity=SecuritySeverity.HIGH,
        )
        self.enricher.finalise_compliance_summary(span=span, rails_evaluated=5, rails_passed=4, rails_failed=1)

        set_calls = {c.args[0]: c.args[1] for c in span.set_attribute.call_args_list}
        self.assertIn(RiskAttributes.RISK_SCORE, set_calls)
        self.assertIn(RiskAttributes.VIOLATION_COUNT, set_calls)
        self.assertIn(RiskAttributes.VIOLATION_SEVERITY, set_calls)

        risk_score = set_calls[RiskAttributes.RISK_SCORE]
        self.assertGreaterEqual(risk_score, 0.0)
        self.assertLessEqual(risk_score, 1.0)
        self.assertEqual(set_calls[RiskAttributes.VIOLATION_COUNT], 1)
        self.assertEqual(set_calls[RiskAttributes.VIOLATION_SEVERITY], SecuritySeverity.HIGH)

    def test_no_violation_severity_when_no_threats(self):
        span = _mock_span()
        self.enricher.finalise_compliance_summary(span=span, rails_evaluated=5, rails_passed=5, rails_failed=0)
        set_keys = [c.args[0] for c in span.set_attribute.call_args_list]
        self.assertNotIn(RiskAttributes.VIOLATION_SEVERITY, set_keys)

    def test_failed_rail_names_joined_as_csv(self):
        span = _mock_span()
        self.enricher.finalise_compliance_summary(
            span=span,
            rails_evaluated=5,
            rails_passed=3,
            rails_failed=2,
            failed_rail_names=["check_jailbreak", "check_pii"],
        )
        span.set_attribute.assert_any_call(ComplianceAttributes.FAILED_RAIL_NAMES, "check_jailbreak,check_pii")

    def test_failed_rail_names_omitted_when_empty(self):
        span = _mock_span()
        self.enricher.finalise_compliance_summary(
            span=span,
            rails_evaluated=5,
            rails_passed=5,
            rails_failed=0,
            failed_rail_names=[],
        )
        set_keys = [c.args[0] for c in span.set_attribute.call_args_list]
        self.assertNotIn(ComplianceAttributes.FAILED_RAIL_NAMES, set_keys)

    def test_failed_rail_names_omitted_when_none(self):
        span = _mock_span()
        self.enricher.finalise_compliance_summary(
            span=span,
            rails_evaluated=5,
            rails_passed=5,
            rails_failed=0,
        )
        set_keys = [c.args[0] for c in span.set_attribute.call_args_list]
        self.assertNotIn(ComplianceAttributes.FAILED_RAIL_NAMES, set_keys)

    def test_non_recording_span_is_skipped(self):
        span = _mock_span(is_recording=False)
        self.enricher.finalise_compliance_summary(span=span, rails_evaluated=5, rails_passed=5, rails_failed=0)
        span.set_attribute.assert_not_called()


class TestEnricherEndToEnd(unittest.TestCase):
    """Integration-style tests that exercise multiple enricher methods together."""

    def test_full_denied_interaction_with_jailbreak(self):
        """Simulate a complete interaction where a jailbreak is detected and denied."""
        enricher = GovernanceTraceEnricher(domain="insurance")
        root_span = _mock_span()
        rail_span = _mock_span()

        # 1. Security threat detected on the input rail span
        enricher.record_security_threat(
            span=rail_span,
            threat_type=SecurityThreatTypes.JAILBREAK,
            severity=SecuritySeverity.CRITICAL,
            detection_method="yara",
            detection_confidence=0.99,
        )

        # 2. Governance decision recorded on the same rail span
        enricher.record_governance_decision(
            span=rail_span,
            decision=GovernanceDecisions.DENY,
            rule_id="no_jailbreak",
            reason="YARA rule matched jailbreak pattern",
            category="content_safety",
            position="input",
            confidence=0.99,
        )

        # 3. Escalation recorded on root span
        enricher.record_escalation(
            span=root_span,
            reason="Critical jailbreak attempt — requires human review",
            priority=EscalationPriority.URGENT,
            queue="security-ops",
        )

        # 4. Finalise compliance summary on root span
        enricher.finalise_compliance_summary(
            span=root_span,
            rails_evaluated=3,
            rails_passed=2,
            rails_failed=1,
            failed_rail_names=["check_jailbreak"],
        )

        # Assertions on internal state
        self.assertEqual(enricher._violation_count, 2)  # threat + deny
        self.assertEqual(enricher._highest_severity, SecuritySeverity.CRITICAL)

        # Root span must have compliance and risk attributes set
        root_set_calls = {c.args[0]: c.args[1] for c in root_span.set_attribute.call_args_list}
        self.assertEqual(root_set_calls[ComplianceAttributes.DOMAIN], "insurance")
        self.assertTrue(root_set_calls[ComplianceAttributes.AUDIT_COMPLETE])
        self.assertEqual(root_set_calls[ComplianceAttributes.RAILS_EVALUATED], 3)
        self.assertEqual(root_set_calls[RiskAttributes.VIOLATION_COUNT], 2)
        self.assertEqual(root_set_calls[RiskAttributes.VIOLATION_SEVERITY], SecuritySeverity.CRITICAL)
        self.assertGreater(root_set_calls[RiskAttributes.RISK_SCORE], 0.5)

        # Root span must have escalation attributes
        self.assertEqual(root_set_calls[EscalationAttributes.TRIGGERED], True)
        self.assertEqual(root_set_calls[EscalationAttributes.PRIORITY], EscalationPriority.URGENT)

    def test_pii_detection_then_compliance_summary(self):
        """Simulate PII detection followed by compliance summary on clean rails."""
        enricher = GovernanceTraceEnricher(domain="healthcare")
        root_span = _mock_span()
        pii_span = _mock_span()

        enricher.record_pii_detection(
            span=pii_span,
            entity_types=["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER"],
            source="user_message",
            anonymised=True,
            detection_engine="presidio",
        )

        # No governance denial — just anonymisation; still counts as a violation
        enricher.finalise_compliance_summary(
            span=root_span,
            rails_evaluated=4,
            rails_passed=4,
            rails_failed=0,
        )

        root_set_calls = {c.args[0]: c.args[1] for c in root_span.set_attribute.call_args_list}
        self.assertEqual(root_set_calls[ComplianceAttributes.RAILS_FAILED], 0)
        self.assertEqual(root_set_calls[RiskAttributes.VIOLATION_COUNT], 1)
        # No severity tracked for PII alone (no security threat recorded)
        self.assertNotIn(RiskAttributes.VIOLATION_SEVERITY, root_set_calls)

    def test_enricher_domain_passed_through(self):
        enricher = GovernanceTraceEnricher(domain="custom-domain")
        span = _mock_span()
        enricher.finalise_compliance_summary(span=span, rails_evaluated=1, rails_passed=1, rails_failed=0)
        span.set_attribute.assert_any_call(ComplianceAttributes.DOMAIN, "custom-domain")

    def test_timestamps_are_integers(self):
        """All add_event calls must pass integer nanosecond timestamps."""
        enricher = GovernanceTraceEnricher(domain="finance")
        span = _mock_span()

        enricher.record_governance_decision(span=span, decision=GovernanceDecisions.DENY)
        enricher.record_security_threat(span=span, threat_type="jailbreak", severity=SecuritySeverity.HIGH)
        enricher.record_escalation(span=span, reason="Test")
        enricher.record_pii_detection(span=span, entity_types=["PERSON"])

        for event_call in span.add_event.call_args_list:
            ts = event_call.kwargs.get("timestamp")
            self.assertIsNotNone(ts, "timestamp keyword arg missing from add_event call")
            self.assertIsInstance(ts, int, f"Expected int timestamp, got {type(ts)}")
            self.assertGreater(ts, 0)


if __name__ == "__main__":
    unittest.main()
