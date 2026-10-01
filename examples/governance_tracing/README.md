# Governance Tracing Example

This example shows how to use NeMo Guardrails'
[OpenTelemetry adapter](../../nemoguardrails/tracing/adapters/opentelemetry.py)
together with the new
[`GovernanceTraceEnricher`](../../nemoguardrails/tracing/governance_enricher.py)
to add governance metadata to every interaction trace.

Enriched traces carry span attributes and events for:

| Signal | OTel namespace |
|---|---|
| Policy enforcement decisions (allow / deny / modify / escalate) | `guardrails.governance.*` |
| Security threat detection (jailbreak, PII, prompt injection) | `guardrails.security.*` |
| Human-in-the-loop escalation | `guardrails.escalation.*` |
| Compliance audit summary | `guardrails.compliance.*` |
| Risk / confidence scoring | `guardrails.risk.*` |

This is particularly useful for regulated industries (insurance, finance,
healthcare) where compliance teams need an audit trail of every policy
decision, exportable to Datadog, Grafana, or Jaeger.

---

## Prerequisites

```bash
pip install "nemoguardrails[tracing]" \
            opentelemetry-sdk \
            opentelemetry-exporter-otlp \
            openai
```

## Start Jaeger (all-in-one)

```bash
docker run -d --name jaeger \
    -p 6831:6831/udp \
    -p 16686:16686 \
    -p 4317:4317 \
    jaegertracing/all-in-one:latest
```

The Jaeger UI will be available at <http://localhost:16686>.

## Set your LLM API key

```bash
export OPENAI_API_KEY="sk-..."
```

## Run the example

```bash
python governance_tracing_example.py
```

The script runs three interaction scenarios:

| Scenario | What it demonstrates |
|---|---|
| `clean_query` | A clean customer query — all rails pass, `allow` decision recorded |
| `pii_in_message` | PII detected and anonymised — `modify` decision + `guardrails.pii.detected` event |
| `jailbreak_attempt` | Jailbreak pattern detected — `deny` decision + security threat + escalation |

After the script finishes, open <http://localhost:16686>, search for service
`nemo_guardrails_governance_demo`, and inspect the waterfall traces.

---

## Architecture

```
Application layer
│
├── app_tracer.start_as_current_span("demo.interaction.*")   ← root span
│       │
│       ├── GovernanceTraceEnricher.record_security_threat()
│       ├── GovernanceTraceEnricher.record_governance_decision()
│       ├── GovernanceTraceEnricher.record_pii_detection()
│       ├── GovernanceTraceEnricher.record_escalation()
│       └── GovernanceTraceEnricher.finalise_compliance_summary()
│
└── NeMo Guardrails (creates its own child spans)
        ├── guardrails.request           ← SERVER span
        │   ├── guardrails.rail          ← INTERNAL span (check_jailbreak)
        │   ├── guardrails.rail          ← INTERNAL span (check_input_sensitive_data)
        │   └── chat gpt-4o-mini         ← CLIENT span (LLM call)
```

The `GovernanceTraceEnricher` enriches the **application root span** with
aggregate governance metadata.  Individual rail spans are created automatically
by the `OpenTelemetryAdapter` inside NeMo Guardrails.

---

## Integrating into your own application

```python
from nemoguardrails.tracing.governance_enricher import GovernanceTraceEnricher
from nemoguardrails.tracing.governance_conventions import (
    GovernanceDecisions, SecurityThreatTypes, SecuritySeverity,
)

# One enricher per interaction
enricher = GovernanceTraceEnricher(domain="insurance")

# After a rail produces a decision:
enricher.record_governance_decision(
    span=current_span,
    decision=GovernanceDecisions.DENY,
    rule_id="no_competitor_mention",
    reason="Competitor product detected",
    category="brand_protection",
    position="input",
    confidence=0.97,
)

# After all rails have run, write the compliance summary to the root span:
enricher.finalise_compliance_summary(
    span=root_span,
    rails_evaluated=5,
    rails_passed=4,
    rails_failed=1,
    failed_rail_names=["check_jailbreak"],
)
```

---

## Querying in Grafana / Datadog

Once traces are flowing into your backend you can write queries like:

**All denied interactions in the last hour:**
```
guardrails.governance.decision = "deny"
```

**High-severity security threats:**
```
guardrails.security.severity IN ["high", "critical"]
```

**Interactions that required human review:**
```
guardrails.escalation.triggered = true
```

**Compliance dashboard — failed rail rate:**
```
sum(guardrails.compliance.rails_failed) / sum(guardrails.compliance.rails_evaluated)
```
