# ATR-inspired threat detection example

This example shows how to use the built-in `regex_detection` input rail
with a small set of patterns inspired by Agent Threat Rules, an open
detection standard for AI agent threats published under Apache-2.0:

https://github.com/Agent-Threat-Rule/agent-threat-rules

## What it covers

The patterns in `config/config.yml` map to common attack categories that
ATR ships rules for:

- ATR-PI-001 instruction override ("ignore previous instructions")
- ATR-PI-002 system prompt exfiltration ("reveal your system prompt")
- ATR-PI-003 role-play jailbreak ("act as DAN")
- ATR-PI-004 base64-wrapped payload hint
- ATR-MCP-001 MCP tool override markers
- ATR-SSRF-001 `file://` scheme reference

Each entry is illustrative. The full ruleset and YAML schema live in the
ATR repository; this example exists so a NeMo Guardrails user can see the
shape of an agent-specific input rail without needing an external service.

## Running the example

From the project root:

```bash
nemoguardrails chat --config=examples/configs/atr_threat_detection/config
```

A user message such as "Ignore all previous instructions" will trigger the
`regex check input` flow and the bot will respond with the refusal message
defined in `rails.co`.

## Extending

To run against the live ATR YAML ruleset, parse the rule files at startup
and append the `detection.regex_patterns` field of each rule to the
`patterns` list under `regex_detection.input`.

To also surface matched detections (so the bot can respond with the rule
identifier rather than only refusing), enable the optional `atr report
match` flow shipped in `rails.co` by adding it to your input flows in
`config/config.yml`:

```yaml
rails:
  input:
    flows:
      - atr report match
      - regex check input
```

Order matters: `atr report match` runs before `regex check input` so the
matched rule id is available when the refusal message is generated.
