# ATR-inspired threat detection example

This example shows how to use the built-in `regex_detection` input rail
with a small set of patterns inspired by Agent Threat Rules, an open
detection standard for AI agent threats published under the MIT license:

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
`regex check input` flow and the bot will respond with the library default
refusal message defined in `nemoguardrails/library/regex/flows.v1.co`
(`"I'm sorry, I can't respond to that."`). Benign messages are forwarded
to the configured main model.

The `config.yml` lists `openai`/`gpt-4o-mini` as the main model so that
chat runs end-to-end. Replace with your preferred provider; the input
rail blocks threats before the model is invoked, so the model only sees
benign inputs.

## Extending

To run against the live ATR YAML ruleset, parse the rule files at startup
and append the `detection.regex_patterns` field of each rule to the
`patterns` list under `regex_detection.input`.

To surface the matched rule id (rather than only refusing), add a custom
flow that calls `detect_regex_pattern` directly and emits a custom event:

```colang
define bot refuse atr_threat
  "I'm sorry, that request was blocked by an ATR input safety rule."

define flow atr report match
  $result = execute detect_regex_pattern(source="input", text=$user_message)
  if $result["is_match"]
    $matched_rules = $result["detections"]
    create event AtrRuleMatchedRailException(message="ATR input rail blocked")
    bot refuse atr_threat
    stop
```

Then wire `atr report match` instead of `regex check input` under
`rails.input.flows`. The custom flow uses a non-conflicting bot utterance
(`bot refuse atr_threat`) so it does not collide with the library default,
and emits a `AtrRuleMatchedRailException` event that downstream observers
(audit logging, metrics) can subscribe to without parsing the refusal
text.
