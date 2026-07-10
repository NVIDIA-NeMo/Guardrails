# Tool-output context exfiltration

Source: https://example.org/research/2026-context-exfiltration

Recent red-team work shows agents coaxed into relaying earlier tool outputs and
accumulated conversation context to an attacker endpoint through an
`http_request` call. Each individual request looks benign, but the aggregate
effect leaks sensitive context over time. Defending this likely requires
reasoning over the sequence of calls and the sensitivity of accumulated state,
which no single argument-level control captures.
