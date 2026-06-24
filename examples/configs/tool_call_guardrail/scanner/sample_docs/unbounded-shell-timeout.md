# Runaway shell commands via inflated timeouts

Source: https://example.org/advisories/2026-agent-shell-timeout-abuse

A growing class of incidents involves an autonomous agent being steered, through
injected instructions in retrieved content, to call a `run_shell` tool with a
`timeout_seconds` value far larger than any legitimate task needs — keeping a
process alive to drain compute. Capping that one argument at a sane ceiling
bounds the blast radius of a single call.

<!-- params: {"arg_name": "timeout_seconds", "ceiling": 300} -->
