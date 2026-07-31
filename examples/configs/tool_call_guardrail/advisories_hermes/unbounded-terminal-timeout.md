# Unbounded shell timeout enables runaway terminal commands

Source: https://example.org/advisories/2026-hermes-terminal-timeout

An agent that exposes a `terminal` tool with a loose or absent timeout ceiling on
the `timeout_seconds` argument can be induced to run arbitrarily long-lived
shell commands. An attacker can request a `timeout_seconds` value of 86400
(one day) or higher to keep a background process alive indefinitely, enabling
persistent access or denial-of-service against the sandbox.

The fix is to enforce a strict numeric ceiling on `timeout_seconds` via an
unbounded-arg rule. A ceiling of 300 seconds (five minutes) is appropriate for
interactive agent tasks; batch jobs requiring longer runtimes should use the
`cronjob` tool instead.

<!-- params: {"arg_name": "timeout_seconds", "ceiling": 300} -->
