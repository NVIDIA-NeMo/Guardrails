# Command injection via terminal tool

Source: https://example.org/advisories/2026-hermes-terminal-injection

Agents that expose a `terminal` tool without argument constraints are vulnerable
to command injection. An attacker can craft a request that causes the agent to
run `terminal` with a `command` argument containing shell-chaining metacharacters
(`;`, `|`, `&`, backtick, `$()`), reading sensitive files such as `/etc/passwd`
or `/etc/shadow`, or executing destructive operations like `rm -rf`. Because the
agent holds a privileged shell, the injected command runs with the agent's full
OS-level authority.

The mitigation is to require the `command` argument to fully match an allowlisted
pattern that excludes metacharacters and sensitive-file paths. A deny-arg-matching
rule on `command` blocking `/etc/passwd`, `/etc/shadow`, `rm -rf`, and shell
chaining characters closes the most exploited paths.

<!-- params: {"arg_name": "command", "pattern": "/etc/(passwd|shadow)|\\brm\\s+-rf\\b|[;&|`]|\\$\\("} -->
