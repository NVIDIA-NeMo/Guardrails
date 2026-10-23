# Agent Hooks Examples

These configurations are for the Guardrails hooks server, `nemoguardrails hooks-server`, which checks a coding agent's tool calls before they run.
The agent's `PreToolUse` hook sends each call to the server, which applies the configuration's tool policy and per-tool rails and answers with allow or deny.
For the full guide, refer to [Use the Guardrails Hooks Server](./../../../docs/run-rails/using-hooks-server/index.mdx).

There is one configuration for each agent:

| Folder | Agent | Hook Path | Minimum Agent Version |
| --- | --- | --- | --- |
| `claude_code` | Claude Code | `/v1/hooks/claude-code` | 2.1.295, for `onFailure: "block"` |
| `codex` | Codex | `/v1/hooks/codex` | 0.124.0; from 0.129.0, Codex runs a hook only after you trust it |
| `nemo_relay` | A NeMo Relay plugin's tool guardrail | `/v1/hooks/nemo-relay` | NeMo Relay 0.10.1 |

Each `config.yml` has `models: []`, a `hooks_server` section that lists the agent's tools with `allow_unlisted_tools: false`, and regular-expression rails under `rails.tool_call.per_tool`.
None calls a model, so none needs an API key.
The patterns illustrate a policy structure and do not provide complete protection.
Regular expressions can be bypassed, so combine them with the agent's permission settings and sandboxing.

## Run an Example

Install the library with the `server` extra, then start the hooks server from the repository root:

```bash
pip install "nemoguardrails[server]"
nemoguardrails hooks-server --config examples/configs/agent_hooks/claude_code
```

The server listens on `http://127.0.0.1:8100`.
Send it a call the way Claude Code's hook would:

```bash
curl -s http://localhost:8100/v1/hooks/claude-code \
  -H "Content-Type: application/json" \
  -d '{"hook_event_name": "PreToolUse", "tool_name": "Bash", "tool_input": {"command": "git push --force origin main"}, "tool_use_id": "toolu_01"}'
```

The example's `Bash` rail blocks the force push:

```json
{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"deny","permissionDecisionReason":"NeMo Guardrails: regex check tool call"}}
```

To point the agent's hook at the server, follow the agent's page: [Claude Code](./../../../docs/run-rails/using-hooks-server/claude-code.mdx), [Codex](./../../../docs/run-rails/using-hooks-server/codex.mdx), or [NeMo Relay](./../../../docs/run-rails/using-hooks-server/nemo-relay.mdx).

## One Server for Each Agent

A hooks server applies its one configuration to every call, whichever agent path it arrives on, and both Claude Code and Codex report shell commands as `Bash`.
So do not merge these configurations: run one server for each agent, each on its own port:

```bash
nemoguardrails hooks-server --config examples/configs/agent_hooks/claude_code --port 8100
nemoguardrails hooks-server --config examples/configs/agent_hooks/codex --port 8101
```

Then point each agent's hook at its own server's port.
