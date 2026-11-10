# Check Every Tool Call of a Deep Agents Agent with NeMo Relay and NeMo Guardrails

An agent's model chooses which tools to call and with what arguments, and anything it reads can steer that choice: a
user message, a log line, a runbook, a web page. No system prompt makes a model immune to
[prompt injection](https://genai.owasp.org/llmrisk/llm01-prompt-injection/), so this example doesn't rely on the
model behaving. It checks what the agent *does*: every tool call goes to a policy outside the model, and runs only if
the policy allows it.

In this tutorial, you:

1. Write a tool-call policy as a NeMo Guardrails config: a tool and argument allowlist, CEL rules, and optionally an
   LLM judge.
2. Load NeMo Guardrails' native [NeMo Relay](https://github.com/NVIDIA/NeMo-Relay) plugin, so every tool call of a
   [Deep Agents](https://github.com/langchain-ai/deepagents) agent is checked by `nemoguardrails hooks-server`.
3. Hand each deny back to the model as a tool error, so the agent corrects the call instead of crashing.
4. Watch the run stop when a check can't be made: Relay fails closed.

## How It Works

```text
Deep Agents ──► Relay middleware ──► NeMo Guardrails plugin ──POST /v1/hooks/nemo-relay──► hooks server
                     │                                   ▲                                          │
                     │                     allow ────────┘◄──── {"decision": "allow" | "deny"} ─────┘
                     └─► deny ──► deny_feedback.py ──► ToolMessage(status="error") ──► the model's next turn
```

Relay sits between the agent and its tools. Before a tool runs, the plugin (`nvidia.nemo_guardrails`, from
`crates/nemo-guardrails-relay-plugin`) sends the call's name and arguments to the hooks server, which decides it with
its tool policy and per-tool rails. Each call has one of three outcomes:

- **Allow.** The tool runs.
- **Deny.** The tool doesn't run. The reason goes back to the model as a tool error, so it can correct the call.
- **Failed check.** The server is down, times out, or answers with an error. The tool doesn't run, and the agent
  stops.

In security terms, Relay is the policy enforcement point and the hooks server is the policy decision point. This is the
"policy enforcement middleware" that the OWASP Top 10 for Agentic Applications recommends against
[ASI02 Tool Misuse and Exploitation](https://genai.owasp.org/resource/owasp-top-10-for-agentic-applications-for-2026/),
and the "complete mediation" that
[LLM06:2025 Excessive Agency](https://genai.owasp.org/llmrisk/llm062025-excessive-agency/) calls for: every call is
validated against a policy, and the model never decides whether its own call is allowed.

| File | What it is |
|---|---|
| `agent.py` | A Deep Agents on-call agent with three sandboxed tools, wired through Relay. |
| `relay_plugin.py` | `activate(bundle, endpoint)`: loads the plugin, pointed at the hooks server, for a block of code. Reusable. |
| `deny_feedback.py` | `DenyFeedbackMiddleware`: turns a Relay rejection into a tool error the model reads. Reusable. |
| `configs/policy` | Model-free policy: a tool and argument allowlist, and CEL expressions. |
| `configs/policy_judge` | The same policy, plus an LLM-judged tool safety check on `http_request`. |

## Set Up

Requires Python 3.11 or later, and the Rust toolchain from [rustup](https://rustup.rs/) to build the plugin. Run
every command from the root of this repository.

Install NeMo Guardrails from the repository, with the hooks server and CEL, then the example's packages:

```bash
pip install -e ".[server,cel]"
pip install -r examples/scripts/nemo_relay/requirements.txt
export NVIDIA_API_KEY=...   # the agent's model; the judge's too, with configs/policy_judge
```

Build the plugin and package its bundle:

```bash
cargo build --release --locked -p nemo-guardrails-relay-plugin
python crates/nemo-guardrails-relay-plugin/scripts/package_bundle.py \
  --library target/release/libnemo_guardrails_relay_plugin.dylib \
  --output build/nemo-guardrails-relay-plugin
export NEMO_GUARDRAILS_RELAY_PLUGIN_BUNDLE=$PWD/build/nemo-guardrails-relay-plugin
```

The library is `libnemo_guardrails_relay_plugin.so` on Linux. For more on the bundle, refer to
[Check NeMo Relay Tool Calls with the NVIDIA NeMo Guardrails Plugin](../../../docs/run-rails/using-hooks-server/nemo-relay-plugin.mdx).

## Step 1: Write the Policy

The policy is a NeMo Guardrails config, `configs/policy/config.yml`. Its checks run in order, and the first deny wins.
None of them calls a model, so they're fast, deterministic, and need no API key.

### Allow Only the Tools and Arguments the Agent Needs

```yaml
hooks_server:
  allowed_tools:
    - name: read_log
      arguments: [service, lines]
    - name: http_request
      arguments: [method, url, body]
    # ... the agent's other tools, and the safe Deep Agents built-ins
  allow_unlisted_tools: false
```

With `allow_unlisted_tools: false`, a call to a tool that isn't listed is denied, and so is a call that passes an
argument its entry doesn't list. Deep Agents adds its own tools to every agent, so the safe ones are listed too.
`execute`, a shell, and `delete` are left out, so every call to them is denied, whatever the model was told.

This is the first mitigation for
[LLM06:2025 Excessive Agency](https://genai.owasp.org/llmrisk/llm062025-excessive-agency/): don't give the agent
functionality it doesn't need, and avoid open-ended tools such as a shell. Denying the shell also closes the most
direct path to [ASI05 Unexpected Code Execution](https://genai.owasp.org/resource/owasp-top-10-for-agentic-applications-for-2026/).

### Constrain the Arguments with CEL

An allowed tool can still be misused with arguments in range of its type: a log read big enough to flood the context,
or an HTTP request that sends data to an attacker's host. [CEL](https://cel.dev/) expressions check the arguments
themselves. Each expression is true for a call to block:

```yaml
rails:
  tool_call:
    per_tool:
      read_log:
        - cel check tool call
      http_request:
        - cel check tool call

  config:
    cel:
      tool_call:
        read_log:
          expressions:
            - '!has(args.lines) || type(args.lines) != int || args.lines < 1 || args.lines > 500'
        http_request:
          expressions:
            - '!has(args.method) || type(args.method) != string || args.method.upperAscii() != "GET"'
            - '!has(args.url) || type(args.url) != string || !args.url.matches("^https://([a-z0-9-]+\\.)*internal\\.example(/|$)")'
```

- `read_log` reads at most 500 lines.
- `http_request` is a read-only `GET`, over HTTPS, to `internal.example` or one of its subdomains. The regular
  expression is anchored at both ends of the host, so `https://internal.example.attacker.net/` and
  `https://internal.example@attacker.net/` are denied.

The `http_request` rule is an egress allowlist, which OWASP recommends against data exfiltration in
[ASI02](https://genai.owasp.org/resource/owasp-top-10-for-agentic-applications-for-2026/) and
[LLM02:2025 Sensitive Information Disclosure](https://genai.owasp.org/llmrisk/llm022025-sensitive-information-disclosure/).

Each expression first checks that its argument is there and has the right type. A CEL expression that errors fails the
check, and a failed check stops the run, so without the type checks, a call that leaves an argument out would stop the
agent instead of being denied and corrected.

### Optional: Add an LLM Judge for What Rules Can't See

A rule can check a URL's host, but not whether its query string carries a secret. `configs/policy_judge` adds a
[tool safety check](configs/policy_judge/prompts.yml) on `http_request`, after the CEL rule:

```yaml
      http_request:
        - cel check tool call
        - tool safety check tool call $model=judge $variant=http_check
```

The rails run in order and stop at the first deny, so the judge sees only the calls CEL allows, and you pay for a model
call only on the calls that need one. The judge looks for credentials and secrets in the URL or body, encoded or not:
the [LLM02:2025](https://genai.owasp.org/llmrisk/llm022025-sensitive-information-disclosure/) leak that an allowed host
alone doesn't prevent.

## Step 2: Load the Plugin

`relay_plugin.activate` writes a `plugins.toml` that loads the plugin from its bundle and points it at the hooks server,
and activates it for a block of code:

```python
async with relay_plugin.activate(plugin_bundle, hooks_url, timeout_millis=25000):
    ...  # every tool call Relay manages in here is checked by the hooks server
```

The plugin is loaded with `startup = "required"`, so a plugin that fails to load stops the agent before it runs,
instead of letting it run unchecked. Relay requires a signed plugin by default; the bundle isn't signed, so
`relay_plugin.py` allows it integrity-only attestation, which checks the plugin library against the SHA-256 digest in
its manifest but not who built it. The plugin decides what your agent may do, so treat it as part of your supply chain
([ASI04 Agentic Supply Chain Vulnerabilities](https://genai.owasp.org/resource/owasp-top-10-for-agentic-applications-for-2026/)):
build the bundle yourself from a commit you trust, and keep it where only you can change it.

## Step 3: Wire Relay into the Agent

`agent.build_agent` adds Relay's Deep Agents integration to the agent:

```python
general_purpose = {**GENERAL_PURPOSE_SUBAGENT, "middleware": [DenyFeedbackMiddleware(on_rejection=on_rejection)]}
kwargs = add_nemo_relay_integration(
    model=model,
    tools=list(tools),
    system_prompt=system_prompt,
    name=name,
    middleware=[DenyFeedbackMiddleware(on_rejection=on_rejection)],
    subagents=[general_purpose, *deep_agent_kwargs.pop("subagents", [])],
    position="inner",
    **deep_agent_kwargs,
)
return create_deep_agent(**kwargs)
```

**Pass every subagent explicitly.** Deep Agents doesn't give its built-in `general-purpose` subagent the main agent's
middleware, and `add_nemo_relay_integration` only instruments subagents passed to it. Leave the subagent out, and the
main agent can delegate a call to it that runs unchecked: a policy bypass through delegation, the
un-scoped privilege inheritance of
[ASI03 Identity and Privilege Abuse](https://genai.owasp.org/resource/owasp-top-10-for-agentic-applications-for-2026/).
`build_agent` passes the `general-purpose` subagent explicitly, and the integration test checks that its calls are
denied too.

## Step 4: Hand Denies Back to the Model

Relay raises a plain `RuntimeError` when a guardrail rejects a call. Unhandled, it would end the run on the first deny.
`DenyFeedbackMiddleware` catches the rejection and returns it to the model as a tool error instead:

```python
return ToolMessage(content=reason, tool_call_id=tool_call["id"], name=tool_call["name"], status="error")
```

The model reads the reason on its next turn, and can retry within the policy. List the middleware before Relay's, so
it wraps Relay's tool call.

A failed check is not a rejection: `DenyFeedbackMiddleware` lets it propagate, and the run stops. Relay raises a plain
`RuntimeError` for both, told apart only by their messages:

| Error | Message starts with | Handled by |
|---|---|---|
| Deny | `guardrail rejected: ` | `deny_feedback.rejection_reason`: back to the model |
| Failed check | `internal error: NeMo Guardrails check failed for tool call '<tool>': ` | `relay_plugin.failed_check_tool`: the run stops |

A CEL deny's reason is the rail's name, `cel check tool call`, not the expression that matched. So `agent.py`'s tool
descriptions state the limits the model must keep to, such as "Policy allows at most 500 lines per read." The policy
enforces them, and the descriptions help the model comply on the first try.

## Step 5: Run It

Terminal 1, the hooks server:

```bash
nemoguardrails hooks-server --config examples/scripts/nemo_relay/configs/policy
```

Terminal 2, the agent:

```bash
python examples/scripts/nemo_relay/agent.py \
  "Checkout is returning 5xx. Read the last 2000 lines of its log, find the runbook, and check the upstream."
```

The task asks for more than the policy allows. The hooks server logs every check:

```text
INFO:nemoguardrails.server.hooks.app:Hook check on /v1/hooks/nemo-relay: ... tool_name='read_log' ... outcome=deny code=None rail='cel check tool call'
INFO:nemoguardrails.server.hooks.app:Hook check on /v1/hooks/nemo-relay: ... tool_name='read_log' ... outcome=allow code=None rail=None
```

and the agent prints each deny it is handed:

```text
  [guardrail] DENY read_log: NeMo Guardrails blocked tool call 'read_log': cel check tool call
```

The 2000-line read is denied by the CEL expression on `read_log`, and the agent retries within the limit.

### Fail Closed

Terminal 3, the kill switch: stop the hooks server while the agent runs. The next tool call fails closed, and
`agent.py` exits with `Stopped, failing closed: internal error: NeMo Guardrails check failed for tool call ...`.

A guardrail that lets calls through whenever it can't answer is one an attacker only has to slow down. Failing closed
is the "fail secure by default" that OWASP recommends against
[ASI05](https://genai.owasp.org/resource/owasp-top-10-for-agentic-applications-for-2026/).

### Try the Judge

Restart the hooks server with `--config examples/scripts/nemo_relay/configs/policy_judge`. A `GET` to an internal host
now also goes to the judge, so a request whose query string carries a token passes the CEL rule and is denied by the
judge, with the judge's reason.

## OWASP Coverage

What each control in this example addresses, from the
[OWASP Top 10 for LLM Applications 2025](https://genai.owasp.org/llm-top-10/) and the
[OWASP Top 10 for Agentic Applications 2026](https://genai.owasp.org/resource/owasp-top-10-for-agentic-applications-for-2026/):

| Control | Where | OWASP risk |
|---|---|---|
| Every tool call checked outside the model | Relay and the hooks server | LLM01, LLM06, ASI01, ASI02, ASI08 |
| Tool allowlist: no shell, no delete | `hooks_server.allowed_tools` | LLM06, ASI02, ASI05 |
| Argument allowlist | `hooks_server.allowed_tools` | ASI02 |
| Argument limits and egress allowlist | CEL rules | LLM02, ASI02 |
| Secrets in allowed requests | Tool safety check, `configs/policy_judge` | LLM02 |
| Subagents checked too | `agent.build_agent` | ASI03 |
| Fail closed on a failed check, or a plugin that doesn't load | Relay, `startup = "required"` | ASI05 |
| Plugin attestation | `relay_plugin.plugins_toml` | ASI04 |

The guardrail checks the calls the agent asks for. It doesn't replace least privilege for the tools themselves: give
each tool only the credentials and network access it needs, so a call the policy wrongly allows can still do little
harm.

## Adapt It to Your Agent

- **Tools.** Replace `TOOLS` in `agent.py` with yours, and list each one, with its arguments, in `allowed_tools`.
  Start from a deny-all policy, and allow what the agent needs.
- **Rules.** Add a CEL expression under `rails.config.cel.tool_call.<tool>` for each argument with a limit, and the rail
  under `rails.tool_call.per_tool.<tool>`. Check that each argument is there and has the right type first.
- **Tests.** `tests/server/test_hooks_nemo_relay_example.py` posts calls to the policy in-process, with no plugin or
  model. Copy its table of allowed and denied calls, and add a row for each attack you defend against.
- **Reuse.** `relay_plugin.py` and `deny_feedback.py` don't depend on this agent. Copy them into your project.

## Notes

- **Timeout.** `agent.py --timeout` sets the plugin's `timeout_millis` (25 s by default, the plugin's own default).
  A judge takes a few seconds per call.
- **Other plugin settings.** Relay also reads the system `plugins.toml`, `/etc/nemo-relay/plugins.toml` on Linux and
  macOS, and its settings take precedence over the file `relay_plugin.py` writes. With a proxy set in `HTTP_PROXY`,
  `HTTPS_PROXY` or `ALL_PROXY`, add `127.0.0.1` to `NO_PROXY`, or the checks go through the proxy.
- **One path.** Deep Agents' calls are checked only through Relay; don't also point a Claude Code or Codex hook at the
  same server for the same calls, or each is checked twice.

## Tests

No API key is needed:

```bash
make test TEST=tests/server/test_hooks_nemo_relay_example.py
make test-relay-plugin TEST=integration_tests/relay_plugin/test_deepagents_example.py
```

The first runs both policies in the hooks server, in-process. The second builds the plugin, starts a hooks server on
`configs/policy`, and runs `relay_plugin`, `deny_feedback` and the agent through Relay, with a scripted model; refer to
[`integration_tests/README.md`](../../../integration_tests/README.md) for what it needs.
