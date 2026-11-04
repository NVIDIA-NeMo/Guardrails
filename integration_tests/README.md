# Integration Tests

The suites here run NeMo Guardrails against the real agents and clients that call it, with fake models behind them, so no test calls a live model or needs an API key.
Each suite installs those programs from its own hash-pinned `requirements.txt`, so none of them is a Guardrails dependency, and each has its own Make target.

`make test` doesn't run these suites, since each needs programs from outside Guardrails' environment: they sit outside the `testpaths` in `pytest.ini`.
They also sit outside `tests/`, so they don't inherit the main suite's fixtures from `tests/conftest.py`.
The hooks suite runs in `.venv` and the Relay suite in `.venv-relay-plugin`, so run each through its own target: `pytest integration_tests` fails at collection.
CI runs both suites on Ubuntu, with the same targets, in [`integration-tests.yml`](../.github/workflows/integration-tests.yml), for pull requests that change Guardrails' code, these suites or the Relay plugin.

| Suite | What it checks | Command |
| --- | --- | --- |
| [`hooks_agents/`](hooks_agents/) | Claude Code and Codex obey the hooks server's answers | `make test-hooks-agents` |
| [`relay_plugin/`](relay_plugin/) | The NeMo Relay plugin checks tool calls with a real hooks server | `make test-relay-plugin` |

## Claude Code and Codex (`hooks_agents/`)

The suite checks that Claude Code and Codex obey the hooks server through the `PreToolUse` hooks their docs configure, with the configs in `examples/configs/agent_hooks`.
A call the config allows runs, and a call it denies never runs, with the agent telling its model the server's reason.
A call also never runs when its rail fails, when the hooks server is down, or when the server doesn't answer within the hook's timeout.

Each test runs the agent headless against a fake model on loopback, which calls one scripted shell command.
The hooks server serves the test's config in the test's own process, except in the tests of a server that's down or doesn't answer: their hook points at a loopback port that refuses the connection, or at one that accepts it and never answers.
The agents are the real binaries from their published wheels.

Prerequisites:

- Linux or macOS, since the docs' hooks and the scripted command need a POSIX shell.
- `curl` on `PATH`, which Codex's hook from the docs runs to call the hooks server.
- Network access to PyPI the first time, to download the agents' wheels.
- On a managed machine, Claude Code's managed settings still apply: an `allowedHttpHookUrls` that doesn't allow `localhost` stops its hook from running.

From the repository root:

```sh
make test-hooks-agents
# One file or test, with pytest's options
make test-hooks-agents TEST=integration_tests/hooks_agents/test_hooks_agents.py::test_allowed_call_runs ARGS=-v
```

`make test-hooks-agents` installs `claude-agent-sdk`, which bundles Claude Code, and `openai-codex-cli-bin`, which bundles Codex, with their dependencies, into `.cache/hooks_agents`, checking each wheel against its hash in `requirements.txt`.
It then runs the suite in `.venv`, with live provider keys unset as for `make test`.
To change an agent's version, edit its pin in `requirements.in`, then recompile `requirements.txt` with the command that `requirements.in` gives.

## NeMo Relay Plugin (`relay_plugin/`)

The suite builds the plugin's release library with Cargo, loads it into Relay's Python plugin host, and checks tool calls against a real NeMo Guardrails hooks server: calls that Relay manages, and a LangChain agent through Relay's middleware.
A call the hooks server allows runs, and a call that the tool policy or a rail denies never runs, with the plugin rejecting it with the server's reason.
A call also never runs when the judge model gives no verdict, when the check outlasts the plugin's `timeout_millis`, or when the hooks server can't be reached.
A scripted chat model drives the LangChain agent, and a stand-in judge answers the configurations that use a judge model.

The suite runs in its own Python 3.11 environment, `.venv-relay-plugin`, without Guardrails: `nemo-relay` needs a newer `langchain` than `uv.lock` pins.
The hooks servers run from the repository's `.venv`, one for each configuration.

Prerequisites:

- Linux or macOS, since the suite loads the plugin as a `.so` or `.dylib`.
- The Rust toolchain from rustup, which reads `rust-toolchain.toml`, to build the plugin's release library.
- No machine-wide `/etc/nemo-relay/plugins.toml`: Relay would merge it into every test's plugin host, so the suite fails when it exists.
- Network access the first time, to download the packages in `requirements.txt`, the Rust toolchain that `rust-toolchain.toml` pins, the plugin's Rust crates and, when the machine lacks it, Python 3.11.

From the repository root:

```sh
make test-relay-plugin
# One file or test, with pytest's options
make test-relay-plugin TEST=integration_tests/relay_plugin/test_langchain.py ARGS=-v
```

`make test-relay-plugin` syncs `.venv`, creates `.venv-relay-plugin` with Python 3.11, and installs it from `requirements.txt`, checking every package against its hash.
To change the `nemo-relay` version, edit its pin in `requirements.in`, then recompile `requirements.txt` with the command that `requirements.in` gives.
