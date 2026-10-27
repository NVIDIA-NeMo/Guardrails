# NVIDIA NeMo Guardrails Plugin for NeMo Relay

A native [NeMo Relay](https://github.com/NVIDIA/NeMo-Relay) plugin, `nvidia.nemo_guardrails`, that checks the tool calls Relay manages against the NeMo Guardrails service.

The plugin is a client and holds no rails of its own.
The NeMo Guardrails service is a hooks server, started with `nemoguardrails hooks-server --config <path>`; its one configuration holds the rails and decides every call.
Before each tool call that Relay manages, the plugin posts the tool's name and arguments to the service's `/v1/hooks/nemo-relay` endpoint.
A deny from the service becomes a guardrail rejection, and a check that fails (an error status, a timeout, a service that can't be reached) becomes an error.
Relay blocks the call on either, from 0.10.1 on, with one exception: on an error, pi's extension lets the call run unless `NEMO_RELAY_PI_FAIL=closed` is set.

## Checks

The plugin registers one tool conditional-execution guardrail, `nemo_guardrails_tool_call_check`, at the configured `priority`.
It runs on the tool calls Relay manages: calls through the Relay SDK and its LangChain and LangGraph integrations, and the PreToolUse hooks of the agents the Relay CLI runs.

- An allow lets the call run.
- A deny rejects the call with `NeMo Guardrails blocked tool call '<tool>': <reason>`, or with the text ending at the tool name when the service gives no reason.
- A failed check returns the error `NeMo Guardrails check failed for tool call '<tool>': <detail>`.
  The detail gives the HTTP status, the service's `error.code` and message, and a hint for the service's known errors, or says that the request timed out or couldn't be sent.
- The plugin doesn't retry a failed check, not even when the service answers 503 with `Retry-After`.

A managed call gets the rejection or the error from Relay as is.
In the CLI's hook modes, an agent can see Relay's own text for a failed check instead, such as `daemon hook forward failed with HTTP 500`.
So the plugin also logs each failed check as a warning, under the log target `nemo_relay.plugin.native.nvidia.nemo_guardrails`.
Its message is the error text, and its fields are `event` (`nemo_guardrails_check_failed`), `tool_name`, `status` and `error_code`.
The plugin puts no header value and none of the tool's arguments into the error text or the warning.

The plugin checks every call; it keeps no earlier decisions.
Through Relay 0.10.4, a call that reaches a permission prompt runs the guardrail twice, on PreToolUse and right after on PermissionRequest, so it costs two requests to the service, and two runs of any LLM judge.
Relay 0.11 runs the guardrail once.

## Compatibility

- NeMo Relay 0.10.1 and later 0.x releases, through native ABI v7 (`compat.relay = ">=0.10.1,<1.0"`).
  Relay 0.10.0 is excluded: the plugin reports a failed check as an error, and at 0.10.0 an error from a guardrail lets a Claude Code tool call run.
- Each Relay minor release changes the native ABI, so the plugin is rebuilt against each one.

## Configuration

Relay passes the plugin's entry in `plugins.toml` as the component config.
`config.schema.json` describes it, and Relay checks each config against that schema before the plugin loads.
Most of its fields say how to reach the NeMo Guardrails service; the rails are configured in the service.

| Field | Default | Description |
|---|---|---|
| `endpoint` | — | Required. Base URL of the NeMo Guardrails service (its hooks server), using http or https, such as `http://127.0.0.1:8100`. The plugin calls the service's `/v1/hooks/nemo-relay` endpoint, keeping any path prefix. It must not contain a username or password; send credentials in `headers`. |
| `headers` | `{}` | Headers sent with every call to the service, as header names and string values. The service has no authentication of its own, so these serve a proxy or gateway in front of it, such as an `authorization` header. Values are treated as sensitive, and no error quotes one. Headers the plugin's HTTP client controls, such as `content-type`, `content-length` and `host`, are rejected. |
| `timeout_millis` | `25000` | Time limit for one call to the service, in milliseconds. Keep it under the host's hook limit: 30 seconds for Claude Code and Codex hooks, 5 seconds by default for pi. |
| `priority` | `0` | Priority of the plugin's tool conditional-execution guardrail among Relay's guardrails. Lower values run earlier. |
| `executor` | — | Reserved by the Relay SDK for its executor settings. |

An invalid config is reported with the diagnostic code `nemo_guardrails.invalid_config`.

## Building and testing

From the repository root:

```sh
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test -p nemo-guardrails-relay-plugin
# As the NeMo-Relay-Plugins release pipeline runs it, loading the release-profile library
cargo test --locked --release -p nemo-guardrails-relay-plugin
cargo build --release -p nemo-guardrails-relay-plugin
# The packaging script's tests
make test TEST=crates/nemo-guardrails-relay-plugin/scripts/test_package_bundle.py
```

`tests/client.rs` runs the hooks client against a stand-in NeMo Guardrails service, from `wiremock`.
`tests/lifecycle.rs` and `tests/tool_call_check.rs` build the plugin library in the test's profile, then load it into a real Relay host through the `nemo-relay` crate; `tests/tool_call_check.rs` points it at a stand-in service.
None of these tests needs a running NeMo Guardrails service.
`.cargo/config.toml` sets `NEMO_RELAY_TEST_SKIP_IMPLICIT_CONFIG=1`, so that the host tests ignore a machine-wide `/etc/nemo-relay/plugins.toml`.

## Packaging

`scripts/package_bundle.py` writes a bundle that Relay can load into an empty directory.
The bundle holds the library, the manifest with its digest filled in, the config schema, and the license files:

```sh
python crates/nemo-guardrails-relay-plugin/scripts/package_bundle.py \
  --library target/release/libnemo_guardrails_relay_plugin.so \
  --output /tmp/nemo-guardrails-relay-plugin
```

The library is `libnemo_guardrails_relay_plugin.dylib` on macOS and `nemo_guardrails_relay_plugin.dll` on Windows.
Pass `--archive <path>.tar.gz` or `--archive <path>.zip` to also archive the bundle.

Bundles aren't signed.
To load one, allow integrity-only attestation for the plugin in `plugins.toml`:

```toml
[plugins.policy.overrides."nvidia.nemo_guardrails"]
attestation = "integrity_only"
```
