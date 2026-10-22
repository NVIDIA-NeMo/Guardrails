# NeMo Guardrails checks plugin for NeMo Relay

A native [NeMo Relay](https://github.com/NVIDIA/NeMo-Relay) plugin, `nvidia.nemo_guardrails_checks`, that will check tool calls and tool results with the checks API of a NeMo Guardrails server: `/v1/checks/output` for tool calls and `/v1/checks/input` for tool results.

> [!NOTE]
> This crate is a scaffold.
> Relay loads the plugin and validates its configuration, but the plugin doesn't register any checks yet.

## Compatibility

- NeMo Relay 0.10 and later 0.x releases, through native ABI v7 (`compat.relay = ">=0.10,<1.0"`).
- Each Relay minor release changes the native ABI, so the plugin is rebuilt against each one.

## Configuration

Relay passes the plugin's entry in `plugins.toml` as the component config.
`config.schema.json` describes it, and Relay checks each config against that schema before the plugin loads.

| Field | Default | Description |
|---|---|---|
| `endpoint` | — | Required. Base URL of the NeMo Guardrails server, using http or https. The plugin calls its `/v1/checks/input` and `/v1/checks/output` endpoints. |
| `config_id` | — | Required. Guardrails configuration to check against. It must name a configuration loaded on the server. |
| `headers` | `{}` | Headers sent with every check. The Guardrails server has no authentication of its own, so these serve a proxy or gateway in front of it, such as an `authorization` header. Values are treated as sensitive. |
| `timeout_millis` | `1500` | Time limit for one check, in milliseconds. |
| `priority` | `0` | Relay execution-intercept priority. |
| `executor` | — | Reserved by the Relay SDK for its executor settings. |

An invalid config is reported with the diagnostic code `nemo_guardrails_checks.invalid_config`.

## Building and testing

From the repository root:

```sh
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test -p nemo-guardrails-relay-plugin
cargo build --release -p nemo-guardrails-relay-plugin
```

`tests/lifecycle.rs` builds the plugin library, then loads it into a real Relay host through the `nemo-relay` crate.
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
[plugins.policy.overrides."nvidia.nemo_guardrails_checks"]
attestation = "integrity_only"
```
