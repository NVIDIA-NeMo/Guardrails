# Telemetry smoke test

Operator-driven local script that fires every realistic deployment scenario against the staging telemetry endpoint, validates each emitted event against the vendored upstream schema, and writes a manifest the operator pastes into Kibana to verify events landed.

The driver lives at [`scripts/telemetry_smoke.py`](../scripts/telemetry_smoke.py).

## When to run

- After any change to `nemoguardrails/telemetry.py`, `server/api.py`, `rails/llm/llmrails.py`, or `guardrails/guardrails.py` that touches the wire format.
- Before merging the upstream `nemo-telemetry` MR or cutting a release that ships telemetry.
- After SMS confirms our schema is registered in staging (run the pre-flight below first to verify).

## Prerequisites

- `poetry install --with dev` so `jsonschema` is available for local schema validation.
- `tests/telemetry/smoke_fixtures/{cfg1,cfg2,cfg3,rich}/` and `examples/configs/nemoguards/` are present in the working tree.
- `CI`, `GITHUB_ACTIONS`, and `PYTEST_CURRENT_TEST` must NOT be exported in the calling shell. The driver refuses to start otherwise. `unset` them first.

## Pre-flight: one event by hand

Before driving the full script, send a single event and verify it lands in Kibana. This proves staging is accepting our event_name.

```bash
unset CI GITHUB_ACTIONS PYTEST_CURRENT_TEST
export NEMO_GUARDRAILS_SMOKE_STAGING_URL=<internal-staging-events-url>/v1.1/events/json
NEMO_GUARDRAILS_USAGE_STATS_SERVER="$NEMO_GUARDRAILS_SMOKE_STAGING_URL" \
  poetry run python - <<'PY'
import json
import time
from nemoguardrails import LLMRails, RailsConfig, telemetry

LLMRails(RailsConfig.from_path("examples/bots/abc"))
audit_file = telemetry._get_audit_file()
deadline = time.monotonic() + 10
while time.monotonic() < deadline:
    if audit_file.exists() and audit_file.read_text().strip():
        break
    time.sleep(0.1)
else:
    raise SystemExit(f"pre-flight audit event not observed at {audit_file}")

time.sleep(20)
lines = [json.loads(line) for line in audit_file.read_text().splitlines() if line.strip()]
startup_ids = sorted({line["sessionId"] for line in lines if line.get("event") == "startup"})
print(f"pre-flight audit event observed at {audit_file}")
print("kibana filter: client.sessionId : (" + " or ".join(f'"{value}"' for value in startup_ids) + ")")
PY
```

The local audit-file poll proves the event was emitted before the process exits; the final sleep gives the daemon network send a chance to complete before the process is terminated. Then wait briefly for indexing and search Kibana with the exact `client.sessionId` filter printed by the command. If nothing lands, defer the full smoke run until SMS has confirmed the schema is registered in staging.
Do not commit internal staging endpoint URLs to this repo; keep them in your shell, password manager, or internal runbook.

## Full run

```bash
unset CI GITHUB_ACTIONS PYTEST_CURRENT_TEST
export NEMO_GUARDRAILS_SMOKE_STAGING_URL=<internal-staging-events-url>/v1.1/events/json
poetry run python scripts/telemetry_smoke.py
```

To target a different endpoint or override config paths:

```bash
poetry run python scripts/telemetry_smoke.py \
  --staging-url "$NEMO_GUARDRAILS_SMOKE_STAGING_URL" \
  --library-config tests/telemetry/smoke_fixtures/cfg1 \
  --rich-config tests/telemetry/smoke_fixtures/rich \
  --feature-alias-config tests/telemetry/smoke_fixtures/feature_aliases \
  --v2-config tests/telemetry/smoke_fixtures/v2_custom_flow \
  --iorails-config examples/configs/nemoguards \
  --server-config-root tests/telemetry/smoke_fixtures
```

To run a single scenario (useful while iterating):

```bash
poetry run python scripts/telemetry_smoke.py --scenario library_llmrails
```

For a quick local regression check while editing the smoke driver, use an intentionally unreachable telemetry URL and run the scenarios most likely to catch local regressions:

```bash
poetry run python scripts/telemetry_smoke.py \
  --staging-url http://127.0.0.1:9 \
  --run-dir /tmp/smoke-rewrite-check \
  --scenario library_feature_aliases \
  --scenario library_v2_custom_flows \
  --scenario server_multi_config \
  --scenario server_multi_worker \
  --scenario heartbeat
```

The unreachable URL keeps receiver delivery out of scope; telemetry send failures are non-fatal, while the local audit-file schema and field assertions still run. `library_feature_aliases` covers documented built-in feature IDs for config-only rails, `library_v2_custom_flows` covers bundled Colang 2.x library-flow exclusion from custom-flow counts with a minimal fixture, `server_multi_config` covers paced server emissions across configs, `server_multi_worker` covers three Uvicorn workers with distinct API sessions, and `heartbeat` covers startup plus heartbeat delivery without a heartbeat burst during network settle.

The driver prints per-scenario PASS/FAIL, the exact Kibana filter, and the offline verification command. It writes `<run_dir>/manifest.json` containing the run id, collected startup session IDs, per-scenario verdicts, audit-file paths, event counts, subprocess return codes, stderr tails, server POST results, expected assertion summaries, and the configured staging URL. Treat the manifest as local verification output and do not commit it.

## Scenarios covered

| Name | Expected wire shape |
|---|---|
| `library_llmrails` | 1 event, `deploymentType=library`, `railsEngine=LLMRails` |
| `library_rich_config` | 1 event with tracing, streaming, knowledge base, and custom-flow fields enabled |
| `library_feature_aliases` | 1 event with config-only `builtinFeatures=["factchecking","patronusai","regex"]` |
| `library_v2_custom_flows` | 1 event with `colangVersion=2.x` and `numCustomFlows=1` despite bundled `core` flows |
| `library_iorails` | 1 event, `deploymentType=library`, `railsEngine=IORails` |
| `server_single_config` | 1 event, `deploymentType=api`, `railsEngine=LLMRails` |
| `server_multi_config` | 3 events, same sessionId, distinct `railTypesInUse` / `builtinFeatures` |
| `server_multi_worker` | 3 events from 3 Uvicorn workers, distinct sessionIds, `deploymentType=api` |
| `cli_chat` | 1 event, `deploymentType=cli` |
| `heartbeat` | 1 startup + at least 1 event with `event=heartbeat` |
| `opt_out_explicit` | audit file empty (NEMO_GUARDRAILS_NO_USAGE_STATS=1) |
| `opt_out_ci` | audit file empty (CI=true) |
| `opt_out_pytest` | audit file empty (PYTEST_CURRENT_TEST=...) |

## Local invariants the driver checks automatically

For every scenario:

- Subprocess exit code is zero.
- Audit-file event count matches expected.
- Each emitted line validates against `schemas/anonymous_events.snapshot.json` using `jsonschema` Draft-07.
- Common payload fields are structurally valid: source, event type, non-empty session ID, version, Python version, platform, OS name, and timestamp.
- Startup payload fields are asserted per fixture: deployment type, rails engine, providers, Colang version, rail counts/types, built-in features, tracing, knowledge base, streaming, and custom-flow count.
- Distinct sessionId count for `server_multi_worker`.
- Presence of `event=heartbeat` for `heartbeat`, with heartbeat and startup sharing the same sessionId.

## Kibana verification

### Offline: copy documents as JSON, verify locally

For accounts that have Kibana Discover access but cannot issue API keys (no Security → API Keys in Stack Management), use [`scripts/kibana_verify_export.py`](./kibana_verify_export.py):

1. Run the smoke driver; note the manifest path.
2. Open Kibana Discover at the `<staging-telemetry-data-view>` data view.
3. Wait roughly a minute for indexing lag to settle.
4. Paste the exact filter printed by the driver and saved as `kibana_filter` in the manifest:

   ```
   client.sessionId : ("id1" or "id2" or "id3")
   ```

5. Click the checkbox in the document-table header to **select all visible rows**.
6. In the table's toolbar a "**N Selected**" dropdown appears. Open it and choose **Copy documents as JSON**.
7. Paste the clipboard into a local file (e.g. `kibana.json` in the repo root). The format is a JSON array where each element has `_index`, `_id`, and a `fields` dict. Kibana exports may contain internal staging telemetry documents; do not commit them.
8. Run:

   ```bash
   poetry run python scripts/kibana_verify_export.py \
     --manifest <run_dir>/manifest.json \
     --export kibana.json
   ```

The script reads the manifest, ignores any non-`guardrails_usage_event` documents in the export (Discover filters apply to *visible rows*; the Selected list may include sibling event types if the time range covers other products), matches each remaining doc by exact `client.sessionId`, and asserts hit counts. Per-scenario PASS / FAIL summary. Exit code 0 on success, 1 on assertion failure, 2 on missing or malformed input files.

### Last-resort: visual inspection

If you cannot download an export at all, paste the driver's exact filter into the Kibana search bar:

```
client.sessionId : ("id1" or "id2" or "id3")
```

Use each result's `startup_session_ids` in `manifest.json` for per-scenario inspection.

Negative scenarios (`opt_out_*`) should match nothing.

## Troubleshooting

**"refusing to run: the driver inherited CI ..."**
`unset CI GITHUB_ACTIONS PYTEST_CURRENT_TEST` in your shell and retry.

**Driver passes locally but Kibana shows nothing**
Staging may be rejecting unregistered events or indexing may be delayed. Confirm with SMS that the schema is registered in staging, then re-run the pre-flight and smoke driver.

**Single scenario fails with "expected N events, got 0"**
Usually means `LLMRails` failed at import or construction time. Check the manifest's `audit_file` for that scenario; if empty, look at the subprocess's `stderr_tail` in the result. Most common cause: a missing dependency or a config-validation error.

**A reused `--run-dir` contains old events**
The driver removes each known per-scenario directory before running that scenario, so rerunning with the same `--run-dir` is supported. If a run is interrupted before the script starts a scenario, only that unstarted scenario's old audit file can remain.

## Out of scope

- Cron-scheduled smoke runs (CI auto-disables telemetry by design).
- Live Kibana queries from the driver. Verification of events landing remains a manual paste of the exact session-ID filter into Kibana.
- Mock LLM. Library, CLI, and heartbeat scenarios construct `LLMRails` only. Server scenarios POST a request that triggers `_get_rails` and then intentionally fails before generation, so no API key is required. Configs that fail to *construct* (bad YAML, missing models field, etc.) are out of scope of the smoke test.
