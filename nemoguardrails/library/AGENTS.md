# AGENTS.md

Subtree guidance for `nemoguardrails/library/` (built-in and vendor rails).
This supplements `nemoguardrails/AGENTS.md` and the repository-root guides; all
of those rules still apply.

A library rail is described by a neutral `rail.py` manifest. Generic gates
discover its structural contract automatically, while pinned engine,
telemetry, and pooling expectations require deliberate updates for applicable
surfaces. The rules below keep those checks meaningful. If your client supports agent
skills, the `add-library-rail` and `review-library-rail` skills under
`.agents/skills/` give the full step-by-step contribution and review guides;
this file is the always-loaded summary of the rules the gates enforce.

## The core rule: fix the rail, not the gate

The conformance suite reads every manifest and checks Python signatures,
Colang declarations, config schema, unique requirement identifiers, and that
privacy services are declared requirements. It does not verify implementation
or docs against env vars, optional imports, outbound calls, or retention
claims. When a generic check fails, the fix is almost always in your rail, not
the test:

- Do NOT edit the generic gates to make your rail pass:
  `tests/rails/llm/test_builtin_rail_manifests.py`,
  `tests/rails/llm/test_builtin_rail_conformance.py`,
  `tests/rails/llm/test_library_flow_files.py`.
- Do NOT add your rail to a hardcoded exception list
  (`LEGACY_UNMANIFESTED_PACKAGES`, `NON_PORTABLE_DECLARED_FLOWS`, or similar).
  Those name pre-existing exceptions; a new rail should satisfy the current
  contract by default. An addition requires an explicit, reviewed design
  decision, not a way to get a test green.
- Regenerate `schemas/rails_config.snapshot.json` only when your config schema
  legitimately changes, and review the diff:
  `uv run --locked python scripts/generate_rails_config_schema_snapshot.py`.
  The conformance gate compares the projected schema against this snapshot, so
  an unreviewed regeneration hides real drift.

The gates prove structural consistency; they do not prove the rail behaves
correctly. Every rail still needs focused behavioral tests and recorded e2e
coverage (see `README.md` and `tests/recorded/rails/library/README.md`).

## Adding a new rail

Each new rail under `nemoguardrails/library/<name>/` provides:

1. `rail.py`: a `RAIL = RailManifest(...)` importing only from
   `nemoguardrails.manifests`, never the implementation or optional
   dependencies. Declare metadata with a valid `docs_url`.
2. Every executable action via `ActionRef`; typed config via `ConfigSpecRef`
   when configurable. Config-only manifests may omit actions and flows.
3. For flow-bearing rails, public flows and both Colang dialect files
   (`flows.co`, `flows.v1.co`).
4. Portable surfaces with direction, bindings, and transform target for each
   exposed flow.
5. Requirements: env vars, services, models, and truthful `RailPrivacy`
   (including `data_retention` when the vendor states one).
6. Optional third-party imports kept inside execution-time code, loaded lazily.
7. Focused tests for applicable behavior and failure modes, plus a recorded
   e2e entry. Provider-backed rails cover allow, block, and a recordable
   provider error; pure-Python rails have no provider-error requirement.

Library actions must stay importable without Colang. Import model-call helpers
from `nemoguardrails.llm.call`, not the compatibility module under
`nemoguardrails.actions.llm.utils`. IORails supplies only its fixed request
dependencies; arbitrary `register_action_param()` injection is LLMRails-only.

HTTP rails accept an injected `HTTPClient`, preserve its ownership, and apply
rail retry policy with `ensure_retrying_http_client` from
`nemoguardrails.http.composition`. Do not directly wrap an injected client in
`RetryingHTTPClient`: retry belongs inside instrumentation and an existing
retry policy takes precedence. Error and telemetry text must not contain
checked content, request or response bodies, query strings, or credentials.

## Changing an existing rail: know the blast radius

Manifests deliberately mirror facts that also live in code, Colang, docs, and
config. When you change one side, update the other or a gate will catch the
drift:

| Change | Required updates |
| --- | --- |
| Implementation only | Focused action tests; the manifest usually stays. |
| Python module or symbol moves | Update the manifest's `ActionRef.target`. |
| Decorated action name changes | Update `ActionRef.name`, Colang 1 `execute` calls, Colang 2 `CamelCaseAction` calls, and surface references. Treat as a public compatibility change. |
| Parameters change | Update surface bindings and flow arguments; every `Binding.action_param` must still exist in the signature. |
| Return contract | Surface actions must still annotate and return `RailOutcome`. |
| Configuration changes | Update the typed config model, regenerate `schemas/rails_config.snapshot.json`, and review the schema diff. |
| Service or model changes | Update manifest services, models, env vars, and privacy declarations. |

## Validate

```bash
make test TEST="tests/rails/llm/test_builtin_rail_manifests.py tests/rails/llm/test_builtin_rail_conformance.py tests/rails/llm/test_library_flow_files.py"
make test TEST="tests/rails/llm/test_config.py tests/test_runtime_flow_gate_equivalence.py"
make test TEST="tests/guardrails/test_compiled_rail.py tests/guardrails/test_guardrails.py tests/guardrails/test_rails_manager.py"
make test TEST="tests/llm/test_call_import_graph.py tests/telemetry/test_usage_reporting.py tests/http/test_library_boundary.py"
```

Regenerate the schema snapshot with
`scripts/generate_rails_config_schema_snapshot.py` whenever the projected
config changes, and run the rail's recorded suite with `--block-network`.
