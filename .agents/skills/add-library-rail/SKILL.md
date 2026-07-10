---
name: "add-library-rail"
description: "Step-by-step contribution guide for adding a new rail to nemoguardrails/library/: manifest, actions returning RailOutcome, config schema, flows, HTTP rules for vendor backends, Python package requirements, unit and recorded tests, docs and examples. Use when implementing, scaffolding, or reviewing a new built-in or vendor guardrail integration. Trigger keywords - add rail, new rail, library rail, vendor integration, guardrail integration, RailManifest, RailOutcome, PythonPackage, rails validate, rail.py, guardrail catalog, third-party rail."
license: "Apache-2.0"
---

# Adding a Rail to the Library

A library rail is a manifest-declared, lazily-loaded unit under
`nemoguardrails/library/<name>/`. Every contract described here is enforced by
a test; the workflow is: copy the closest exemplar, adapt, then loop until the
enforcement tests listed at the end are green. Do not invent structure; the
exemplars are the specification.

## Step 1: Pick the archetype and exemplar

| Archetype | Exemplar to copy | Extra files beyond the core set |
| --- | --- | --- |
| Pure Python (no network, no model) | `nemoguardrails/library/regex/` | none |
| HTTP / vendor API | `nemoguardrails/library/clavata/` | `request.py` (HTTP layer), `errs.py` (vendor errors), `utils.py` |
| Model-backed (LLM judge) | `nemoguardrails/library/content_safety/` | none; model bound via `Binding.surface_param("model_name", "model")` |

Core file set required for every rail:

- `rail.py` (the manifest; discovery keys off this file)
- `actions.py` (`@action` functions returning `RailOutcome`)
- `rail_config.py` (pydantic config models + `build_config_spec()`)
- `flows.co` (Colang 2.x) and `flows.v1.co` (Colang 1.0)
- `__init__.py` (empty package marker)

Bundled data files (e.g. `injection_detection/yara_rules/*.yara`) are allowed
when the rail needs them.

## Step 2: The manifest (`rail.py`)

Define a module-level `RAIL = RailManifest(...)`. Discovery rglobs `rail.py`
under `library/` (`nemoguardrails/manifests/catalog.py`), so the file name and
the `RAIL` attribute are load-bearing.

HARD RULE: `rail.py` imports ONLY from `nemoguardrails.manifests`. No vendor
SDKs, no heavy imports, no sibling modules. Enforced by
`test_builtin_rail_modules_only_import_manifest_types`.

Copy the shape from `nemoguardrails/library/regex/rail.py`:

- `name`: unique across the catalog.
- `RailMetadata`: `display_name` and `description` must be non-empty
  (enforced), plus `categories`, `capabilities`, `tags`, and `docs_url`
  pointing at the docs catalog page (Step 7).
- `RailSpec.actions`: one `ActionRef(name=..., target="module:function")` per
  action. The ref `name` MUST equal the name the `@action` decorator
  registers; enforced by
  `test_builtin_action_refs_match_decorated_names_and_bindings`.
- `RailSpec.surfaces`: one `RailSurface` per flow entry point, with
  `direction` (input/output/retrieval), the action, and `bindings`
  (`Binding.context(...)` for `user_message`/`bot_message`/`relevant_chunks`,
  `Binding.literal(...)` for fixed params, `Binding.surface_param(...)` for
  values the user passes in the flow name). Every surface's action must be in
  `actions.refs`, and every binding's `action_param` must be a real parameter
  of the action.
- `RailSpec.config_schema`: `key` plus a `ConfigSpecRef` to
  `rail_config:build_config_spec`.
- `RailRequirements`: declare `env_vars` (e.g.
  `EnvVar(name="CLAVATA_API_KEY", required=True)`), `services`, `models`, and
  `python_packages` truthfully; this is what users, tooling, and the
  `nemoguardrails rails validate` CLI see (Step 5).
- `RailPrivacy`: declare `sends_user_text` / `sends_bot_text` /
  `remote_services` / `data_retention` honestly for any rail that ships text
  off-box; if the vendor states a retention period, record it here, not just
  in the docs page.

The catalog rejects duplicates at construction: manifest name, config key,
action name, and `(direction, surface name)` must all be unique across the
library.

## Step 3: Actions and the outcome contract

Actions live in `actions.py` and return `RailOutcome`
(`nemoguardrails/actions/rail_outcome.py`):

```python
@action(is_system_action=True)
async def my_rail_check(source: str, text: str, config: RailsConfig, **kwargs) -> RailOutcome:
    if violation_found:
        return RailOutcome.block(reason="matched policy X", rule_id=rule.id)
    return RailOutcome.allow()
```

- The three decisions are `allow`, `block`, and `transform`
  (`RailOutcome.transform(rewrites=[(TransformTarget.RELEVANT_CHUNKS, new_text)])`).
  Transforms are required iff the decision is TRANSFORM.
- Put neutral evidence in `metadata` kwargs. Do NOT put refusal text,
  exception types, or presentation decisions in the outcome; engines own
  presentation.
- Flows consume the outcome via `$response.is_blocked`,
  `$response.is_transform`, and `$response.transform_text["<context var>"]`;
  copy `regex/flows.co` and `regex/flows.v1.co` for the idiom.
- Keep `flows.co` and `flows.v1.co` semantically identical, and never
  reference a metadata key the action does not actually set. If you include
  the `enable_rails_exceptions` branch, build the exception message only
  from metadata the action provides, and cover that branch with a test; an
  untested exceptions path that reads a missing key is a latent crash.

Config models go in `rail_config.py`: pydantic models subclassing
`RailConfigBaseModel` plus `build_config_spec() -> RailConfigSpec`. The spec's
`key` must match the manifest's `config_schema.key` (enforced in
`nemoguardrails/rails/llm/rails_config_fields.py`), and exported names become
importable from `nemoguardrails.rails.llm.config`.

## Step 4: HTTP rules (vendor rails only)

All outbound HTTP goes through `nemoguardrails.http`. An AST test,
`tests/http/test_library_boundary.py`, fails the build if any file under
`nemoguardrails/library/` imports `httpx`, `requests`, `aiohttp`, or
`urllib3` directly.

- The action declares `http_client: HTTPClient | None = None`; LLMRails
  injects a managed, instrumented client as an action param at runtime. Never
  construct or manage a transport client inside the rail.
- Send requests with the canonical helpers, threading the injected client
  down (from `nemoguardrails/library/clavata/request.py`):

```python
async with resolve_http_client(self.http_client, factory=self._create_http_client) as client:
    retrying_client = RetryingHTTPClient(client, _MY_RETRY_POLICY)
    response = await http_call(retrying_client, "POST", url, json=payload, headers=headers, raise_for_status=False)
```

- Retries are rail-owned: define a module-level `RetryPolicy` constant. The
  default policy never retries POST; if your vendor call is a POST and safe
  to resend, you must opt in explicitly with
  `retryable_methods=frozenset({"POST"})` (see `_CLAVATA_RETRY_POLICY`).
- Know what `RetryingHTTPClient` already gives you before documenting or
  reimplementing anything: exponential backoff with jitter between
  `initial_delay` and `max_delay`, and the `Retry-After` response header IS
  honored (capped at `max_retry_after`, default 60s); the `x-should-retry`
  override header is opt-in via `honor_retry_override_header`. Do not
  document these semantics from guesswork; read
  `nemoguardrails/http/retry.py`.
- Telemetry must stay content-free: never log request or response bodies,
  exception messages containing payloads, URLs with query strings, or
  credentials. The instrumentation layer already sanitizes; do not undo it in
  rail-level logging.
- Wrap vendor failures in rail-specific error types (`errs.py` pattern) built
  on the `nemoguardrails.http.errors` hierarchy.

## Step 5: Vendor Python dependencies

Rail dependencies are declared in the manifest, NOT in packaging. Do NOT add
the vendor package to `pyproject.toml` or to a poetry extra; the packaging
gate `tests/test_rail_packaging.py::test_rail_dependencies_are_not_package_extras`
fails if you do. Users install the package themselves, guided by the
manifest declaration and the `nemoguardrails rails validate` CLI.

- Declare the package in `rail.py` as a module-level constant and reference
  it from the requirements (exemplar:
  `nemoguardrails/library/injection_detection/rail.py`):

```python
YARA_PACKAGE = PythonPackage(distribution="yara-python", import_name="yara", version=">=4.5.1,<5")
...
requirements=RailRequirements(python_packages=(YARA_PACKAGE,)),
```

  `PythonPackage` also supports `required=False`, an environment `marker`,
  and a `description`; `version` must be a valid PEP 440 specifier
  (validated at manifest construction).

- Load the package lazily in `actions.py` via
  `require_python_package("<rail name>", PACKAGE)` from
  `nemoguardrails.manifests`, typically behind an `lru_cache` helper
  (exemplar: `_load_yara` in
  `nemoguardrails/library/injection_detection/actions.py`). It raises
  `RailDependencyError` with an actionable install message when the package
  is missing or incompatible. Never import the vendor package at module top
  level in a way that breaks `import nemoguardrails`, and never in `rail.py`
  (Step 2 rule; `PythonPackage` comes from `nemoguardrails.manifests`, so
  declaring it there is fine).
- Static and runtime checking is handled for you:
  `nemoguardrails rails validate --config <path> [--runtime]` reports every
  configured rail's requirement status and prints the `pip install` line for
  anything missing (`nemoguardrails/cli/rails.py`); behavior is pinned by
  `tests/rails/llm/test_rail_requirements.py`.

## Step 6: Tests

Three layers, all required (per `nemoguardrails/library/README.md`):

1. **Catalog gates (free).** `tests/rails/llm/test_builtin_rail_manifests.py`
   and `tests/rails/llm/test_library_flow_files.py` pick up the new rail
   automatically: manifest, lazy refs, action names, bindings, and that both
   dialect flow files parse, define the declared flows, and invoke only
   dispatcher-resolvable actions. Run them first; they catch most wiring
   mistakes.
2. **Unit tests** in `tests/test_<rail>*.py`:
   - `TestChat` end-to-end for flow behavior, `FakeLLMModel` for
     deterministic main-model output.
   - Direct action-level tests parametrized over allow/block/transform and
     config-error paths (exemplar: `tests/test_injection_detection.py`).
   - For HTTP rails, inject the recording double instead of monkeypatching:
     `chat.app.register_action_param("http_client", RecordingHTTPClient(responses=[...]))`
     (exemplar: `tests/test_policyai_rail.py`; helper in
     `nemoguardrails/http/testing.py`). Secrets via `monkeypatch`. Unit tests
     must never reach live services.
   - Include one flow-level test of what happens when the action RAISES
     (vendor down): fail-closed is a claim about the runtime, not your code,
     so test it rather than asserting it in a summary.
   - Cover BOTH Colang dialects end to end, not just the default. `TestChat`
     runs Colang 1 (`flows.v1.co`) unless the config sets
     `colang_version: "2.x"`, which exercises `flows.co` instead (exemplar:
     `tests/test_polygraf.py`). Minimum: the block path in both dialects,
     plus the `enable_rails_exceptions` variant wherever the flow has that
     branch. The flow-files gate checks structure only; dialect behavior
     needs these tests.
3. **Recorded e2e suite** in `tests/recorded/rails/library/`: add a config
   dir under `configs/<name>/`, a constant in `configs.py`, and a
   `test_<rail>.py` covering the outcome triad (allow, block, and
   provider-error) using `check_rails`, `generate_with_fake_main`, and
   `stream_with_fake_main` from `helpers.py`, with `assert_rails_result` +
   `snapshot`. Provider-backed rails use `pytest.mark.vcr` and record
   cassettes with `make record-cassettes`; pure-Python rails use
   `@pytest.mark.pure_runtime(reason=...)` instead. Recorded tests are
   dialect-single: do not duplicate them for Colang 2 (the wire traffic is
   identical; dialect behavior is unit-test territory). For what belongs in
   the recorded suite versus `tests/`, follow the `recorded-tests` skill.
   Snapshot the NORMALIZED output and leave `snapshot()` empty for the
   record workflow to fill: `--inline-snapshot=create`/`fix` rewrites your
   test file in place (review that diff, do not revert it), and it only
   works serially, never under xdist `make test`. See the README's
   Snapshots section for the exact behavior.

## Step 7: Docs and examples

- Author a catalog page at
  `docs/configure-rails/guardrail-catalog/community/<name>.mdx` (exemplars:
  `regex.mdx`, `clavata.mdx`) and set the manifest's `docs_url` to it. The
  `docs_url` value uses the `.md` extension even though the file is authored
  as `.mdx` (see `regex/rail.py`).
- Add an example config under `examples/configs/<rail>/`.
- Document the required Python packages (the `pip install` line), required
  env vars / API keys, the remote service and what text is sent to it, and
  known limitations, per the integration rules in `nemoguardrails/AGENTS.md`
  and `docs/AGENTS.md`.

## Final verification loop

Run until green, in this order (cheapest first):

```bash
make test TEST=tests/rails/llm/test_builtin_rail_manifests.py
make test TEST="tests/rails/llm/test_rail_requirements.py tests/test_rail_packaging.py"
make test TEST=tests/http/test_library_boundary.py
make test TEST=tests/test_<rail>.py
poetry run pytest tests/recorded/rails/library --block-network -q
poetry run pre-commit run --files <changed files>
make docs-fern
```

A rail is contribution-ready only when all of these pass and every
declaration in the manifest (python_packages, env vars, privacy, docs_url)
matches what the code actually does. Reviewers of rail PRs apply the
`review-library-rail` skill; running its judgment dimensions on your own
diff before handoff is the cheapest review you will get.
