---
name: "add-library-rail"
description: "Step-by-step contribution guide for adding a new rail to nemoguardrails/library/: manifest, actions returning RailOutcome, config schema, flows, HTTP rules for vendor backends, Python package requirements, unit and recorded tests, docs and examples. Use when implementing, scaffolding, or reviewing a new built-in or vendor guardrail integration. Trigger keywords - add rail, new rail, library rail, vendor integration, guardrail integration, RailManifest, RailOutcome, RailRequirements, optional_dependencies, rail.py, guardrail catalog, third-party rail."
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

For an HTTP rail, pick the exemplar by complexity: `clavata/` factors its
vendor client into `request.py` for a multi-endpoint API, while
`nemoguardrails/library/f5/` inlines a single `http_call` in `actions.py` and
returns `RailOutcome` directly. F5 is the smallest complete vendor rail on the
managed HTTP client, so copy it when the vendor is one endpoint.

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
  `EnvVar(name="CLAVATA_API_KEY", required=True)`), `services`, `models`,
  `extras`, and `optional_dependencies` truthfully; this is what users and
  tooling see (Step 5).
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
        return RailOutcome.block(reason="matched policy X", metadata={"rule_id": rule.id})
    return RailOutcome.allow()
```

The seam here is **actions decide, flows present**: the action returns a
neutral verdict (allow/block/transform plus evidence in `metadata`), and the
flow owns everything a user sees -- refusal wording, exception-vs-bot-message,
localization. When you are unsure where a new piece of logic goes, ask which
side of that line it is on. Anything a human reads belongs in the flow, never
in the outcome.

- The three decisions are `allow`, `block`, and `transform`
  (`RailOutcome.transform(rewrites=[(TransformTarget.RELEVANT_CHUNKS, new_text)])`).
  Transforms are required iff the decision is TRANSFORM.
- Put neutral evidence in the single `metadata` mapping argument, meaning
  machine-shaped values only: identifiers, category names, scores, booleans,
  and the vendor's own parsed response. `allow`, `block`, and `transform` are
  keyword-only and take `reason` and `metadata` and nothing else, so evidence
  goes inside the mapping, not as loose keyword arguments. Do NOT put refusal
  text, exception types, or presentation decisions in the outcome; engines own
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

The seam here is **the helper owns provider semantics; resilience is
composed, not hand-rolled**. Your code owns the endpoint, body, auth, and how
to read the response; retries, backoff, timeout, TLS, and telemetry belong to
the injected or composed client. The rail's only resilience input is a
declared `RetryPolicy` value -- the client applies it.

- Declare `http_client: HTTPClient | None = None` on the action. When it is
  `None`, `http_call` creates and closes a call-scoped client. Applications
  can register a caller-owned shared or instrumented client as an action
  parameter when their workload warrants it; never close an injected client
  inside the rail.
- Send requests with the canonical helpers, threading the injected client
  down (from `nemoguardrails/library/clavata/request.py`):

```python
client = self._retrying_http_client(self.http_client) if self.http_client is not None else None
response = await http_call(
    client,
    "POST",
    url,
    json=payload,
    headers=headers,
    raise_for_status=False,
    factory=self._create_http_client,
)
```

- `http_call` manages the fallback client's lifecycle. Wrapping an injected
  client must not transfer ownership; the factory must return the fully
  composed closable client for the fallback path.
- Retries are rail-owned: define a module-level `RetryPolicy` constant. The
  default policy never retries POST; if your vendor call is a POST and safe
  to resend, you must opt in explicitly with
  `retryable_methods=frozenset({"POST"})` (see `_CLAVATA_RETRY_POLICY`). If
  you find yourself writing `for attempt in range(...)`, an `asyncio.sleep`
  backoff, or a manual timeout race in the action or request helper, stop:
  that is resilience, so declare it in the `RetryPolicy` and wrap the client
  instead of implementing it by hand (this is the F5-v1 mistake the managed
  client migration removed).
- Know what `RetryingHTTPClient` already gives you before documenting or
  reimplementing anything: exponential backoff with jitter between
  `initial_delay` and `max_delay`, and the `Retry-After` response header IS
  honored (capped at `max_retry_after`, default 60s); the `x-should-retry`
  override header is opt-in via `honor_retry_override_header`. Do not
  document these semantics from guesswork; read
  `nemoguardrails/http/retry.py`.
- Telemetry must stay content-free: never log request or response bodies,
  exception messages containing payloads, URLs with query strings, or
  credentials. Explicitly composed instrumentation sanitizes URLs; do not
  undo that protection in rail-level logging.
- Wrap vendor failures in rail-specific error types (`errs.py` pattern) built
  on the `nemoguardrails.http.errors` hierarchy.

## Step 5: Vendor Python dependencies

Rail dependencies are declared in the manifest, NOT in packaging. Do NOT add
the vendor package to `pyproject.toml`; users install it themselves, guided by
the manifest declaration and the docs page install line.

- Declare the distribution name in `rail.py` through
  `RailRequirements.optional_dependencies` (exemplar:
  `nemoguardrails/library/injection_detection/rail.py`):

```python
requirements=RailRequirements(optional_dependencies=("yara-python",)),
```

  `optional_dependencies` is a tuple of PyPI distribution names
  (`Tuple[str, ...]`, `nemoguardrails/manifests/manifest.py`).
  `RailRequirements` accepts only `extras`, `env_vars`, `services`, `models`,
  and `optional_dependencies`, and it is declared `extra="forbid"`, so any
  other key fails at manifest construction. Version bounds are not
  expressible here; state them in the docs page `pip install` line. Use
  `extras=("<extra>",)` only when the package already ships in a
  nemoguardrails extra (exemplar:
  `nemoguardrails/library/sensitive_data_detection/rail.py`).

- Import the vendor package lazily in `actions.py` behind a module-level
  guard that leaves the name bound to `None` on ImportError, then check
  availability inside the action and raise with the exact `pip install` line
  (exemplar: the `yara = None` / `try: import yara` guard and
  `_check_yara_available` in
  `nemoguardrails/library/injection_detection/actions.py`). Never import the
  vendor package at module top level in a way that breaks
  `import nemoguardrails`, and never in `rail.py` (Step 2 rule).

## Step 6: Tests

Three layers, all required (per `nemoguardrails/library/README.md`):

1. **Catalog gates (free).** `tests/rails/llm/test_builtin_rail_manifests.py`,
   `tests/rails/llm/test_library_flow_files.py`, and
   `tests/rails/llm/test_builtin_rail_conformance.py` pick up the new rail
   automatically: manifest, lazy refs, action names, bindings, both dialect
   flow files parsing and invoking only dispatcher-resolvable actions, and
   cross-artifact conformance (surfaces declare `RailOutcome`, requirements
   and privacy are consistent, and the projected config schema matches
   `schemas/rails_config.snapshot.json`). Run them first; they catch most
   wiring mistakes. If a config change moves the schema, regenerate the
   snapshot with `scripts/generate_rails_config_schema_snapshot.py` and
   review the diff. Never edit a generic gate or add your rail to an
   exception list (`LEGACY_UNMANIFESTED_PACKAGES`,
   `NON_PORTABLE_DECLARED_FLOWS`) to get green; fix the manifest.
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
   - Cover EVERY error branch the action can take, not just one. With a
     mocked transport (`RecordingHTTPClient` or `aioresponses`) synthetic
     errors are cheap and deterministic, so this is where exhaustive error
     coverage belongs. For an HTTP/vendor rail, enumerate the action's
     failure branches and give each a test: timeout, connection error, each
     handled status class (4xx, 5xx), rate limiting (429 retried-then-success
     and 429 retry-exhausted), malformed or unexpected payload, missing
     credential, and the missing optional-dependency path -- each crossed
     with fail-open and fail-closed where the rail supports both. A silently
     swallowed vendor error is the worst failure mode for a guardrail, so
     "one raise test" is not enough.
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
   `snapshot`. The recorded provider-error is only the RECORDABLE real error
   (an invalid key yielding a real 401 is the canonical one; a real 429 with
   a `Retry-After` header is a common second). Do NOT try to mirror every
   error branch here: a synthetic timeout or 5xx injected by test code
   belongs in the unit layer by the placement rule, and most error paths
   cannot be recorded against a live vendor anyway. Provider-backed rails use `pytest.mark.vcr` and record
   cassettes with `make record-cassettes`; pure-Python rails carry only the
   module-level `pytestmark = [pytest.mark.recorded, pytest.mark.asyncio]`
   and no `vcr` mark (exemplar:
   `tests/recorded/rails/library/test_regex.py`). Recorded tests are
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
  `regex.mdx`, `clavata.mdx`) and set the manifest's `docs_url` to that same
  repo-relative path, extension included, so it ends in `.mdx` (see
  `regex/rail.py`). `test_builtin_manifest_docs_urls_resolve` asserts
  `Path(docs_url).is_file()`, so an `.md` value fails a gate this skill
  tells you to run.
- Add an example config under `examples/configs/<rail>/`.
- Document the required Python packages (the `pip install` line), required
  env vars / API keys, the remote service and what text is sent to it, and
  known limitations, per the integration rules in `nemoguardrails/AGENTS.md`
  and `docs/AGENTS.md`.

## Modifying an existing rail

Steps 1-7 assume a new rail. When you change one that already exists, the
manifest deliberately mirrors facts that also live in the code, Colang files,
config schema, and docs, so a change on one side needs the other or a
conformance gate flags the drift. Update the mirrored artifacts together:

| Change | Required updates |
| --- | --- |
| Implementation only | Focused action tests; the manifest usually stays. |
| Python module or symbol moves | Update the manifest's `ActionRef.target`. |
| Decorated action name changes | Update `ActionRef.name`, Colang 1 `execute` calls, Colang 2 `CamelCaseAction` calls, and surface references. Treat as a public compatibility change. |
| Parameters change | Update surface bindings and flow arguments; every `Binding.action_param` must still exist in the signature. |
| Return contract | Surface actions must still annotate and return `RailOutcome`. |
| Configuration changes | Update the typed config model, regenerate `schemas/rails_config.snapshot.json`, and review the schema diff. |
| Dependency or service changes | Update manifest `optional_dependencies`, `extras`, env vars, services, models, and privacy declarations. |

Then re-run the catalog gates from Step 6; they are the mechanical check that
the mirrored copies still agree. This same matrix is the always-loaded
tripwire in `nemoguardrails/library/AGENTS.md`.

## Final verification loop

Run until green, in this order (cheapest first):

```bash
make test TEST="tests/rails/llm/test_builtin_rail_manifests.py tests/rails/llm/test_builtin_rail_conformance.py tests/rails/llm/test_library_flow_files.py"
make test TEST=tests/http/test_library_boundary.py
make test TEST=tests/test_<rail>.py
make test TEST=tests/recorded/rails/library ARGS="--block-network -q"
uv run --locked pre-commit run --files <changed files>
make docs-fern
```

A rail is contribution-ready only when all of these pass and every
declaration in the manifest (optional_dependencies, env vars, privacy, docs_url)
matches what the code actually does. Reviewers of rail PRs apply the
`review-library-rail` skill; running its judgment dimensions on your own
diff before handoff is the cheapest review you will get.
