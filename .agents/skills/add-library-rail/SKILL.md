---
name: "add-library-rail"
description: "Step-by-step contribution guide for adding a new rail to nemoguardrails/library/: manifest, actions returning RailOutcome, config schema, flows, HTTP rules for vendor backends, Python package requirements, unit and recorded tests, docs and examples. Use when implementing, scaffolding, or reviewing a new built-in or vendor guardrail integration. Trigger keywords - add rail, new rail, library rail, vendor integration, guardrail integration, RailManifest, RailOutcome, RailRequirements, optional_dependencies, rail.py, guardrail catalog, third-party rail."
license: "Apache-2.0"
---

# Adding a Rail to the Library

A library rail is a manifest-declared, lazily-loaded unit under
`nemoguardrails/library/<name>/`.

The STRUCTURAL contracts here are checked by the conformance gates in Step 6:
manifest shape, action refs and bindings, flow-file validity, and the
projected config schema. The BEHAVIORAL ones are not, and no gate will tell
you when they are wrong: privacy honesty, retry semantics, fail-closed
behavior, telemetry content-freeness, docs accuracy, and cassette provenance
need the rail-specific tests you write plus a reviewer's judgment.

The workflow is: start from the closest maintained pattern, adapt it, run the
gates until green, then write the behavioral and recorded end-to-end coverage
the gates cannot stand in for. Shipped rails span several generations of the
architecture, so an exemplar is authoritative only for the specific pattern
named below; apply the current contracts in this guide to everything else.

## Step 1: Pick the archetype and exemplar

| Archetype | Exemplar to copy | Extra files beyond the core set |
| --- | --- | --- |
| Pure Python (no network, no model) | `nemoguardrails/library/regex/` | none |
| HTTP / vendor API | `nemoguardrails/library/hf_classifier/backends.py` for client composition; `nemoguardrails/library/clavata/` for request/error layering | optional `request.py` and `errs.py` |
| Model-backed (LLM judge) | `nemoguardrails/library/content_safety/` | none; configurable model bound via `Binding.model_param("model_name", "model")` |

Use `Binding.model(...)` for a model type fixed by the manifest, as in
`nemoguardrails/library/llama_guard/rail.py`. Both model constructors set
`resource="model"`; ordinary `surface_param` and `literal` bindings do not.
IORails uses that resource to reject unconfigured models before execution.
`RailsConfig` also validates model bindings on global and per-tool rails, so
using an ordinary binding for a model would bypass that validation.

For an HTTP rail, pick the layout by complexity: `clavata/` factors its vendor
client into `request.py` for a multi-endpoint API, while
`nemoguardrails/library/f5/` inlines a single `http_call` in `actions.py` and
returns `RailOutcome` directly. Use those layouts and their behavior-test
shapes, but use `hf_classifier/backends.py` for current retry/instrumentation
composition. Do not copy F5's deterministic retry random source or Clavata's
payload-bearing exception strings; both predate the current resilience and
telemetry contracts in Step 4.

Use Clavata's separation between request handling, typed vendor errors, and
response-model tests; `tests/test_clavata_models.py` is a solid example of
validating a vendor payload shape. Rewrite error messages so they identify the
operation and status without response bodies, validation input, or arbitrary
nested exception text. Its rail-behavior coverage is NOT exemplary. Copy
`tests/test_f5_guardrails.py` and
`tests/recorded/rails/library/test_f5_guardrails.py` for the behavior test
shape, applying the archetype conditions in Step 6.

For a model-backed rail, the model call is not `http_call`. Use the shared
generation surface, in this fixed order (exemplar:
`nemoguardrails/library/content_safety/actions.py`):

```python
from nemoguardrails.context import llm_call_info_var
from nemoguardrails.llm.call import llm_call, warn_if_truncated
from nemoguardrails.llm.taskmanager import LLMTaskManager
from nemoguardrails.logging.explain import LLMCallInfo

prompt = llm_task_manager.render_task_prompt(task=task, context={...})
stop = llm_task_manager.get_stop_tokens(task=task)
max_tokens = llm_task_manager.get_max_tokens(task=task)
llm_call_info_var.set(LLMCallInfo(task=task))
llm_response = await llm_call(llm, prompt, stop=stop, ...)
result = llm_task_manager.parse_task_output(task, output=llm_response.content)
```

`llm_task_manager` is injected by the runtime (see Step 3). Setting
`llm_call_info_var` before the call is what attributes the call in tracing
and the explain log; skip it and the rail's model call becomes invisible.
Use `warn_if_truncated` when the response can hit the token limit. Repeated
judgments can go through the `model_caches` action parameter, which the
runtime also registers.

Every rail requires:

- `rail.py` (the manifest; discovery keys off this file)
- `__init__.py` (empty package marker)

Executable rails also require:

- `actions.py` (`@action` functions returning `RailOutcome`)
- `rail_config.py` (pydantic config models + `build_config_spec()`), only
  when the rail has configurable options
- `flows.co` (Colang 2.x) and `flows.v1.co` (Colang 1.0), only when the rail
  exposes Colang flows

Config-only manifests such as `factchecking` legitimately omit actions and
flows. Their manifest, config schema, docs, and tests must still describe and
exercise the configuration they contribute.

Tool-only rails such as `cel` and `tool_safety_check` are executable through
IORails without Colang files. They declare `TOOL_CALL`/`TOOL_RESULT` surfaces
and `RailFlows(files=(), v1_files=(), flow_names=(...))`. Use their IORails
integration tests instead of adding Colang files or both-dialect tests for
surfaces LLMRails cannot execute.

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
- `RailSpec.actions`: a `RailActions(refs=(...))` wrapping one
  `ActionRef(name=..., target="module:function")` per action, as
  `actions=RailActions(refs=(DETECT_REGEX_PATTERN,))` in
  `nemoguardrails/library/regex/rail.py`. Passing a bare tuple of `ActionRef`
  fails at manifest construction. The ref `name` MUST equal the name the
  `@action` decorator registers; enforced by
  `test_builtin_action_refs_match_decorated_names_and_bindings`.
- `RailSpec.flows`: `RailFlows(flow_names=(...), files=(...), v1_files=(...))`
  naming the flows the rail exposes and the two dialect files. Every rail
  that ships `.co` files sets this, and the flow-files gate reads it to find
  the files to parse. A config-only rail with no flow files omits it
  (`nemoguardrails/library/factchecking/rail.py`), and the gate skips such
  manifests rather than failing them.
- `RailSpec.surfaces`: one `RailSurface` per entry point, with `name`,
  `direction` (input/output/retrieval/tool_call/tool_result), the action, and
  `bindings`. Use `Binding.context(...)` for message, retrieval, or tool data;
  `Binding.literal(...)` for fixed non-model params; `Binding.surface_param(...)`
  for configurable non-model params; and `Binding.model(...)` or
  `Binding.model_param(...)` for model types. IORails supplies the context keys
  `user_message`, `bot_message`, `tool_call`, `tool_result`, and
  `tool_definition`; required retrieval-context bindings are unsupported there.
  Every surface's action must be in
  `actions.refs`, and every binding's `action_param` must be a real parameter
  of the action. A transform surface must also set `transform_target`.
- `RailSpec.config_schema`: a `RailConfigSchema` with `key` plus a
  `ConfigSpecRef` to `rail_config:build_config_spec`. Rails without configurable
  options omit both this and `rail_config.py`.
- `RailRequirements`: declare `env_vars` (e.g.
  `EnvVar(name="CLAVATA_API_KEY", required=True)`), `services`
  (`ServiceRequirement(name=...)`), `models`
  (`ModelRequirement(type=...)`, keyed by type, not name), `extras`, and
  `optional_dependencies` truthfully; this is what users and tooling see
  (Step 5). Self-check by grepping `actions.py` for every `os.getenv` and
  `os.environ` read and for every import inside a function, then confirming
  each appears here and in the docs install line.
- `RailPrivacy`: `sends_user_text`, `sends_bot_text`, and
  `sends_retrieved_chunks` must match the surfaces' bindings;
  `sends_tool_call_text` and `sends_tool_result_text` cover tool data sent to
  a remote judge or vendor;
  `remote_services` must be non-empty for any rail that makes an HTTP call;
  `data_retention` must be stated whenever the vendor states one, here and
  not only in the docs page. A defaults-only `RailPrivacy()` on a vendor rail
  is almost always wrong.

The catalog rejects duplicates at construction: manifest name, config key,
declared flow name, action name, and `(direction, surface name)` must all be
unique across the library, and a surface whose action its own manifest does
not declare is rejected (`nemoguardrails/manifests/catalog.py`).

## Step 3: Actions and the outcome contract

Actions live in `actions.py` and return `RailOutcome`:

```python
from nemoguardrails.actions import action
from nemoguardrails.actions.rail_outcome import RailOutcome

@action(is_system_action=True)
async def my_rail_check(source: str, text: str, config: RailsConfig, **kwargs) -> RailOutcome:
    if violation_found:
        return RailOutcome.block(reason="matched policy X", metadata={"rule_id": rule.id})
    return RailOutcome.allow()
```

The signature is the injection contract. IORails supplies its fixed request
dependencies by name: `events`, `config`, `llm_task_manager`,
`llm`, `llms`, `model_caches`, and `http_client`. It does not support
`register_action()` or `register_action_param()`. LLMRails additionally
supplies `context`, `kb`, and application-registered action parameters; a per-action
`<action_name>_llm` registration overrides `llm` there. Unit tests that use
`TestChat` exercise this LLMRails registration path when they inject an HTTP
double. Anything else an action needs must come from a manifest `Binding`: in
the example above, `source` is not injected, it is
`Binding.literal("source", "input")` in `rail.py`, and `text` is
`Binding.context("text", "user_message")`.
IORails resolves explicit context bindings, but does not inject the entire
`context` dictionary. Bind every message or tool value the action needs rather
than relying on an LLMRails-only `context` fallback.

`config` is the whole `RailsConfig`, not your section. Read your own
settings at `config.rails.config.<config_schema key>` and handle `None`
(exemplar: `nemoguardrails/library/regex/actions.py`).

The `is_system_action` flag defaults to `False`. Setting it to `True`, as in
the example above, omits the action's start and finish events from LLMRails'
rendered prompt history and keeps it local when an actions server is configured.
The `content_safety`, `clavata`, and `injection_detection` exemplars do not set
it; copy their scoped patterns rather than assuming their dispatch choice is
the template's. Choose the flag deliberately for a new action, and preserve an
existing action's dispatch contract unless the task changes it. IORails invokes
manifest actions locally and does not use this flag. No gate checks the choice.

The seam here is **actions decide, flows present**: the action returns a
neutral verdict (allow/block/transform plus evidence in `metadata`), and the
flow owns everything a user sees -- refusal wording, exception-vs-bot-message,
localization. When you are unsure where a new piece of logic goes, ask which
side of that line it is on. Anything a human reads belongs in the flow, never
in the outcome.

One exception exists in the tree and you should not copy it:
`content_safety`'s `detect_language` action returns localized refusal prose
from a `DEFAULT_REFUSAL_MESSAGES` table, which predates this seam. Refusal
strings and localization belong in the flow or in bot message definitions.

- The three decisions are `allow`, `block`, and `transform`
  (`RailOutcome.transform(rewrites=[(TransformTarget.RELEVANT_CHUNKS, new_text)])`).
  Transforms are required iff the decision is TRANSFORM. `TransformTarget`
  has two legitimate import paths by design: `actions.py` imports it from
  `nemoguardrails.actions.rail_outcome`, while `rail.py` must take it from
  `nemoguardrails.manifests` because that is the only module `rail.py` may
  import from.
- Put neutral evidence in the single `metadata` mapping argument, meaning
  machine-shaped values only: identifiers, category names, scores, booleans,
  and the vendor's own parsed response. `allow` and `block` are keyword-only
  and take `reason` and `metadata` and nothing else; `transform` takes
  `rewrites` positionally and then the same two keyword-only arguments. So
  evidence goes inside the mapping, not as loose keyword arguments. Do NOT put refusal
  text, exception types, or presentation decisions in the outcome; engines own
  presentation.
- Flows consume the outcome via `$response.is_blocked`,
  `$response.is_transform`, and `$response.transform_text["<context var>"]`;
  copy `regex/flows.co` and `regex/flows.v1.co` for the idiom.
- Keep `flows.co` and `flows.v1.co` semantically identical, checked as four
  concrete items, since no gate compares them beyond flow-name presence:
  (1) both files declare the same set of flow names; (2) each same-named
  flow invokes the same action with the same argument bindings; (3) each has
  the same branch structure over `is_blocked`, `is_transform`, and
  `enable_rails_exceptions`; (4) the both-dialect block test from Step 6
  asserts the same output in both. Never
  reference a metadata key the action does not actually set. If you include
  the `enable_rails_exceptions` branch, build the exception message only
  from metadata the action provides, and cover that branch with a test; an
  untested exceptions path that reads a missing key is a latent crash.

Config models go in `rail_config.py`: pydantic models subclassing
`RailConfigBaseModel` plus `build_config_spec() -> RailConfigSpec`. Import
these from `nemoguardrails.manifests.config_schema`, which also provides
`Field` and `rail_field` for declaring the fields themselves; this is a
different module from the `nemoguardrails.manifests` restriction that
applies to `rail.py`. When the spec sets a `key`, it must match the
manifest's `config_schema.key` (enforced in
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
  down. Compose retry with `ensure_retrying_http_client`, which preserves an
  existing retry policy, places retry inside instrumentation, and rejects
  duplicate retry or instrumentation layers:

```python
def _retrying_http_client(client: HTTPClient) -> ClosableHTTPClient:
    return ensure_retrying_http_client(client, _RETRY_POLICY)

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
  composed closable client (a `ClosableHTTPClient`) for the fallback path,
  built with `create_http_client` from `nemoguardrails.http`. Import
  `ensure_retrying_http_client` from `nemoguardrails.http.composition`; do not
  construct `RetryingHTTPClient(injected_client, policy)` directly.
- Retries are rail-owned: define the `RetryPolicy` in one named module-level
  place, either a constant when it is fixed (`_CLAVATA_RETRY_POLICY` in
  `nemoguardrails/library/clavata/request.py`) or a single builder function
  when it is derived from the rail config (`_retry_policy` in
  `nemoguardrails/library/f5/actions.py`). Never construct it inline at the
  call site. The default policy never retries POST; if your vendor call is a
  POST and safe to resend, you must opt in explicitly with
  `retryable_methods=frozenset({"POST"})`, and you may do so only when you
  can name the resend-safety mechanism next to the policy itself, in
  one of exactly three forms: the vendor documents the endpoint as
  idempotent (link the doc), the request carries an idempotency key (name
  the header), or the endpoint is a stateless scan or classification call
  that changes nothing server-side. No comment means no POST retry. If
  you find yourself writing `for attempt in range(...)`, an `asyncio.sleep`
  backoff, or a manual timeout race in the action or request helper, stop:
  that is resilience, so declare it in the `RetryPolicy` and compose the
  client instead of implementing it by hand. Some shipped policies predate
  this documentation requirement; do not copy that omission.
- Know what `RetryingHTTPClient` already gives you before documenting or
  reimplementing anything. Full-jitter exponential backoff: each sleep is a
  uniform random value in `[0, min(initial_delay * 2**retries, max_delay)]`,
  so the lower bound is zero, not `initial_delay`. The `Retry-After` response
  header is honored only when its value already falls within
  `max_retry_after` (default 60s); a larger value is DISCARDED in favor of
  backoff unless you set `clamp_retry_after=True`, which clamps it instead
  (see `nemoguardrails/library/f5/actions.py`). The `x-should-retry` override
  header is ignored unless `honor_retry_override_header=True`. Do not
  document these semantics from guesswork; read
  `nemoguardrails/http/retry.py`.
- Vendor failure is never a silent allow. A timeout, transport error,
  non-success status, or unparseable payload must either raise a rail error
  type or return `RailOutcome.block`; the default is fail-closed. Offer
  fail-open only when the vendor's users ask for it, and then all five of
  these hold: the switch is a field on the rail's pydantic config model,
  never an env var, so a config reviewer sees it (exemplar: `fail_open` in
  `nemoguardrails/library/f5/rail_config.py`); it defaults to `False`; every
  fail-open return carries an explicit marker in metadata so it is
  distinguishable from a genuine vendor clear (`_fail_open_outcome` in
  `nemoguardrails/library/f5/actions.py`); the fail-open path logs at warning
  level; and it has its own test alongside the fail-closed one. Sixth, the
  docs page must enumerate the failure classes the switch actually covers.
  If the code fails open on any non-success status, say so, including 401
  and 403 from a revoked key; do not write "availability only" unless the
  code excludes 4xx from the fail-open branch. An operator whose key is
  revoked would otherwise run silently unguarded while the docs say that
  cannot happen.
- Telemetry must stay content-free. This applies to EVERY rail, not only
  HTTP ones: never log the checked user or bot text, request or response
  bodies, exception messages containing payloads, URLs with query strings,
  or credentials. URL sanitization is unconditional in the transport and
  error layers, so it already applies to your rail; do not undo it by
  logging the raw URL yourself. Self-check by reading every
  `log.` call and every f-string in a raised error.
- Wrap vendor failures in rail-specific error types rooted at your own base
  exception (the `errs.py` pattern; exemplar
  `nemoguardrails/library/clavata/errs.py`), catching the
  `nemoguardrails.http.errors` types at the call site.
- Security rules the reviewer will check, so satisfy them while writing:
  read the API key from `os.getenv` only, never from a config field that can
  reach logs or snapshots (exemplar: `nemoguardrails/library/f5/actions.py`);
  if the config exposes an `api_url` or `base_url`, use it only as the origin
  of the vendor call with a fixed path suffix, never as a template a caller
  can steer; never interpolate user or bot text into a URL, path, header, or
  shell command, since it belongs in the JSON body; parse responses with
  `response.json()` and nothing else, never pickle or a yaml load over vendor
  bytes; and leave redirect handling at the transport default, which is
  disabled (`nemoguardrails/http/transport.py`).

## Step 5: Vendor Python dependencies

The one hard prohibition: never add the vendor package to
`[project.dependencies]`. It is optional, and a normal install must not pull
it in. Beyond that there are three separate questions, and they have
different answers.

**What the manifest declares.** Always list the distributions in
`RailRequirements.optional_dependencies`, whatever else you do. This is what
users and tooling read.

**Whether to add a named extra.** `pyproject.toml` defines rail-scoped
extras (`sdd`, `gcp`, `jailbreak`, `multilingual`, `cel`), so `pip install
nemoguardrails[<extra>]` is an established install path and adding one for
your rail is legitimate. If you add an extra: mirror it into the `all` extra
in the same commit, since nothing tests that mirror; declare it as
`extras=("<name>",)` alongside `optional_dependencies`; and use it as the
docs install line. Note the tree is inconsistent here, so do not infer the
rule from the nearest rail: `sensitive_data_detection` and `cel` declare
`extras=`, while `injection_detection` and `gcp_moderate_text` declare
`optional_dependencies` alone even though `jailbreak` and `gcp` exist for
them.

**Whether CI can run your tests.** Separate from both of the above. If your
unit test module imports the vendor package at top level, add it to the
`test_integration` dependency group and regenerate `uv.lock` in the same
commit; `dev` includes that group, so CI installs it. That is why
`yara-python` is in `test_integration`:
`tests/test_injection_detection.py` does a bare `import yara`. If you would
rather leave packaging untouched, use one of the patterns already in the
tree rather than inventing one: `check_optional_dependency` from
`nemoguardrails/imports.py`, a `setup_module` skip
(`tests/test_sensitive_data_detection.py`), a try/except flag with
`@pytest.mark.skipif` (`tests/test_gcp_text_moderation_input_rail.py`), or
`mock.patch.dict("sys.modules", {"<pkg>": None})` to exercise the
missing-package path with nothing installed
(`tests/test_hf_classifier.py`). Whichever you pick, say so in the PR: a
skipped test suite is not a passing one.

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
  expressible here; state them in the docs page `pip install` line. Set
  `extras=("<extra>",)` alongside it when the rail has a named extra
  (exemplar: `nemoguardrails/library/sensitive_data_detection/rail.py`,
  which declares both).

- Import the vendor package lazily in `actions.py` behind a module-level
  guard that leaves the name bound to `None` on ImportError, then check
  availability inside the action and raise with the exact `pip install` line
  (exemplar: the `yara = None` / `try: import yara` guard and
  `_check_yara_available` in
  `nemoguardrails/library/injection_detection/actions.py`). Never import the
  vendor package at module top level in a way that breaks
  `import nemoguardrails`, and never in `rail.py` (Step 2 rule).

## Step 6: Tests

Three layers apply. Catalog coverage and a recorded-suite entry are required
for every rail; behavioral cases within the unit and recorded layers depend on
the rail's declared actions, flows, surfaces, and provider usage.

1. **Catalog gates (free).** `tests/rails/llm/test_builtin_rail_manifests.py`,
   `tests/rails/llm/test_library_flow_files.py`, and
   `tests/rails/llm/test_builtin_rail_conformance.py` pick up the new rail
   automatically: manifest, lazy refs, action names, bindings, both dialect
   flow files parsing, invoking only actions their own manifest declares
   (`test_library_flow_actions_are_declared_by_owning_manifest`), never
   awaiting a registered action's snake_case name as a Colang 2 flow, which
   must be `CamelCaseAction`
   (`test_library_flows_do_not_invoke_actions_as_flows`), and
   structural conformance (surfaces declare `RailOutcome`, requirement
   identifiers are unique, privacy services are declared requirements, and
   the projected config schema matches `schemas/rails_config.snapshot.json`).
   They do not compare implementation or docs with env vars, imports, outbound
   calls, or retention claims; verify those manually. Run the gates first;
   they catch most wiring mistakes. If a config change moves the schema, regenerate the
   snapshot with `scripts/generate_rails_config_schema_snapshot.py` and
   review the diff. Never weaken a generic check or add your rail to an
   exception list (`LEGACY_UNMANIFESTED_PACKAGES`,
   `NON_PORTABLE_DECLARED_FLOWS`) to get green; fix the manifest.

   **Pinned contracts you must update deliberately.** The catalog tests also
   pin inventories of model bindings, retrieval inputs, and other shipped
   contracts. Extend applicable expected inventories without weakening their
   assertions. This differs from changing a generic check or adding an exception
   to admit an invalid rail. Other exact expectations include:
   - `tests/rails/llm/test_builtin_rail_manifests.py`: include new model-resource
     surfaces in `test_model_dependent_surfaces_declare_model_bindings` and
     extend applicable pinned binding contracts.
   - `tests/rails/llm/test_config.py`: config key and exported model, when the
     rail contributes typed configuration.
   - `tests/test_runtime_flow_gate_equivalence.py`: one case per
     LLMRails-supported manifest surface, including transforms; tool-call and
     tool-result surfaces are excluded.
   - `tests/guardrails/test_compiled_rail.py`: update the pinned servable or
     refused surface inventory for each input/output surface.
   - `tests/guardrails/test_guardrails.py`: add each IORails-admitted input/output
     surface to `ADMITTED_SURFACES`; an unsupported surface instead needs an
     explicit, reviewed refusal reason. That inventory excludes tool surfaces.
   - `tests/guardrails/test_rails_manager.py`: verify that compiled rails borrow
     the runtime-owned HTTP client. This is a shared-client contract, not a
     per-surface registry to populate.
   - `docs/telemetry.mdx`, checked by
     `tests/telemetry/test_usage_reporting.py`: add the manifest's normalized
     built-in feature identifier.

   Also run `tests/llm/test_call_import_graph.py`; it discovers manifest
   action modules automatically and rejects library actions that import
   Colang. A green subset is not proof that the relevant cases ran: inspect
   skips for missing optional dependencies.

   For tool surfaces, exercise `RailsConfig.from_content` with both global
   `flows` and `per_tool` configurations. Model bindings must reject missing
   configured model types, including literal model selections; optional bindings
   may remain unset. See `TestToolRailModelReferences` in
   `tests/test_rails_config.py` for the current validation contract.
2. **Unit tests** in `tests/test_<rail>*.py`:
   - `TestChat` end-to-end for flow behavior, `FakeLLMModel` for
     deterministic main-model output.
   - Direct action-level tests parametrized over allow/block/transform and
     config-error paths (exemplar: `tests/test_injection_detection.py`).
   - Every field in `rail_config.py` needs an action-level test at a
     NON-default value, and any field that can change the decision
     (thresholds, policy lists, allowlists) needs one test on each side of
     its boundary. A declared config field with no test is an unverified
     public option, and an inverted or off-by-one threshold comparison
     passes every gate.
   - For HTTP rails, inject the recording double instead of monkeypatching:
     `chat.app.register_action_param("http_client", RecordingHTTPClient([...]))`
     (exemplar: `tests/test_activefence_rail.py`; helper in
     `nemoguardrails/testing/http.py`, re-exported from
     `nemoguardrails.testing`). Secrets via `monkeypatch`. Unit tests
     must never reach live services.
   - For an executable provider-backed rail, include one engine-level test of
     what happens when the action RAISES (vendor down): fail-closed is a claim
     about the runtime, not your code, so test it rather than asserting it in
     a summary. Use a Colang flow test for LLMRails and an IORails integration
     test for a tool-only rail. IORails converts ordinary action exceptions to
     `RailOutcome.failure`; status-bearing `LLMCallException` and
     `ModelEngineError` propagate. Test both paths when applicable; see
     `tests/guardrails/test_compiled_rail.py` and
     `tests/guardrails/test_tool_safety_check_iorails.py`. Config-only rails have
     no action-error path.
   - Cover EVERY error branch the action can take, not just one. With a
     mocked transport synthetic errors are cheap and deterministic, so this
     is where exhaustive error coverage belongs. Use `RecordingHTTPClient`,
     which is the double that intercepts the httpx-backed
     `nemoguardrails.http` client; `aioresponses` mocks aiohttp and will
     never intercept a library rail, so a test built on it silently attempts
     a real network call. For an HTTP/vendor rail, enumerate the action's
     failure branches and give each a test: timeout, connection error, each
     handled status class (4xx, 5xx), rate limiting (429 retried-then-success
     and 429 retry-exhausted), malformed or unexpected payload, missing
     credential, and the missing optional-dependency path -- each crossed
     with fail-open and fail-closed where the rail supports both. A silently
     swallowed vendor error is the worst failure mode for a guardrail, so
     "one raise test" is not enough.
   - For a flow-bearing rail, cover BOTH Colang dialects end to end, not just the default. `TestChat`
     runs Colang 1 (`flows.v1.co`) unless the config sets
     `colang_version: "2.x"`, which exercises `flows.co` instead (exemplar:
     the `config_v2` fixture and the `test_f5_guardrails_colang_2_*` tests in
     `tests/test_f5_guardrails.py`). Minimum: the block path in both dialects,
     plus the `enable_rails_exceptions` variant wherever the flow has that
     branch. The flow-files gate checks structure only; dialect behavior
     needs these tests.
   - Colang-exposed transform surfaces need both-dialect behavior tests too.
     If the manifest declares such a surface with a `transform_target`, add a
     `colang_version: "2.x"` test asserting the transformed context variable
     reaches the downstream flow, not just the Colang 1 version. The
     flow-gate equivalence harness builds Colang 1 configs only, so a
     transform surface can otherwise ship with its Colang 2 path never
     executed while still satisfying the block-path minimum above.
   - Tool-only rails need IORails allow/block and error-path integration tests,
     covering the bound tool call, result, definition, and any argument scope.
     See `tests/guardrails/test_per_tool_cel_rails_iorails.py` and
     `tests/guardrails/test_tool_safety_check_iorails.py`. Both-dialect
     requirements apply only to surfaces with Colang flows.
3. **Recorded e2e suite** in `tests/recorded/rails/library/`: add a config
   dir under `configs/<name>/`, a constant in `configs.py`, and a
   `test_<rail>.py` covering the applicable public contract. For LLMRails use
   `check_rails`, `generate_with_fake_main`, or `stream_with_fake_main` from
   `helpers.py`; for IORails use `check_iorails`, as in
   `tests/recorded/rails/library/test_tool_safety_check.py`. Assert the result
   with `assert_rails_result` and `snapshot`.
   Provider-backed rails cover allow, block, and one recordable
   provider-error. Pure-Python rails cover their applicable allow/block or
   transform behavior and have no provider-error requirement. Config-only
   manifests cover the public configuration contract they contribute. The
   recorded provider-error is only the RECORDABLE real error
   (an invalid key yielding a real 401 is the canonical one; a real 429 with
   a `Retry-After` header is a common second). Do NOT try to mirror every
   error branch here: a synthetic timeout or 5xx injected by test code
   belongs in the unit layer by the placement rule, and most error paths
   cannot be recorded against a live vendor anyway. Provider-backed rails use `pytest.mark.vcr` and record
   cassettes with `make record-cassettes`; pure-Python rails carry the
   module-level `pytestmark = [pytest.mark.recorded, pytest.mark.asyncio]`
   and no `vcr` mark (exemplar:
   `tests/recorded/rails/library/test_regex.py`). The `asyncio` mark is
   there because these rail tests are async, not because the module is
   non-vcr; synchronous non-vcr modules elsewhere in the suite carry
   `pytest.mark.recorded` alone, and the suite-infrastructure test
   `tests/recorded/test_cassette_provenance.py` carries no `pytestmark` at
   all. Nothing enforces any of this, so follow the rail exemplar rather
   than the nearest neighbour. LLMRails library recordings are
   dialect-single: do not add a Colang 2 duplicate for YOUR rail when the
   wire traffic is identical; dialect behavior is unit-test territory.
   IORails parity tests may reuse the same cassettes and may include a
   deterministic no-cassette case in a module carrying a shared `vcr` mark.
   For what belongs in
   the recorded suite versus `tests/`, follow the `recorded-tests` skill.
   Normalize composite or volatile output before snapshotting; stable scalar
   public outputs may be asserted directly. Leave a new `snapshot()` empty and
   fill it with `--inline-snapshot=create`; `fix` updates an incorrect existing
   value but does not create an empty one. Snapshot updates work only serially,
   never under xdist `make test`. See the README's Snapshots section for the
   exact two-pass behavior. If you do not hold a real vendor
   key, do NOT hand-author a cassette that looks recorded: either ask a
   maintainer to record it, or mark the test `@pytest.mark.fake_cassette` and
   supply the metadata `tests/recorded/README.md` requires (`reason`,
   `frozen_fields`, `fake_llm_model_considered`). Reviewers will ask how each
   new cassette was recorded and will not merge an unverifiable one as
   recorded truth.

## Step 7: Docs and examples

- Author a catalog page at
  `docs/configure-rails/guardrail-catalog/community/<name>.mdx` (exemplars:
  `regex.mdx`, `clavata.mdx`) and set the manifest's `docs_url` to that same
  repo-relative path, extension included, so it ends in `.mdx` (see
  `regex/rail.py`). `test_builtin_manifest_docs_urls_resolve` asserts
  `Path(docs_url).is_file()`, so an `.md` value fails a gate this skill
  tells you to run.
- Add an example config under `examples/configs/<rail>/`.
- Document the required Python packages (the `pip install` line, carrying
  the version bound the manifest cannot express), required env vars / API
  keys, the remote service and what text is sent to it, the vendor's data
  retention period (the same string as `RailPrivacy.data_retention`), and
  known limitations, per the integration rules in `nemoguardrails/AGENTS.md`
  and `docs/AGENTS.md`. State each limitation with the code path or config
  field that causes it, so a reviewer can check the citation.

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
the mirrored copies still agree. `nemoguardrails/library/AGENTS.md` carries
the same matrix as an always-loaded tripwire; its last row is scoped to
services and models, while the row above covers dependencies as well.

## Final verification loop

Run until green, in this order (cheapest first):

```bash
make test TEST="tests/rails/llm/test_builtin_rail_manifests.py tests/rails/llm/test_builtin_rail_conformance.py tests/rails/llm/test_library_flow_files.py"
make test TEST="tests/rails/llm/test_config.py tests/test_rails_config.py tests/test_runtime_flow_gate_equivalence.py"
make test TEST="tests/guardrails/test_compiled_rail.py tests/guardrails/test_guardrails.py tests/guardrails/test_rails_manager.py"
make test TEST="tests/llm/test_call_import_graph.py tests/telemetry/test_usage_reporting.py"
make test TEST=tests/http/test_library_boundary.py
make test TEST=tests/test_<rail>.py
make test TEST=tests/recorded/rails/library ARGS="--block-network -q"
make test          # the focused lines above can be green while the suite is red
uv run --locked pre-commit run --files <changed files>
make docs-fern
```

Report the outcome of every command in this list by name, including any you
could not run and why. A command that was skipped is not a command that
passed; do not summarize the loop as green unless each line above actually
ran.

A rail is contribution-ready only when all of these pass and every
declaration in the manifest (optional_dependencies, env vars, privacy, docs_url)
matches what the code actually does. Reviewers of rail PRs apply the
`review-library-rail` skill; running its judgment dimensions on your own
diff before handoff is the cheapest review you will get.
