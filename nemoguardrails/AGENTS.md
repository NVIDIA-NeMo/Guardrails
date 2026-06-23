# AGENTS.md

Subtree guidance for the `nemoguardrails/` package. This supplements the
repository-root `AGENTS.md`, `CONTRIBUTING.md`, and `AI_POLICY.md`; all
root-level rules still apply. The rules below are the runtime, public-API, and
provider-integration specifics that matter when editing this package.

## Architecture Map

- `Guardrails` (`guardrails/guardrails.py`) is the modern entry point; it
  delegates to `IORails` (`guardrails/iorails.py`, input/output rails dispatched
  through the engine registry) or to the legacy `LLMRails`
  (`rails/llm/llmrails.py`, the full event-driven Colang pipeline).
- `RailsConfig` (`rails/llm/config.py`) is the user-facing config, loaded via
  `from_path`/`from_content`.
- Colang has two runtimes, 1.0 (`colang/v1_0/`) and 2.x (`colang/v2_x/`),
  dispatched by `colang/__init__.py`; actions resolve through
  `actions/action_dispatcher.py`.
- LLM access goes through the framework/provider abstraction (`llm/frameworks/`,
  `types.py`): the default OpenAI-compatible client or the LangChain framework.
- Built-in rails live in `library/` (optional, lazily imported); the FastAPI
  server is in `server/`; request-scoped state lives in `context.py`.

## Code Changes

- Preserve public APIs unless the task explicitly changes them. Treat public
  imports, constructor signatures, documented methods, config schemas, server
  request/response shapes, Colang behavior, examples, and shipped defaults as
  compatibility-sensitive.
- Keep sync and async API behavior aligned for `LLMRails`, `Guardrails`, and
  related public methods.
- Keep optional providers, frameworks, extras, and secret-bearing integrations
  optional. Do not move integration dependencies into the default install path
  unless the task is specifically about packaging policy.
- When changing API request or response shapes, keep required fields explicit
  and never mirror API keys, credentials, or provider secrets back in response
  bodies. Secrets belong in headers, environment variables, or local
  configuration paths.
- Route LLM calls through existing framework/model abstractions and helpers such
  as `nemoguardrails.actions.llm.utils.llm_call` unless the surrounding code
  already establishes a more specific path. Avoid ad hoc provider calls that
  bypass shared parameter handling, tracing, metrics, or streaming behavior.
- For OpenAI-compatible providers, prefer the built-in default framework and
  OpenAI-compatible client path. Use the LangChain framework only for engines
  that need LangChain or when the task explicitly changes LangChain behavior.
- Treat HTTP header names as case-insensitive. Only normalize or compare header
  values case-insensitively when the relevant HTTP spec or provider contract
  defines them that way.
- Avoid broad filesystem walks, import-time side effects, and global state
  changes in runtime paths unless the surrounding code already establishes that
  pattern.
- Wrap provider/LLM failures in the domain exceptions in
  `nemoguardrails/exceptions.py` (`LLMCallException`, `LLMClientError`
  subclasses) and re-raise with `from`; do not raise bare exceptions.
- Use a module-level `log = logging.getLogger(__name__)`; never `print`.
- Sync public methods delegate to their `_async` twin via
  `get_or_create_event_loop()` and must raise if called inside a running loop;
  keep the logic in the async method.
- The public surface is what top-level `__all__` exports. Domain types in
  `types.py` are plain dataclasses (keep them dependency-free); config models use
  Pydantic with `@model_validator` and `ConfigDict(extra="forbid")`.
- Deprecate with `warnings.warn(..., DeprecationWarning, stacklevel=...)` and
  keep the old path working.

## NeMo Guardrails Invariants

- Preserve non-text model metadata such as reasoning content, usage data, finish
  reasons, request IDs, and streamed metadata chunks. Do not drop reasoning-only
  or usage-only chunks just because `delta_content` or message content is empty.
- Keep observability signals independently configurable. Tracing, metrics, logs,
  and anonymous usage telemetry have different contracts and should not be
  enabled, disabled, or configured as a single implicit bundle.
- Mark experimental behavior clearly in docs and keep it isolated from stable
  contracts.

## Testing

- Test config-driven behavior against a real `RailsConfig` (for example
  `RailsConfig.from_content` with YAML), not `SimpleNamespace` or attribute
  stubs, when adding or wiring a config field. Stubbing the whole config tree
  does not validate the wiring you are adding.
- For metadata or stats propagation changes, assert on the actual propagation
  targets (for example both `LLMCallInfo` fields and `LLMStats` counters) and
  reset every context variable the code path touches in fixtures.
- Mock LLMs with the project's test doubles, not bespoke mocks: `FakeLLMModel`
  (`nemoguardrails/testing/fake_model.py`) for deterministic responses and token
  usage, and the `TestChat` harness (`nemoguardrails/testing/chat_harness.py`,
  `>>`/`<<`) for end-to-end rail tests.
- Mock provider HTTP with `pytest-httpx` (`httpx_mock`) and set secrets via
  `monkeypatch`. There is no global live-test mode; gate any real-network test
  behind an explicit skip.

## Adding A Provider Or Library Integration

When adding a new optional third-party integration (LLM provider, embedding
provider, or similar library):

- Keep the dependency optional: import the third-party package lazily inside
  `__init__` or the method that needs it, wrapped in a `try/except ImportError`
  that names the poetry extra to install. Never import it at module top level in
  a way that breaks core import when the package is absent.
- Add the dependency as an optional dependency under a poetry extra in
  `pyproject.toml`; do not add it to the default runtime dependencies.
- Packaging-patch hygiene: read the current `pyproject.toml` and `poetry.lock`
  first, anchor edits to lines that actually exist, and regenerate the lock with
  project tooling. Do not generate dependency edits from memory or templates; if
  the lock cannot be regenerated, stop and flag that it will be inconsistent.
- Follow the existing provider pattern. For an embedding provider, mimic
  `nemoguardrails/embeddings/providers/openai.py`: subclass the `EmbeddingModel`
  ABC, set `engine_name`, implement `encode()` and `encode_async()`, and register
  the class with one `register_embedding_provider(...)` call in
  `providers/__init__.py`. For an LLM provider, implement the `LLMModel` protocol
  and route through the default OpenAI-compatible framework unless LangChain is
  required.
- Tests must not call the live service: mock the client (`MagicMock`/`AsyncMock`
  or httpx fixtures, as in `tests/test_embeddings_openai.py`); guard any
  real-network test behind an explicit skip (there is no global live-test mode).
- Document the engine/provider name, the required extra, and the expected API
  keys or environment variables, and note whether it uses the default framework
  or LangChain.
- Keep the registration name and constructor signature consistent with sibling
  providers.
