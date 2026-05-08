---
title:
  page: Custom LLM Frameworks for NeMo Guardrails
  nav: Custom LLM Framework
description: Replace the LLM framework layer to plug NeMo Guardrails into LiteLLM, an in-house orchestrator, or any non-default LLM stack.
topics:
- Configuration
- Customization
- LLM
tags:
- LLM
- LLMFramework
- DefaultFramework
- Registry
- Python
content:
  type: how_to
  difficulty: technical_advanced
  audience:
  - engineer
  - AI Engineer
---

# Custom LLM Frameworks

NeMo Guardrails has two layers of LLM extensibility: providers and frameworks. Most users only ever touch the provider layer. This guide is for the smaller set of cases where you need to swap the framework layer itself.

## The Two-Layer Model

```
Framework Layer (system-wide, swappable)
|-- DefaultFramework (built-in, all OpenAI-compatible HTTP)
|     |-- openai (provider)
|     |-- nim (provider)
|     |-- ollama (provider)
|     '-- <your custom provider>
|-- LangChainFramework (built-in, opt-in)
|     '-- LangChain providers
'-- <YourCustomFramework>
      '-- <your providers>
```

A **provider** is an engine name inside a framework. `openai`, `nim`, and `ollama` are providers inside `DefaultFramework`; they all speak the OpenAI-compatible chat-completions wire protocol and differ only in default base URLs and small per-provider conventions. Adding a provider is the right move when you want to add or replace one engine and the surrounding framework's behavior is fine. See [Custom LLM Providers](custom-llm-providers.md) and [Custom LLM Model](custom-llm-model.md).

A **framework** owns the entire LLM stack: how models are constructed, how providers are looked up, and how resources are released at shutdown. Adding a framework is the right move when you want to replace the entire stack (for example, route everything through LiteLLM, a proprietary in-house orchestrator, or a service mesh).

| Decision | Pick a provider | Pick a framework |
|---|---|---|
| You need one new engine alongside the existing ones | Yes | No |
| You have one new HTTP backend with custom auth | Yes (subclass `OpenAICompatibleClient` if it is OpenAI-shaped) | No |
| You want all engines to flow through your own gateway | No | Yes |
| You want to disable LangChain entirely and replace it with LiteLLM | No | Yes |
| You want per-call observability hooks across every model | Maybe | Yes if you also need to control construction and shutdown |

In practice almost every customization is a provider. A custom framework is reserved for the cases where you are replacing more than one engine and you need shared lifecycle management across them.

## The LLMFramework Contract

The protocol is {py:class}`nemoguardrails.types.LLMFramework` and is `@runtime_checkable`, so callers can verify a framework with `isinstance(instance, LLMFramework)`. As a Python `Protocol`, it expresses a contract; nothing prevents you from passing an object that duck-types most of it, but the rest of NeMo Guardrails assumes both invariants below hold:

1. The registered object structurally matches the `LLMFramework` protocol (the four methods and their signatures listed below).
2. Its `reset` attribute is an `async` coroutine function. The registry awaits it directly during shutdown / test teardown.

A custom framework implements four methods.

```python
from typing import Any, Dict, List, Optional

from nemoguardrails.types import LLMModel


class MyFramework:
    def create_model(
        self,
        model_name: str,
        provider_name: str,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> LLMModel: ...

    def register_provider(self, name: str, provider_cls: Any) -> None: ...

    def get_provider_names(self) -> List[str]: ...

    async def reset(self) -> None: ...
```

### `create_model`

Called once per `models:` entry in `config.yml` when `LLMRails` builds its task models. `model_name` is the model identifier from `model:`. `provider_name` is the value of `engine:` — a **selector**, not the runtime itself. Your framework uses it to pick which `LLMModel` class to construct; multiple selectors can map to the same runtime (in the built-in framework, `openai`, `nim`, `nvidia_ai_endpoints`, and `ollama` all construct an `OpenAIChatModel`, differing only in their default base URL and API-key environment variable). `model_kwargs` carries everything from the entry's `parameters` block plus a few platform keys like `mode`. Return any object that implements `LLMModel` (see [Custom LLM Model](custom-llm-model.md)).

The framework owns construction. It is free to:

- Cache and reuse expensive resources (HTTP clients, gRPC channels, auth tokens).
- Translate `provider_name` into its own internal taxonomy or fall back to a default URL/credential preset.
- Inject defaults for headers, timeouts, retries.

`DefaultFramework._get_or_create_client` is a worked example of pooled HTTP-client construction keyed off `(base_url, api_key, ...)`, with `provider_name` driving the default-URL lookup.

### `register_provider`

Called by user code (usually from a `config.py`) to add a new selector to this framework. Implementations typically just record the class in an in-memory dict. The framework's `create_model` then dispatches to that class when `provider_name` matches.

`DefaultFramework.register_provider` and `LangChainFramework.register_provider` are both one-line implementations.

### `get_provider_names`

Returns the list of engine names this framework knows about, including built-ins and anything registered at runtime. Used by tooling (`nemoguardrails find_providers`) and for debugging.

### `reset`

```{important}
`reset` MUST be `async`. The registry's `validate()` rejects frameworks whose `reset` is a regular synchronous function with `TypeError: '<name>'.reset must be an async coroutine function`.
```

`reset` is called at process or test boundaries to release framework-owned resources. It must:

- Close any pooled HTTP clients, gRPC channels, file handles, or database connections.
- Clear any registered-provider state if you want a clean slate (some frameworks like `DefaultFramework` separate `aclose` from `clear_providers` and call both from `reset`; others may want to keep registrations).
- Be idempotent: calling `reset` twice in a row must not raise.
- Be safe to call from a running event loop. The registry awaits it directly via `_areset_frameworks`.

After `reset`, the instance must remain usable. New resources are constructed lazily on the next `create_model` call.

Today `reset` is invoked only by the test suite; the runtime does not call it on `nemoguardrails server` shutdown. Implement it for test isolation, not for production cleanup.

## Minimal Working Example

The example below is fully self-contained and runs end-to-end without any
external dependencies. The model is an "echo" implementation that returns a
fixed string for every prompt; swap in real HTTP calls or SDK invocations once
you have verified the registration and dispatch path works (see
`custom-llm-model.md` for the canonical `httpx`-based pattern).

```python
from typing import Any, Dict, List, Optional

from nemoguardrails.llm.frameworks import register_framework, set_default_framework
from nemoguardrails.types import LLMModel, LLMResponse, LLMResponseChunk


class EchoLLMModel:
    """Minimal LLMModel that echoes a fixed string."""

    def __init__(self, model: str, **kwargs: Any):
        self._model = model
        self._default_kwargs = kwargs

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def provider_name(self) -> Optional[str]:
        return "my_engine"

    @property
    def provider_url(self) -> Optional[str]:
        return None

    async def generate_async(self, prompt, *, stop=None, **kwargs) -> LLMResponse:
        return LLMResponse(content=f"echo from {self._model}")

    async def stream_async(self, prompt, *, stop=None, **kwargs):
        yield LLMResponseChunk(delta_content=f"echo from {self._model}")
        yield LLMResponseChunk(finish_reason="stop")


class MyFramework:
    def __init__(self):
        self._providers: Dict[str, Any] = {}

    def create_model(
        self,
        model_name: str,
        provider_name: str,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> LLMModel:
        kwargs = dict(model_kwargs) if model_kwargs else {}
        kwargs.pop("mode", None)
        if provider_name in self._providers:
            return self._providers[provider_name](model=model_name, **kwargs)
        return EchoLLMModel(model=model_name, **kwargs)

    def register_provider(self, name: str, provider_cls: Any) -> None:
        self._providers[name] = provider_cls

    def get_provider_names(self) -> List[str]:
        return sorted({"my_engine", *self._providers})

    async def reset(self) -> None:
        # Release any framework-scoped resources you hold (HTTP clients,
        # connection pools, caches). The echo framework only owns a registry
        # dict, so clearing it is sufficient. A real framework typically
        # closes a shared `httpx.AsyncClient` here.
        self._providers.clear()


register_framework("my", MyFramework())
set_default_framework("my")
```

### Trying it out

Point a NeMo Guardrails config at the framework by setting the model engine
to one the framework recognizes:

```yaml
# config.yml
models:
  - type: main
    engine: my_engine
    model: echo
```

Then run a smoke test:

```python
from nemoguardrails import LLMRails, RailsConfig

# After running the framework registration code above:
config = RailsConfig.from_path("./my_config")
app = LLMRails(config)

result = app.generate(messages=[{"role": "user", "content": "hi"}])
print(result["content"])  # -> echo from echo
```

If the smoke test prints `echo from echo`, the framework is wired up. From
there, replace `EchoLLMModel.generate_async` and `stream_async` with real
backend calls (see `custom-llm-model.md`).

After `register_framework("my", MyFramework())`, the framework is selectable in three ways:

1. **Process-wide default at import time.** Set the environment variable before importing NeMo Guardrails:

   ```bash
   export NEMOGUARDRAILS_LLM_FRAMEWORK=my
   ```

   The registry reads `NEMOGUARDRAILS_LLM_FRAMEWORK` at module load and uses it as the active framework name.
2. **Programmatic flip in `config.py`.** Call `set_default_framework("my")` after registering. All subsequent `LLMRails` constructions use it.
3. **Targeted dispatch.** If you want different frameworks for different model entries, route via `framework.create_model` directly in your own initialization code (advanced; not the standard path).

`config.yml` entries do not name the framework; they name a provider. The framework is implicit in whichever one is active.

```yaml
models:
  - type: main
    engine: my_engine
    model: my-flagship-model
    parameters:
      temperature: 0.2
```

## Reference Implementations

Read these to see production-grade frameworks:

- [`nemoguardrails/llm/frameworks/default.py`](https://github.com/NVIDIA-NeMo/Guardrails/blob/develop/nemoguardrails/llm/frameworks/default.py): `DefaultFramework`. Pools `OpenAICompatibleClient` instances keyed on `(base_url, api_key, timeouts, headers, query)`. Splits lifecycle into `aclose` (HTTP teardown), `clear_providers` (registry teardown), and `reset` (both, used in tests).
- [`nemoguardrails/integrations/langchain/llm_adapter.py`](https://github.com/NVIDIA-NeMo/Guardrails/blob/develop/nemoguardrails/integrations/langchain/llm_adapter.py): `LangChainFramework`. Defers to `nemoguardrails.integrations.langchain.providers` for registration, calls `init_langchain_model` for construction, wraps the result in `LangChainLLMAdapter`. Has a no-op `reset` because the LangChain side has no pooled state of its own.
- [`nemoguardrails/llm/frameworks/registry.py`](https://github.com/NVIDIA-NeMo/Guardrails/blob/develop/nemoguardrails/llm/frameworks/registry.py): `LLMFrameworkRegistry`, `register_framework`, `get_framework`, `set_default_framework`, `get_default_framework`, `_areset_frameworks`. Read this to understand the env var, lazy lookup, and registration behavior.

## Failure Modes

The registry's `validate()` is your friend. It catches the two most common authoring mistakes at registration time, before any model is constructed.

### Sync `reset` raises `TypeError`

```python
class BadFramework:
    def reset(self):  # missing async
        ...

register_framework("bad", BadFramework())
# TypeError: Framework 'bad'.reset must be an async coroutine function.
```

The check uses `inspect.iscoroutinefunction(getattr(item, "reset", None))`. A regular `def reset(self): ...` fails it. So does an `async def` method that has been wrapped by a non-coroutine decorator (rare, but worth knowing).

### Object does not implement the protocol

```python
class NotAFramework:
    pass

register_framework("nope", NotAFramework())
# TypeError: Framework 'nope' does not implement LLMFramework. Required methods: create_model, get_provider_names, register_provider, reset.
```

`@runtime_checkable` Protocols verify by attribute name and signature compatibility. Missing any of `create_model`, `register_provider`, `get_provider_names`, or `reset` triggers this.

### Registering a provider before any framework is active

`register_provider` from `nemoguardrails.llm.providers` resolves the active framework via `get_default_framework()` and calls `framework.register_provider` on it. The registry has a built-in `default` framework that is constructed lazily on first access, so this almost always works without explicit setup. The failure mode appears only when the user sets `NEMOGUARDRAILS_LLM_FRAMEWORK` to a name that has not been registered yet:

```bash
export NEMOGUARDRAILS_LLM_FRAMEWORK=my
```

```python
# config.py runs BEFORE `register_framework("my", ...)`
from nemoguardrails.llm.providers import register_provider

register_provider("echo", EchoLLMModel)
# KeyError: Unknown framework 'my'. Available: ['default', 'langchain']
```

The fix is simple: register the framework before any provider, or keep `NEMOGUARDRAILS_LLM_FRAMEWORK` unset until after `register_framework` has run.

### Unknown framework on activation

```python
set_default_framework("typo")
# KeyError: Unknown framework 'typo'. Available: ['default', 'langchain']
```

The error lists the framework names that the registry currently knows about. Both built-in names appear because they are constructed lazily by factory; if you are working with only your own framework, register it first then call `set_default_framework`.

## Best Practices

1. **Treat `reset` as a hard contract, not a hint.** Test it. Pooled HTTP connections that survive across tests cause surprising flakes elsewhere.
2. **Prefer composition over inheritance.** `MyFramework` does not need to subclass `DefaultFramework`. The protocol is small enough to implement from scratch.
3. **Cache HTTP clients on the framework, not on the model.** Models are constructed per task; clients should be reused. `DefaultFramework._get_or_create_client` shows the keying strategy.
4. **Do not import LangChain in a default-framework-style implementation.** The whole point of swapping the framework layer is to avoid pulling in dependencies you do not need. Keep your imports tight.
5. **Document your framework's provider taxonomy.** `get_provider_names` is what `nemoguardrails find_providers` shows users.

## Related Topics

- [Custom LLM Model](custom-llm-model.md) - Implement the `LLMModel` protocol that your framework constructs.
- [Custom LLM Providers](custom-llm-providers.md) - LangChain `BaseLLM`/`BaseChatModel` providers (uses `engine: langchain`).
- [Init Function](init-function.md) - Where `register_framework` and `set_default_framework` calls usually go.
- [Configuration Reference](../configuration-reference.md) - `config.yml` schema and the `engine`, `model`, and `parameters` fields.
