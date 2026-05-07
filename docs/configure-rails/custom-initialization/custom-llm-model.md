---
title:
  page: Custom LLM Models for NeMo Guardrails
  nav: Custom LLM Model
description: Implement the LLMModel protocol to plug a non-OpenAI-compatible backend into NeMo Guardrails without depending on LangChain.
topics:
- Configuration
- Customization
- LLM
tags:
- LLM
- LLMModel
- Protocol
- DefaultFramework
- Python
content:
  type: how_to
  difficulty: technical_advanced
  audience:
  - engineer
  - AI Engineer
---

# Custom LLM Models

NeMo Guardrails defines a small `LLMModel` protocol that every backend implements. The built-in `DefaultFramework` ships an `OpenAIChatModel` for any OpenAI-compatible HTTP endpoint, and the optional `LangChainFramework` ships a `LangChainLLMAdapter` that wraps any LangChain `BaseChatModel` or `BaseLLM`. When neither matches your backend, you can implement `LLMModel` directly.

This guide covers when to do that, the contract you must satisfy, a minimal worked example, and pointers to the reference implementations and to the testing helpers.

## When to Use a Custom LLMModel

There are three options for connecting a backend to NeMo Guardrails. Pick the highest one that fits.

| Backend shape | Recommended path | Where it lives |
|---|---|---|
| OpenAI-compatible HTTP endpoint (vLLM, TGI, OpenRouter, self-hosted, NIM, ...) | Use `engine: openai` (or the matching built-in engine) and set `parameters.base_url` | [Custom LLM Providers](custom-llm-providers.md) and the configuration reference |
| You already have a LangChain `BaseChatModel` or `BaseLLM` | Use `engine: langchain` and register the LangChain class via `register_chat_provider` | [Custom LLM Providers](custom-llm-providers.md) |
| Custom HTTP API that is not OpenAI-shaped, and you do not want a LangChain dependency | Implement `LLMModel` and register it with the active framework | This guide |

Concretely, choose a custom `LLMModel` when:

- Your provider speaks a non-OpenAI wire format and you do not want to depend on LangChain.
- You want full control over retries, headers, streaming parsing, and tool-call accumulation.
- You want a lean install footprint (no `langchain-*` packages) and you control the HTTP layer yourself.

## The LLMModel Contract

The protocol is defined in `nemoguardrails/types.py` (lines 218 to 257). It is `@runtime_checkable`, so the framework registry can verify with `isinstance(model, LLMModel)`.

A custom model class must implement two async methods and three properties.

```python
from typing import AsyncIterator, List, Optional, Union

from nemoguardrails.types import (
    ChatMessage,
    LLMResponse,
    LLMResponseChunk,
)


class LLMModel:
    async def generate_async(
        self,
        prompt: Union[str, List[ChatMessage]],
        *,
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> LLMResponse: ...

    def stream_async(
        self,
        prompt: Union[str, List[ChatMessage]],
        *,
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> AsyncIterator[LLMResponseChunk]: ...

    @property
    def model_name(self) -> str: ...

    @property
    def provider_name(self) -> Optional[str]: ...

    @property
    def provider_url(self) -> Optional[str]: ...
```

### `prompt`

Adapters must accept either a plain string or a list of `ChatMessage` objects. `ChatMessage` is a stdlib dataclass with `role`, `content`, optional `tool_calls`, optional `tool_call_id`, optional `name`, and a `provider_metadata` dict for non-standard fields. Convert messages to whatever shape your SDK expects.

### `stop` and `**kwargs`

`stop` is the canonical name for stop sequences; keep it as a keyword-only argument. `**kwargs` carries everything the caller passed under `parameters` in `config.yml` plus any per-call overrides (`temperature`, `max_tokens`, `top_p`, ...). Forward these to the underlying SDK.

### `generate_async` returns `LLMResponse`

`LLMResponse` is a dataclass in `nemoguardrails/types.py`:

```python
@dataclass
class LLMResponse:
    content: str
    reasoning: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    model: Optional[str] = None
    finish_reason: Optional[FinishReason] = None
    stop_sequence: Optional[str] = None
    request_id: Optional[str] = None
    usage: Optional[UsageInfo] = None
    provider_metadata: Optional[Dict[str, Any]] = None
```

`content` is required and must be a string (use the empty string when the model only produced tool calls). `finish_reason` is one of `"stop"`, `"length"`, `"tool_calls"`, `"content_filter"`, `"error"`, or `"other"`. Populate `tool_calls` only when the response is a function-calling/tool-calling response.

### `stream_async` is an async generator

Implementations must be `async def` generator functions that `yield` `LLMResponseChunk` objects. The protocol's return type is `AsyncIterator[LLMResponseChunk]`. Each chunk has the shape:

```python
@dataclass
class LLMResponseChunk:
    delta_content: Optional[str] = None
    delta_reasoning: Optional[str] = None
    delta_tool_calls: Optional[List[ToolCall]] = None
    model: Optional[str] = None
    finish_reason: Optional[FinishReason] = None
    request_id: Optional[str] = None
    usage: Optional[UsageInfo] = None
    provider_metadata: Optional[Dict[str, Any]] = None
```

Conventions to follow so the rest of the pipeline works:

- Yield text deltas in `delta_content` as soon as they arrive.
- Yield `delta_reasoning` for chain-of-thought tokens emitted before the visible answer (OpenAI reasoning models, NIM `reasoning_content`).
- Tool-call streaming is incremental on the wire: provider chunks usually carry argument fragments. Accumulate them and emit a single completed `delta_tool_calls` list on the chunk whose `finish_reason == "tool_calls"`. The reference `OpenAIChatModel._finalize_tool_calls` shows the pattern.
- Set `finish_reason` only on the final chunk that carries it. Earlier chunks should leave it `None`.
- Emit a final usage-only chunk (no `delta_content`, only `usage` and `request_id`) when the provider sends an end-of-stream usage record. The pipeline tolerates either inline or trailing usage.

### Tool calling

`ToolCall` and `ToolCallFunction` are dataclasses:

```python
@dataclass
class ToolCallFunction:
    name: str
    arguments: Dict[str, Any]


@dataclass
class ToolCall:
    id: str
    type: str = "function"
    function: ToolCallFunction = ...
```

`function.arguments` is a `Dict[str, Any]`, not a JSON string. If your provider returns arguments as a JSON string, `json.loads()` it before constructing the `ToolCall`. If parsing fails for a streamed response, fall back to an empty dict; the tool layer will surface the real error when the function is invoked.

### Properties

- `model_name` returns the concrete model identifier (for example `gpt-4o-mini`, `meta/llama-3.1-70b-instruct`). Used in logs and error contexts.
- `provider_name` returns the engine name as it appears in `config.yml` (for example `openai`, `nim`, `my_engine`). Return `None` only if you genuinely cannot determine it.
- `provider_url` returns the base URL for HTTP backends, or `None` for backends that do not have one (for example a SageMaker endpoint addressed by ARN).

### Error handling

The pipeline expects errors to be normalized. Raise the exception classes defined in `nemoguardrails.exceptions`:

- `LLMConnectionError` for network or DNS failures.
- `LLMTimeoutError` for read or connect timeouts.
- `LLMAuthenticationError` for 401 or 403.
- `LLMRateLimitError` for 429.
- `LLMResponseValidationError` for malformed provider responses.
- `LLMClientError` is the common base if you need a generic fallback.

Populate `model_name`, `provider_name`, and `base_url` on the exception when you raise it so downstream logs are usable. The reference `OpenAIChatModel._enrich` shows the pattern.

## Minimal Worked Example

Below is a 40-line `EchoLLMModel` that returns canned responses without making any network call. It is useful as a starting skeleton and as a sanity check for new framework wiring.

```python
import asyncio
from typing import Any, AsyncIterator, List, Optional, Union

from nemoguardrails.llm.providers import register_provider
from nemoguardrails.types import (
    ChatMessage,
    LLMResponse,
    LLMResponseChunk,
    UsageInfo,
)


class EchoLLMModel:
    """Returns a canned response. Useful as a skeleton or in offline tests."""

    def __init__(self, model: str, response: str = "echo", **kwargs: Any):
        self._model = model
        self._response = response
        self._default_kwargs = kwargs

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def provider_name(self) -> Optional[str]:
        return "echo"

    @property
    def provider_url(self) -> Optional[str]:
        return None

    async def generate_async(
        self,
        prompt: Union[str, List[ChatMessage]],
        *,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        return LLMResponse(
            content=self._response,
            model=self._model,
            finish_reason="stop",
            usage=UsageInfo(input_tokens=0, output_tokens=len(self._response)),
        )

    async def stream_async(
        self,
        prompt: Union[str, List[ChatMessage]],
        *,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[LLMResponseChunk]:
        for token in self._response.split():
            await asyncio.sleep(0)
            yield LLMResponseChunk(delta_content=token + " ", model=self._model)
        yield LLMResponseChunk(model=self._model, finish_reason="stop")


register_provider("echo", EchoLLMModel)
```

Drop that snippet into `config.py` (the file next to your `config.yml`). The `register_provider` call attaches `EchoLLMModel` as the `echo` engine on whichever framework is currently active. By default that is `DefaultFramework`. See [Custom LLMFramework](custom-llm-framework.md) for the framework layer.

Reference it from `config.yml`:

```yaml
models:
  - type: main
    engine: echo
    model: echo-v1
    parameters:
      response: "Hello from echo"
```

### What `register_provider` does

`register_provider(name, cls)` from `nemoguardrails.llm.providers` resolves the active framework via `get_default_framework()` and calls `framework.register_provider(name, cls)` on it. For `DefaultFramework`, that adds `name` to its in-memory dict; subsequent `create_model("echo", ...)` calls use your class as the factory.

If you want the provider to be available regardless of which framework is active, register it once per framework:

```python
from nemoguardrails.llm.frameworks import get_framework

get_framework("default").register_provider("echo", EchoLLMModel)
get_framework("langchain").register_provider("echo", EchoLLMModel)
```

The `LangChainFramework` accepts the same call shape. It treats your class as a chat provider.

### Calling-convention contract for your `__init__`

`framework.create_model(model_name, provider_name, model_kwargs)` calls your class as `EchoLLMModel(model=model_name, **model_kwargs)`. Make `model` a required keyword and accept additional `**kwargs` so that future configuration keys do not break instantiation.

## Reference Implementations

Read these to see a production-grade `LLMModel`:

- `nemoguardrails/llm/models/openai_chat.py`: `OpenAIChatModel` for any OpenAI-compatible HTTP endpoint. Shows tool-call accumulation, reasoning-content extraction, response validation, and exception enrichment. Uses `OpenAICompatibleClient` from `nemoguardrails/llm/clients/openai_compatible.py` for the HTTP layer.
- `nemoguardrails/integrations/langchain/llm_adapter.py`: `LangChainLLMAdapter` that bridges any LangChain `BaseChatModel` or `BaseLLM`. Shows how to map LangChain's `tool_call_chunks`, `usage_metadata`, `response_metadata`, and `additional_kwargs` onto the `LLMResponse` and `LLMResponseChunk` shapes.

Both files import their types directly from `nemoguardrails.types`. Custom models should do the same.

## Testing Your Model

NeMo Guardrails ships a pytest-friendly `FakeLLMModel` under `nemoguardrails.testing` that is shaped exactly like the protocol and accepts a list of canned strings or `LLMResponse` objects:

```python
from nemoguardrails.testing import FakeLLMModel
```

The two recommended approaches:

1. Write unit tests for your `LLMModel` class in isolation: instantiate it, call `await model.generate_async(prompt)`, and assert on the returned `LLMResponse`. No framework needed.
2. Write end-to-end tests with a real `LLMRails` instance by registering a `FakeLLMModel` (or `FakeLLMModel`-style class) as a custom provider in the test's `config.py`, then driving the full pipeline. See [Testing Your Config](../../user-guides/testing-your-config.md) for the full set of helpers (`FakeLLMModel`, `TestChat`, fixtures).

The contract is small enough that property-based tests are straightforward: any string `prompt` and any list of `ChatMessage` objects must produce a non-`None` `LLMResponse.content`, and `stream_async` must always yield a final chunk with a non-`None` `finish_reason`.

## Best Practices

1. **Implement both methods even if your backend has no native streaming.** A simple `stream_async` that yields a single chunk built from `generate_async` keeps the streaming consumer paths working.
2. **Pre-flight validate provider responses.** The reference `OpenAIChatModel._validate_response` rejects non-dict bodies and missing `choices` entries before parsing. This keeps user-facing errors actionable.
3. **Forward `**kwargs` to the SDK.** Anything the user wrote under `parameters` in `config.yml` lands here. Letting unknown keys pass through means new SDK options work without a guardrails release.
4. **Keep `__init__` cheap.** `create_model` is called per task. If your client is expensive to build, cache it on the framework (see `DefaultFramework._get_or_create_client`) or as a module-level singleton.
5. **Do not raise vanilla `Exception`.** Use the `nemoguardrails.exceptions` hierarchy so retries and structured logging behave correctly.

## Related Topics

- [Custom LLM Providers](custom-llm-providers.md) - LangChain `BaseLLM`/`BaseChatModel` providers (uses `engine: langchain`).
- [Custom LLM Framework](custom-llm-framework.md) - Replace the framework layer wholesale, not just one engine.
- [Init Function](init-function.md) - Where `register_provider` calls usually go.
- [Configuration Reference](../configuration-reference.md) - `config.yml` schema, including `engine`, `model`, and `parameters`.
