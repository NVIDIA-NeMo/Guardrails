# Canonical Outbound Clients and Observability

## Status

This note describes the target design for model and HTTP calls made by engines
and rail actions. It defines shared contracts and ownership boundaries; it does
not require every existing integration to migrate at once.

## Context

NeMo Guardrails has two recurring outbound-call patterns:

1. Model calls made by the main generation path or by rail actions.
2. Ordinary HTTP calls made by rails that invoke external safety services.

The canonical model contract already exists. `LLMModel` in
`nemoguardrails/types.py` exposes `generate_async()`, `stream_async()`, and
provider identity. `OpenAIChatModel` implements that protocol on top of
`OpenAICompatibleClient`, and `llm_call()` accepts an `LLMModel`.

IORails currently has a parallel model path. `EngineRegistry` constructs
`ModelEngine` objects, `ModelEngine` makes OpenAI-compatible HTTP requests, and
`EngineRegistry` applies GenAI OpenTelemetry spans and metrics around those
calls. This duplicates model construction, transport, lifecycle, parameter
merging, response parsing, and observability that belong behind the canonical
model interface.

HTTP calls are more fragmented. Library actions independently create
`aiohttp.ClientSession`, `httpx.AsyncClient`, or provider SDK clients. That
produces inconsistent pooling, retries, timeout behavior, errors, privacy
handling, and telemetry. IORails also has `APIEngine`, another request and
lifecycle implementation for external REST APIs.

The desired shape is one canonical contract per kind of outbound operation,
with optional instrumentation added through composition.

## Goals

- Make `LLMModel` the only model-call contract consumed by engines and actions.
- Give `llm_call()` and every engine free GenAI instrumentation when the model
  supplied by the runtime is instrumented.
- Define a transport-neutral `HTTPClient` contract for ordinary REST calls.
- Give actions free HTTP instrumentation when the injected client is
  instrumented.
- Keep OpenTelemetry, provider SDKs, and concrete HTTP libraries out of rail
  action logic.
- Reuse connections and make client lifecycle explicit and runtime-owned.
- Apply consistent retry, timeout, error, redaction, and content-capture rules.
- Preserve custom model and client implementations through structural
  protocols.
- Allow telemetry to be disabled without changing execution behavior.
- Support gradual migration and behavior-equivalence validation.

## Non-Goals

- Treating every HTTP call as a model call.
- Making arbitrary provider SDKs pretend to be raw HTTP clients.
- Globally monkey-patching `aiohttp`, `httpx`, or third-party SDKs.
- Capturing prompts, request bodies, response bodies, or credentials by
  default.
- Replacing rail-level spans with transport spans.
- Requiring OTEL packages when tracing and metrics are disabled.
- Creating one inheritance hierarchy for models, HTTP transports, and service
  SDKs.

## Design Principles

### Instrument operation semantics at the highest shared boundary

An LLM operation has model, provider, token, finish-reason, tool-call, and
streaming semantics. Those are visible at `LLMModel`; a generic HTTP client
cannot reliably infer them. GenAI instrumentation therefore wraps `LLMModel`.

An ordinary REST operation has method, server, URL, status, payload size,
timeout, and retry semantics. Those are visible at `HTTPClient`. Standard HTTP
instrumentation therefore wraps `HTTPClient`.

### Prefer composition over inheritance

`LLMModel` is a runtime-checkable protocol, not a mandatory base class.
`OpenAIChatModel`, LangChain adapters, LiteLLM adapters, fake models, and user
models can all satisfy it without sharing an implementation hierarchy.

Instrumentation should preserve that property:

```text
InstrumentedLLMModel
  -> any LLMModel

InstrumentedHTTPClient
  -> any HTTPClient
```

### Separate behavior from policy

The wrapped client determines how the operation is executed. The wrapper
determines which observations are emitted. Configuration determines whether
tracing, metrics, and content capture are enabled.

### The runtime owns resources

Actions receive clients but do not create, cache, or close shared clients. The
runtime or registry creates them, injects them, and closes owned resources at
shutdown.

## Canonical Model Path

### Existing contract

The canonical model protocol is:

```python
@runtime_checkable
class LLMModel(Protocol):
    async def generate_async(
        self,
        prompt: str | list[ChatMessage],
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> LLMResponse: ...

    def stream_async(
        self,
        prompt: str | list[ChatMessage],
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[LLMResponseChunk]: ...

    @property
    def model_name(self) -> str: ...

    @property
    def provider_name(self) -> str | None: ...

    @property
    def provider_url(self) -> str | None: ...
```

`llm_call()` already validates this contract, converts dictionary messages to
`ChatMessage`, invokes the model, and stores response information in the
existing logging and processing context.

### Instrumented model decorator

GenAI observability belongs in a decorator that also satisfies `LLMModel`:

```python
class InstrumentedLLMModel:
    def __init__(self, model: LLMModel, telemetry: LLMTelemetry):
        self._model = model
        self._telemetry = telemetry

    @property
    def model_name(self) -> str:
        return self._model.model_name

    @property
    def provider_name(self) -> str | None:
        return self._model.provider_name

    @property
    def provider_url(self) -> str | None:
        return self._model.provider_url

    async def generate_async(self, prompt, *, stop=None, **kwargs):
        with self._telemetry.call(self, prompt, stop, kwargs) as call:
            response = await self._model.generate_async(
                prompt,
                stop=stop,
                **kwargs,
            )
            call.record_response(response)
            return response

    async def stream_async(self, prompt, *, stop=None, **kwargs):
        with self._telemetry.stream(self, prompt, stop, kwargs) as call:
            async for chunk in self._model.stream_async(
                prompt,
                stop=stop,
                **kwargs,
            ):
                call.record_chunk(chunk)
                yield chunk
            call.complete()
```

The sketch omits error handling for brevity. The telemetry context records the
exception before re-raising it. It must not translate provider exceptions or
change the returned response.

### What the decorator records

For non-streaming operations:

- GenAI client span.
- Provider and requested model.
- Stable operation name such as `chat`.
- Request parameters that are safe and semantically relevant.
- Response model and response identifier.
- Finish reason.
- Input and output token usage when available.
- Operation duration.
- Error type and span status on failure.
- Prompt and response content only when explicitly enabled.

For streaming operations it additionally records:

- Stream mode on the request span.
- Time to first content-bearing chunk.
- Time between content-bearing output chunks.
- Latest non-null model, response ID, finish reason, and usage information.
- Token metrics after natural stream exhaustion.
- Cancellation and provider errors without manufacturing a successful final
  response.
- Optional accumulated output content when capture is enabled.

The span remains open for the complete async-generator lifetime. Creating a
span when `stream_async()` is called but closing it before iteration begins
would produce incorrect duration, parenting, and error behavior.

### Free instrumentation for actions

Actions should continue to call:

```python
response = await llm_call(model, messages, llm_params=params)
```

`llm_call()` does not need rail-specific OTEL code. When the runtime supplies
an `InstrumentedLLMModel`, the call is observed automatically. The same action
works with an uninstrumented model in tests or applications that disable
telemetry.

Instrumentation at `llm_call()` alone would be insufficient because engines
and integrations can call `LLMModel` directly. Instrumenting the model contract
covers both helper-mediated and direct calls.

### Engine ownership

IORails should store models by configured type:

```python
models: dict[str, LLMModel]
```

The model factory constructs the concrete adapter and applies instrumentation
once:

```text
Model config
  -> OpenAICompatibleClient
  -> OpenAIChatModel
  -> InstrumentedLLMModel when enabled
  -> model registry
```

Main-generation and rail-specific models use the same registry. IORails
converts its public OpenAI-shaped message dictionaries to `ChatMessage` at its
request boundary, not inside each rail action.

Tool definitions, tool results, and tool-exchange validation remain request
and engine concerns. They should not be moved into the telemetry wrapper.

### Custom models

The factory or engine may receive a caller-provided `LLMModel`. It applies the
same instrumentation decorator unless the model is already wrapped or the
caller explicitly disables Guardrails-managed instrumentation.

Wrapping must be idempotent. A helper can return an existing
`InstrumentedLLMModel` unchanged instead of nesting duplicate GenAI spans.

Provider SDK auto-instrumentation can coexist with the Guardrails semantic
span only if the spans describe different layers. Configuration must allow
transport spans to be disabled when a provider integration would otherwise
emit duplicate client spans.

### Model lifecycle

`LLMModel` intentionally describes inference, not ownership. It currently has
no mandatory `close()` method. The registry therefore tracks ownership
separately:

```python
@dataclass
class ModelHandle:
    model: LLMModel
    close: Callable[[], Awaitable[None]] | None = None
```

A runtime-created OpenAI client has a close callback. A caller-owned custom
model does not unless ownership was explicitly transferred. The instrumentation
decorator delegates lifecycle but never assumes it owns the wrapped model.

## Canonical HTTP Path

### Why a separate contract is needed

An ordinary REST call does not have token usage, model identity, finish reason,
or GenAI content semantics. Reusing `LLMModel` or GenAI instrumentation would
produce misleading telemetry.

The HTTP contract should be small enough for actions, fake clients, `httpx`,
and `aiohttp` adapters to implement without exposing a concrete library.

### HTTP request contract

The first version should support non-streaming requests because that covers the
existing library API-call pattern. Model-specific SSE remains behind
`OpenAICompatibleClient` and `LLMModel.stream_async()`. A generic streaming HTTP
contract can be added when a non-model rail requires it.

```python
@runtime_checkable
class HTTPClient(Protocol):
    async def request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        params: Mapping[str, Any] | None = None,
        json: Any = None,
        content: bytes | str | None = None,
        timeout: float | None = None,
    ) -> HTTPResponse: ...
```

The neutral response owns decoded bytes and exposes convenience parsing without
leaking an `httpx.Response` or `aiohttp.ClientResponse`:

```python
@dataclass(frozen=True)
class HTTPResponse:
    status_code: int
    headers: Mapping[str, str]
    content: bytes

    @property
    def text(self) -> str: ...

    def json(self) -> Any: ...
```

The transport raises neutral errors such as:

- `HTTPConnectionError`
- `HTTPTimeoutError`
- `HTTPStatusError`
- `HTTPResponseDecodeError`

LLM adapters translate transport errors into `LLMClientError` variants with
model and provider context. Ordinary API actions receive the neutral HTTP
errors.

### Lifecycle contract

Actions only need `HTTPClient`. The runtime needs a managed extension:

```python
class ManagedHTTPClient(HTTPClient, Protocol):
    async def close(self) -> None: ...
```

An action must not close an injected client. The registry closes only clients
it owns.

### Composition

The default composition is:

```text
InstrumentedHTTPClient
  -> RetryingHTTPClient
    -> HttpxHTTPClient
```

The order is deliberate. An outer instrumentation decorator creates one span
for the logical operation. The retry decorator performs one or more transport
attempts inside it. The final span records the retry count and final result.

The opposite order would create a separate peer HTTP span for every attempt.
That can be useful for deep diagnostics but is too noisy as the default and
makes one application operation look like several unrelated calls.

Retry events may be added to the logical span with attempt number, delay, and
safe status information. They must not include bodies or credentials.

### Instrumented HTTP decorator

```python
class InstrumentedHTTPClient:
    def __init__(self, client: HTTPClient, telemetry: HTTPTelemetry):
        self._client = client
        self._telemetry = telemetry

    async def request(self, method, url, **kwargs):
        with self._telemetry.request(method, url, kwargs) as call:
            response = await self._client.request(method, url, **kwargs)
            call.record_response(response)
            return response
```

The decorator records standard HTTP client information:

- Request method.
- URL scheme.
- Server address and port.
- Sanitized route or URL without credentials and query values.
- Response status code.
- Request and response sizes when known.
- Operation duration.
- Retry count when exposed by the retry layer.
- Error type and span status.

It must not emit `gen_ai.*` attributes. When an instrumented model uses an
instrumented HTTP transport, the trace may contain two nested spans with
different meanings:

```text
rail/action span
  -> GenAI model-operation span
    -> HTTP transport span
```

The HTTP child span is optional. Deployments that already instrument `httpx`,
`aiohttp`, or a provider SDK can disable Guardrails-managed transport spans
while retaining the GenAI operation span.

### HTTP helper for actions

The recommended action API mirrors `llm_call()`:

```python
response = await http_call(
    http_client,
    "POST",
    endpoint,
    headers=headers,
    json=payload,
    timeout=timeout,
)
```

`http_call()` provides a stable public entry point, normalizes invocation, and
integrates with existing action logging and request context. Instrumentation
still belongs to the client decorator, so direct `HTTPClient.request()` calls
remain observable.

An action becomes independent of the concrete transport:

```python
@action()
async def check_text(
    text: str,
    config: RailsConfig,
    http_client: HTTPClient,
) -> RailOutcome:
    response = await http_call(
        http_client,
        "POST",
        config.endpoint,
        headers={"Authorization": f"Bearer {config.get_api_key()}"},
        json={"text": text},
    )
    data = response.json()
    return RailOutcome.block(**data) if data["blocked"] else RailOutcome.allow(**data)
```

Tests inject a fake `HTTPClient` and assert requests without patching
`aiohttp`, `httpx`, or the network.

### Client injection

Both engines register a shared HTTP client through the existing action
parameter injection path. The minimum contract is one runtime-owned client
that accepts absolute URLs and pools connections per origin.

Service-scoped clients can be added when a rail needs a fixed base URL, auth
policy, retry policy, or transport settings:

```python
client = http_clients.get("trend_micro")
```

The registry is generic. It must not contain a hard-coded branch for every
rail. A rail config or future manifest contribution can describe how its
service client is configured without teaching core code the rail's behavior.

### Relation to existing clients

`nemoguardrails.llm.clients.base.BaseClient` already contains reusable pieces:

- `httpx.AsyncClient` ownership.
- Connection pooling.
- Timeout handling.
- Retry policy and `Retry-After` support.
- Status and JSON validation.
- Streaming SSE decoding.

Those pieces are currently coupled to LLM exceptions and LLM response
validation. The transport-neutral parts should move behind a canonical HTTP
client, for example:

```text
nemoguardrails/http/
  __init__.py
  types.py
  client.py
  retry.py
  instrumentation.py
```

`OpenAICompatibleClient` then composes the general transport and adds
OpenAI-specific payload construction, SSE interpretation, and LLM error
translation.

`APIEngine` can become a small configuration adapter around `HTTPClient` or be
removed once all callers use the registry directly. It should not remain a
second implementation of transport, retry, lifecycle, and instrumentation.

## Provider SDKs

Some rails use provider SDKs rather than raw REST. For example, a generated
cloud client may own authentication, retries, protobuf conversion, and gRPC or
HTTP transport internally.

There are two supported patterns:

1. Raw REST integrations use `HTTPClient`.
2. Provider SDK integrations expose a small rail-owned service protocol and
   use provider-supported telemetry or a generic external-operation wrapper.

Example service protocol:

```python
class TextModerationService(Protocol):
    async def moderate(self, text: str) -> ModerationResult: ...
```

The action depends on `TextModerationService`; an SDK adapter implements it.
The adapter may use an `external_call()` context to create a stable service
span if the SDK has no instrumentation.

We should not wrap an SDK object in `HTTPClient` when callers cannot observe or
control its HTTP request. Doing so invents transport data and couples the
action to undocumented SDK behavior.

## Privacy and Security

### Default redaction

Instrumentation never records these values by default:

- Authorization and proxy-authorization headers.
- Cookies and set-cookie headers.
- API-key headers.
- URL user information.
- Query parameter values.
- Request and response bodies.
- Prompts, completions, or retrieved chunks.

The URL recorded on spans excludes query values and fragments. Stable route
templates are preferred over raw high-cardinality paths when available.

### Content capture

Model prompt and response capture remains separately configurable from HTTP
body capture. Enabling model content does not implicitly enable raw transport
body capture, which could duplicate content and expose provider envelopes or
credentials.

HTTP content capture should require an explicit policy with:

- Allowed content types.
- Maximum captured size.
- Header allowlist.
- Body redactor.
- Service allowlist.
- Separate request and response controls.

Rail privacy declarations can inform this policy, but the transport wrapper
receives a resolved capture decision rather than importing or interpreting
manifests itself.

### Untrusted endpoints

A shared client does not make arbitrary URLs safe. Existing configuration and
deployment controls remain responsible for allowed endpoints. Logs and spans
must not record credentials embedded in a URL. Redirect behavior should not
forward sensitive headers to a different origin.

## Errors and Retries

Instrumentation must preserve the original exception type and traceback. It
records an error and re-raises without translating it.

Translation belongs at semantic adapters:

```text
HTTP transport error
  -> OpenAICompatibleClient adds provider context
  -> OpenAIChatModel exposes LLMClientError

HTTP transport error
  -> ordinary API action handles HTTP error directly
```

Retry policy belongs in `RetryingHTTPClient`, not in actions. Per-call policy
overrides should be explicit and bounded. Streaming operations must not retry
after yielding user-visible data unless the protocol provides a safe resume
mechanism.

## Span Ownership and Parenting

The runtime establishes request, rail, and action spans. Canonical client
wrappers create child spans using the active context:

```text
guardrails request
  -> input rail
    -> rail action
      -> GenAI model call
        -> optional HTTP transport

guardrails request
  -> output rail
    -> rail action
      -> external HTTP API call
```

The model or HTTP wrapper does not search for a specific engine span or mutate
the parent. Normal OTEL context propagation determines parenting.

Instrumentation helpers must be no-ops when OTEL is unavailable or disabled.
Importing a rail, manifest, model adapter, or HTTP protocol must not require the
OTEL SDK.

## Avoiding Duplicate Instrumentation

Duplicate spans can arise from:

- Wrapping an already instrumented model twice.
- Guardrails HTTP instrumentation plus global `httpx` instrumentation.
- Guardrails SDK instrumentation plus provider auto-instrumentation.
- Instrumentation in both `llm_call()` and `LLMModel`.

The design uses these rules:

- GenAI semantic instrumentation exists at `LLMModel`, not `llm_call()` and
  not `OpenAIChatModel`.
- Model wrapping is idempotent.
- HTTP transport instrumentation is independently configurable.
- The runtime records which wrappers it owns.
- Provider auto-instrumentation is documented as a separate lower layer.
- One span name and attribute set describe one operation layer.

## Proposed Runtime Shape

```text
Runtime clients
  models: dict[str, LLMModel]
    main -> InstrumentedLLMModel(OpenAIChatModel(...))
    content_safety -> InstrumentedLLMModel(OpenAIChatModel(...))
    topic_control -> InstrumentedLLMModel(OpenAIChatModel(...))

  http_client: HTTPClient
    InstrumentedHTTPClient(
      RetryingHTTPClient(
        HttpxHTTPClient(...)
      )
    )

  services: dict[str, object]
    provider SDK adapters where raw HTTP is not appropriate
```

LLMRails and IORails construct the same runtime clients. Their orchestration
differs, but model and transport behavior does not.

## Migration

### Model path

1. Extract GenAI call instrumentation from IORails `EngineRegistry` into an
   `InstrumentedLLMModel` decorator without changing attributes or metrics.
2. Add contract tests proving the decorator preserves model responses,
   exceptions, streaming chunks, cancellation, and provider properties.
3. Create canonical models from existing `Model` configuration through the
   shared model factory.
4. Change the IORails registry from `ModelEngine` values to `LLMModel` values.
5. Convert IORails input messages to `ChatMessage` at the boundary.
6. Route main generation and rail model calls through the same registry.
7. Remove `ModelEngine` after equivalence tests show matching requests,
   responses, streaming behavior, retry behavior, and telemetry.
8. Allow caller-provided `LLMModel` instances once ownership and wrapping are
   explicit.

### HTTP path

1. Introduce neutral HTTP types, errors, and the `HTTPClient` protocol.
2. Extract the transport-neutral parts of the existing LLM `BaseClient` into a
   reusable `HttpxHTTPClient` and retry decorator.
3. Add `InstrumentedHTTPClient` with disabled-by-default content capture.
4. Register one runtime-owned HTTP client in both engines and inject it into
   actions.
5. Convert `APIEngine` to the canonical client or remove it.
6. Migrate representative rails from both `aiohttp` and `httpx` to validate the
   abstraction.
7. Migrate remaining raw REST rails gradually.
8. Keep provider SDK rails behind service adapters.
9. Remove redundant per-action session creation and transport-specific test
   mocks.

Migration is behavior-preserving. A rail may continue using its current client
until its action and tests are moved. Core code should not add a compatibility
branch for each rail.

## Validation

### Model decorator

- Non-streaming response passes through unchanged.
- Streaming chunks pass through in the same order.
- Stream span covers first iteration through exhaustion.
- Early cancellation closes the wrapped generator and span.
- Provider exceptions remain the same exception type.
- Provider and model properties delegate unchanged.
- Request parameters, response attributes, and usage are recorded correctly.
- Token metrics are absent when usage is absent.
- Time-to-first-chunk ignores role-only and usage-only frames.
- Content capture is off by default and redacted when enabled.
- Disabled telemetry produces no spans or metrics.
- Wrapping is idempotent.

### HTTP client

- Method, URL, headers, query, JSON, and content are forwarded unchanged.
- Response status, headers, bytes, text, and JSON parsing are stable.
- Timeout, connection, status, and decode errors are distinct.
- One logical span covers all retry attempts.
- Retry count is recorded without exposing bodies or credentials.
- Authorization, cookies, API keys, and query values never appear in spans.
- Body capture requires explicit opt-in and enforces size limits.
- Disabled telemetry is a no-op.
- Runtime-owned clients close exactly once.
- Caller-owned clients are never closed by the runtime.
- Fake clients require no concrete HTTP dependency.

### Engine equivalence

- LLMRails and IORails send equivalent canonical model requests.
- Non-streaming responses and structured metadata match.
- Streaming content, reasoning, tool calls, usage, and finish reasons match.
- Existing IORails GenAI span and metric expectations remain satisfied.
- Rail actions using `llm_call()` receive the same instrumentation as main
  generation.
- Rail actions using `http_call()` produce correctly parented HTTP client spans.

## Alternatives

### Put OTEL directly in `OpenAIChatModel`

This covers only one implementation, couples the adapter to optional telemetry,
and leaves custom or framework-backed models inconsistent.

### Instrument only `llm_call()`

Direct engine calls to `LLMModel` would not be observed, and engines would
continue maintaining their own instrumentation.

### Instrument only the HTTP transport for model calls

HTTP spans cannot reliably describe token usage, finish reasons, tool calls,
model identity, or semantic streaming milestones.

### Globally instrument `httpx` and `aiohttp`

Global instrumentation is application policy, can create duplicates, does not
cover SDK semantics, and gives Guardrails insufficient control over capture and
redaction.

### Keep `ModelEngine` and `APIEngine` as IORails-only clients

This preserves the current duplication and prevents actions, LLMRails, and
caller-provided models from sharing the same behavior and observability.

### Require every provider SDK to implement `HTTPClient`

SDKs can use gRPC, internal retries, generated request types, and hidden
transports. Pretending they are raw HTTP loses information and creates a false
abstraction.

## Result

The target model is straightforward:

```text
model semantics -> LLMModel -> optional GenAI instrumentation
REST semantics -> HTTPClient -> optional HTTP instrumentation
SDK semantics -> service adapter -> provider or external-call instrumentation
```

Engines orchestrate. Actions express rail behavior. Canonical clients own
outbound execution. Decorators own observability. The same rail action then
behaves and emits telemetry consistently under LLMRails, IORails, tests, and
future execution engines.
