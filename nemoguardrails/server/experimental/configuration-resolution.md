# Guardrails configuration resolution

The private experimental proxy supports deployment-defined Guardrails policy
resolution without allowing callers to select an arbitrary configuration.
Applications declare logical config IDs during assembly and may inject a
resolver that selects one of those IDs for each guarded request.

This mechanism does not add native server configuration discovery. Config
locations and resolution policy belong to the embedding application.

## Config entries and resolution

`ConfigEntry` maps a deployment-owned logical ID to a configuration directory:

```python
from pathlib import Path

from nemoguardrails.server.experimental.rails import ConfigEntry

configs = (
    ConfigEntry("standard", Path("/configs/standard")),
    ConfigEntry("restricted", Path("/configs/restricted")),
)
```

The resolver returns a declared ID, never a path:

```python
from nemoguardrails.server.experimental.rails import ConfigContext


def resolve_config(context: ConfigContext) -> str:
    subject = context.request.state.guardrails_subject
    if "restricted-users@example.com" in subject.distribution_lists:
        return "restricted"
    return "standard"
```

The catalog and resolver are supplied when the private application is
assembled:

```python
from nemoguardrails.server.experimental._application import create_proxy_application

application = create_proxy_application(
    providers=providers,
    rails_configs=configs,
    config_resolver=resolve_config,
)
```

The runtime verifies that every selected ID belongs to the catalog. Resolver
failures, empty IDs, and unknown IDs fail closed with the provider-native
`config_resolution_failed` response before an upstream request is made. A
single declared config gets a static resolver automatically.

Resolution failures emit a warning containing the server-supplied provider and
operation identity and the exception type. Exception messages, tracebacks, and
request data are omitted from that warning.

## Resolution context

The resolver receives a `ConfigContext` containing:

| Value | Meaning |
| --- | --- |
| `request` | The Starlette request for the guarded provider call. |
| `provider_type` | The provider protocol family; instances may select different reviewed dialects. |
| `provider_instance_id` | The deployment-local provider instance. |
| `operation_name` | The guarded operation, such as `chat_completions.create`. |

The provider and operation values are supplied by the server. Deployments do
not need to reconstruct them from paths. The request provides headers,
cookies, authentication state, and application middleware state, but raw
caller values are not trusted merely because they are available to the
resolver.

The proxy reads and validates the guarded request body within its configured
size limit before invoking the resolver. Asynchronous resolvers can use
`await context.request.body()` or `await context.request.json()` to read that
buffered body without consuming the connection again. Reading the body does
not change the original bytes forwarded upstream. Configuration resolution
runs once per guarded request; the selected checker handles both input and
output inspection, including streaming responses.

Middleware can attach normalized deployment-owned policy facts before a
guarded route runs:

```python
@application.middleware("http")
async def add_policy_subject(request, call_next):
    identity = await authenticate_request(request)
    request.state.guardrails_subject = PolicySubject(
        tenant_id=identity.tenant_id,
        application_id=identity.application_id,
        distribution_lists=frozenset(identity.distribution_lists),
    )
    return await call_next(request)
```

The resolver may be synchronous or asynchronous, and its result must be one of
the IDs declared at application construction. Synchronous resolvers run on the
application event loop and must perform only short, non-blocking selection from
trusted in-memory state. Authentication middleware can prepare those facts
before the resolver runs. A resolver that consults an authorization service,
tenant registry, or session store must use asynchronous I/O with cancellation
support and appropriate client timeouts.

For example, an asynchronous resolver can look up the policy assigned to an
authenticated tenant:

```python
async def resolve_config_async(context: ConfigContext) -> str:
    subject = context.request.state.guardrails_subject
    registry = context.request.app.state.policy_registry
    return await registry.config_id_for(tenant_id=subject.tenant_id)


application = create_proxy_application(
    providers=providers,
    rails_configs=configs,
    config_resolver=resolve_config_async,
    config_resolution_timeout=2.0,
)
application.state.policy_registry = policy_registry
```

Here, `policy_registry` is an application-supplied asynchronous client, whose
startup, shutdown, and I/O timeouts are owned by the embedding application.
Its `config_id_for()` method returns a declared ID such as `"standard"` or
`"restricted"`. Attach the authentication middleware shown above to this
application so `guardrails_subject` comes from authenticated identity. The proxy
awaits the resolver and checks its result against `configs`; pass the function
itself as `config_resolver`, without calling it during assembly. Exceptions and
timeouts fail closed rather than selecting a fallback policy.

`create_proxy_application(config_resolution_timeout=5.0, ...)` sets the resolver's
cooperative deadline in seconds, independently of outbound provider timeouts.
The default is five seconds; an override must be finite and positive. A timeout
fails closed through the provider-native `config_resolution_failed` response,
without forwarding the request or selecting a fallback policy. Request
cancellation propagates instead of becoming a configuration error.

The deadline cannot preempt blocking synchronous work or force a resolver that
suppresses cancellation to stop. Preloading removes request-time engine
initialization; it does not make external policy selection free of latency.

## Engine selection

The proxy follows the library's `NEMO_GUARDRAILS_IORAILS_ENGINE` environment
switch. Set it before the process imports `nemoguardrails`:

```bash
export NEMO_GUARDRAILS_IORAILS_ENGINE=1
```

With `1`, `true`, or `yes` (case-insensitive), compatible configurations use
IORails. Other configurations fall back to LLMRails with the library's warning.
With the variable unset or set to `0` or `false`, configurations use LLMRails.
Restart the process to change the selection. The proxy uses the `Guardrails`
facade for startup, checks, and shutdown with either engine.

The catalog and request resolver work the same way with either engine. The
proxy supports Colang 1.0 input/output checks. Colang 2.x configurations prevent
readiness because the LLMRails check API requires logging options that its
Colang 2.x runtime does not support. Retrieval and tool rails also prevent
readiness. IORails-specific features, including its
admission queue and OpenTelemetry metrics, are available only for configurations
that actually run on IORails.

## Lifecycle

Config IDs and directory paths are validated and fixed during application
assembly. Configuration discovery and parsing use `RailsConfig.from_path()`
during startup, including its nested YAML layouts and `.railsignore` handling.
There is no requirement for a file named `config.yml` or `config.yaml`. Invalid
configuration contents, including an empty directory, prevent readiness.

Guardrails cores for **every declared configuration** are initialized sequentially before
the application becomes ready, including configurations the resolver might
never select. Different IDs receive independent cores, even when their paths
are identical. Requests perform a synchronous lookup of an already-started
checker; they do not parse files, construct engines, or share startup tasks.

If parsing, engine construction, or startup fails, the application does not
accept traffic. Cleanup is attempted for every acquired engine, including the
one whose startup failed, and for the shared HTTP client. Cancellation during
startup follows the same rollback path. There is no partially ready catalog or
request-time retry of a failed configuration. Incomplete engine cleanup stays
owned by the application without publishing a ready runtime. A subsequent
application lifespan retries that cleanup and constructs a fresh catalog only
after it succeeds; otherwise startup is rejected again.

Preloading validates each adapter's inspection policy before starting its engine.
Unsupported retrieval or tool rails, including per-tool declarations, prevent
readiness even when the resolver would never select that configuration. Any
previously started engines and the rejected engine remain owned for cleanup.

Preloading trades longer startup and resources for potentially unused configs
for fail-fast initialization and predictable request latency. Resolver errors
and operation-specific capability checks remain request-time concerns; a valid
inspection policy does not prove compatibility with every endpoint or streaming
mode. Request-time validation remains in place as a defensive check.

Configuration file loading and parsing run in a worker thread, so filesystem
work does not block the event loop. Engine construction and startup remain on
the application loop, before serving; arbitrary engine constructors are not
assumed to be thread-safe or non-blocking.

The application lifespan starts, publishes, and closes `ServerRuntime`. The
runtime owns checker initialization and exposes whether engine cleanup remains
pending, so the lifespan does not inspect repository internals. Publication
still waits for the entire catalog; a pending runtime stays owned by the
application until a subsequent cleanup attempt completes.

The runtime closes the checker repository and the shared outbound HTTP client.
Shutdown rejects new checker lookups and independently attempts
every loaded checker's cleanup. Each checker gets a
shielded five-second cooperative budget; timeouts and ordinary failures are
logged without replacing the application's primary error. Incomplete checker
cleanup remains retryable. HTTP client cleanup still runs and has its own
shielded five-second budget.

These budgets cannot forcibly stop blocking or cancellation-suppressing engine
code. Cancelling file parsing also cannot forcibly stop the worker thread;
the cancelled load will not proceed to construct an engine. A logged timeout
does not establish that all resources were released.

The fixed catalog is an explicit allow-list and resource-ownership boundary.
It does not define hot reload, cache invalidation, concurrent replacement, or a
configuration CRUD control plane. Adding those capabilities later requires
separate decisions about authentication, persistence, versioning, atomicity,
rollback, auditing, and in-flight requests.

## Inspection failures

Both engines populate the `RailsResult.failed` flag. It
distinguishes a rail execution failure from a policy block while keeping both
results `BLOCKED` for existing callers. The proxy maps reported failures to
`ContentCheckFailed` and returns a generic HTTP 502 inspection error. A policy
block remains HTTP 400. If an output stream has already started, the proxy
emits the corresponding error event and terminates the stream; it cannot
change the HTTP status already sent.

LLMRails reads action completion events and typed rail outcomes for the current
check, including parallel rails. The adapter does not infer failures from
`content` or `reason` text. Exceptions escaping a check continue to map to
`ContentCheckFailed`.

## Trust boundary

- Do not resolve filesystem paths from request headers, query parameters, or
  provider payload fields.
- Do not treat `x-nemo-guardrails-config` or a similar caller header as trusted
  policy selection. Such a header remains ordinary provider traffic unless
  trusted infrastructure authenticates and replaces it before resolution.
- Prefer authenticated `request.user`, `request.auth`, or deliberately written
  `request.state` values over raw caller metadata.
- Avoid storing or logging secrets that policy selection does not require.
- Keep model routing and Rails policy resolution independent: one selects the
  provider destination, while the other selects a declared Guardrails policy.
