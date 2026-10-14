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

The resolver may be synchronous or asynchronous. It may consult trusted
application state, an authorization service, a tenant registry, a session
store, or rollout policy, but its result must still be one of the IDs declared
at application construction.

## Lifecycle

Config entries are validated and fixed during application assembly. Guardrails
cores for **every declared configuration** are initialized sequentially before
the application becomes ready, including configurations the resolver might
never select. Different IDs receive independent cores, even when their paths
are identical. Requests perform a synchronous lookup of an already-started
checker; they do not parse files, construct engines, or share startup tasks.

If parsing, engine construction, or startup fails, the application does not
accept traffic. Already-started checkers and the shared HTTP client are closed;
the engine whose startup failed owns rollback of its incomplete initialization.
Cancellation during startup follows the same rollback path. There is no
partially ready catalog or request-time retry of a failed configuration. A new
application lifespan constructs a fresh catalog.

Preloading trades longer startup and resources for potentially unused configs
for fail-fast initialization and predictable request latency. Resolver errors
and operation-specific capability checks remain request-time concerns; loading
a configuration does not prove it supports every endpoint or streaming mode.

Configuration file loading and parsing run in a worker thread, so filesystem
work does not block the event loop. Engine construction and startup remain on
the application loop, before serving; arbitrary engine constructors are not
assumed to be thread-safe or non-blocking.

The application runtime closes the checker repository and the shared outbound
HTTP client. Shutdown rejects new checker lookups and independently attempts
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
