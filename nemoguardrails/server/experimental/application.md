# Python proxy application

Declare provider instances and select their reviewed operation bindings through
one application factory. Typed Python projections, classifiers, and handwritten
hooks remain the source of policy. Application configuration selects those
bindings; it does not compile or load guard contracts.

This Python interface requires the `server` extra. It does
not add YAML loading, a CLI launcher, or a stable configuration-file format.
The application owns outbound HTTP and preloads its Rails catalogue during
lifespan startup. Configuration directories are trusted: their actions may
execute Python. Application construction validates paths but does not start
engines or contact providers.

## Declare a built-in provider

```python
from pathlib import Path

from nemoguardrails.server.experimental.application import (
    ConfigEntry,
    ProviderConfig,
    ProxyConfig,
    create_proxy_application,
)

config = ProxyConfig(
    rails_configs=(
        ConfigEntry(id="standard", config_path=Path("./rails/standard")),
    ),
    providers=(
        ProviderConfig(
            provider="openai",
            instance_id="gateway",
            base_url="https://gateway.example/v1",
            mount_path="/gateway",
            operations=("chat_completions", "responses"),
            rails_config_id="standard",
        ),
    ),
)

app = create_proxy_application(config)
```

The Rails directory must exist and contain a supported configuration. Relative
paths resolve against the working directory at application construction; use
absolute paths when embedding in another application.

This forwards provider generation transparently through HTTPX. It does not use
the Guardrails LLM framework or LangChain to generate the provider response.
Rails configuration separately determines any models, optional integrations,
credentials, or services required by its checks. Upstream authorization is
forwarded from the caller under the existing transport rules; the declaration
does not store credentials or configure inbound authentication. Establish that
authentication separately when deploying beyond a trusted local environment.

### Operations and dialects

| Selection | Configuration |
| --- | --- |
| Native Chat only | `operations=("chat_completions",)`; omit `dialect`. |
| Native Chat and Responses | `operations=("chat_completions", "responses")`; omit `dialect`. |
| Reviewed LiteLLM Chat and Responses | Same operations, with `dialect="litellm"`. |

Responses-only selection is also supported. Both operations have buffered and
streaming bindings. Order is preserved; empty, duplicate, or unknown operations
are rejected. There is no default that automatically adds future operations.

Omitted dialect means the chosen provider's native reviewed bindings, not a
global OpenAI default. Initially the built-in implementation is `openai`, with
native and explicit LiteLLM bindings. Provider, instance, upstream, and dialect
are distinct: changing the URL does not select a dialect. Unsupported names
fail rather than falling back. The LiteLLM selection describes reviewed wire
extensions, not compatibility with every LiteLLM backend.

Native bindings still enforce the supported single-text policy; they do not
accept every feature in the upstream API. See the
[LiteLLM Chat boundary](providers/openai/dialects/litellm/README.md) and
[LiteLLM Responses boundary](providers/openai/dialects/litellm/responses.md).

**Operation selection is not a route allowlist.** Unselected routes retain
passthrough behavior. Selected bindings also do not activate Rails checks
disabled in the chosen Rails configuration. Inspect input/output configuration
when deciding which traffic is actually checked.

## Fixed and dynamic Rails configuration

Define the catalogue once, even when it contains dozens of entries.
`rails_config_id` selects one fixed entry per provider instance. No repeated
lists of allowed IDs are required:

```python
config = ProxyConfig(
    rails_configs=(
        ConfigEntry(id="standard", config_path=Path("./rails/standard")),
        ConfigEntry(id="strict", config_path=Path("./rails/strict")),
    ),
    providers=(
        ProviderConfig(
            provider="openai",
            instance_id="public",
            mount_path="/public",
            base_url="https://gateway.example/v1",
            operations=("chat_completions", "responses"),
            rails_config_id="strict",
        ),
        ProviderConfig(
            provider="openai",
            instance_id="customers",
            mount_path="/customers",
            base_url="https://customers.example/v1",
            operations=("chat_completions",),
        ),
    ),
)

def resolve_config(context):
    tenant = context.request.state.authenticated_tenant
    return tenant.guardrails_config_id

app = create_proxy_application(config, config_resolver=resolve_config)
```

The example requires trusted middleware to establish `authenticated_tenant`.
It must not be copied directly from an untrusted request header.

- `public` always uses `strict` and never invokes the user callback.
- `customers` has no fixed ID, so the callback selects a registered entry.
- The callback receives the original request, provider type, provider instance
  ID, and canonical operation name, such as `chat_completions.create`.
- Synchronous and asynchronous callbacks are supported. A dynamic instance
  requires a callback even with a one-entry catalogue.
- All-fixed applications need no callback, even with many catalogue entries.
- Unknown fixed IDs fail construction. Unknown/invalid callback results and
  callback failures fail closed before upstream dispatch, without fallback.
- Selection chooses one configuration; it does not run multiple configurations
  in sequence. Dynamic callbacks own authorization; registration in the shared
  catalogue alone does not authorize a caller to select that configuration.

There are no per-provider allowlist fields or declarative per-operation
overrides in this interface. A trusted dynamic resolver can use operation
context as needed. All catalogue entries are still preloaded under the existing
repository behavior, including entries not selected by a particular request.

## Custom Python composition

The same factory still accepts explicit integrations, ordinary resolver
callbacks, and an application-owned HTTP transport factory:

```python
from nemoguardrails.server.experimental.providers.openai.integration import (
    create_openai_chat_integration,
)

app = create_proxy_application(
    providers=(
        create_openai_chat_integration(
            instance_id="gateway",
            base_url="https://gateway.example/v1",
        ),
    ),
    rails_configs=(ConfigEntry(id="standard", config_path=Path("./rails/standard")),),
)
```

Direct `ProviderIntegration` objects allow custom endpoints, mixed dialects,
or custom implementations without registration. The existing direct mode still
infers static selection with one catalogue entry and no resolver, requires a
resolver with multiple entries, and invokes an explicit resolver even with one
entry. Those semantics are unchanged.

Do not mix `ProxyConfig` with direct providers, Rails entries, or limit/prefix
overrides. Put data settings in `ProxyConfig`; `config_resolver`,
`http_transport_factory`, and `local_routers` remain separate Python arguments in
either mode. Local routers register before provider catch-alls; choose local
paths outside guarded operation paths. YAML does not load Python routers.
See [HTTP transport](http-transport.md) for local routes and transport ownership.

## Validation, defaults, and limitations

Unknown fields are rejected, including inside catalogue entries. Configuration
models snapshot input collections and are frozen. Provider IDs and mounts must
be unique, and existing route validation rejects conflicting bindings. The
upstream URL must be HTTP(S), without userinfo, query, or fragment. Declarative
upstreams are explicit; provider-specific environment defaults are not used.

The default application prefix is empty, the request limit is 1 MiB, the
response limit is 10 MiB, and the outbound timeout is 30 seconds. The limits
retain existing guarded-versus-passthrough behavior; they are not a newly added
global response-buffering rule.

The new selection path reuses existing lifecycle owners. It does not fix or
claim complete safety for interrupted startup/shutdown or failed HTTP cleanup.
Those owners require their own real-resource validation. Successful in-process
buffered/SSE tests are not proof of socket-disconnect or shutdown safety.
