# Proxy application

Declare provider instances and select their reviewed operation bindings through
one application factory. Typed Python projections, classifiers, and handwritten
hooks remain the source of policy. Application configuration selects those
bindings; it does not compile or load guard contracts.

The Python interface and server-owned YAML launcher require the `server` extra.
The manifest's numeric revision identifies its format, not a stability promise.
The application owns outbound HTTP and preloads its Rails catalogue during
lifespan startup. Configuration directories are trusted: their actions may
execute Python. Application construction validates paths but does not start
engines or contact providers.

## Run from YAML

The server owns its launcher; it is not a subcommand of the library CLI.
Its current module location is temporary. A later independent server package
will register its own executable against the same CLI implementation.

From this checkout, install dependencies with `uv sync --locked --extra server`,
then use:

```bash
uv run --locked python -m nemoguardrails.server.experimental validate nemoguardrails/server/experimental/examples/basic/proxy.yaml
uv run --locked python -m nemoguardrails.server.experimental inspect nemoguardrails/server/experimental/examples/basic/proxy.yaml
uv run --locked python -m nemoguardrails.server.experimental serve nemoguardrails/server/experimental/examples/basic/proxy.yaml
```

The [example manifest](examples/basic/proxy.yaml)
selects native OpenAI Chat Completions and Responses bindings against a local
upstream at `http://127.0.0.1:9000/v1`; it does not start an upstream. Both
operations support the reviewed single-text buffered and streaming modes.
Set `base_url` to your deployed provider endpoint and select `dialect: litellm`
only if that provider needs the reviewed dialect. A URL does not imply a dialect.

```yaml
version: 1
rails_configs:
  - id: standard
    config_path: ./rails/standard
providers:
  - provider: openai
    instance_id: gateway
    base_url: https://gateway.example/v1
    operations: [chat_completions, responses]
    rails_config_id: standard
```

Keep the proxy manifest outside the Rails directory: Rails loads YAML files
recursively from its own directory, and the two formats must not be mixed.
The example uses `proxy.yaml` beside `rails/config.yml`.

The YAML fields are the Python configuration fields below, plus required integer
`version: 1`. Every provider must specify a fixed `rails_config_id` in YAML;
there is no implicit selection, callback import or dynamic resolver syntax.
Multiple providers and dozens of shared catalogue entries do not require
repeating the catalogue. Use Python for trusted dynamic resolution.

Relative Rails paths resolve against the manifest's directory, not the shell's
working directory. If the manifest is a symlink, its target directory is used.
No environment or tilde expansion is performed. Unknown fields, duplicate keys,
explicit YAML tags, anchors, aliases, merges, non-string mapping keys and multiple
documents are rejected. Input is UTF-8 and bounded to 1 MiB, depth 32 and 10,000
nodes (including keys). The manifest is deployment-owned; never accept uploads
from callers as configuration.

`validate` checks the declaration, directory/config-file availability, upstream
URL rules and route composition. It does not read Rails contents, import their
actions, start engines, contact providers or prove checker compatibility.
`inspect` returns JSON describing actual bindings and directional profiles,
including passthrough behavior; it does not claim that particular rails are
enabled. Upstream credentials, queries and fragments are rejected, and manifest
diagnostics omit source values/excerpts. Inspection still contains deployment metadata,
including upstream addresses; treat it accordingly.

For editor guidance, emit JSON Schema directly from the manifest model:

```bash
uv run --locked python -m nemoguardrails.server.experimental schema > proxy.schema.json
```

This is the configuration schema, not an exported guard contract. Semantic
checks such as fixed-ID references and routing still require `validate`.
There is no separately maintained schema artifact.

`serve` defaults to `127.0.0.1:8000`; `--host` and `--port` override it.
Access logging and forwarded-proxy-header trust are disabled. Provide your own
inbound access controls before listening beyond loopback. Caller authorization
headers still follow the existing upstream forwarding rules; this launcher
does not store provider keys or introduce service authentication.

### Requests and blocking

The [example Rails configuration](examples/basic/rails/config.yml) blocks common
ASCII email-address patterns in guarded input and output text using local regex
checks. It needs no rail model, additional package or rail-service API key;
generation still uses the configured upstream. This is a small, editable policy,
not comprehensive PII detection or semantic content moderation. It can miss
obfuscated or internationalized addresses and can block harmless example addresses.
Only text selected by the operation binding is checked, not every field in the
provider payload.

Output streaming explicitly uses `chunk_size: 200`, `context_size: 50` and
`stream_first: false`. The sizes match the Rails configuration defaults, but in
this proxy they count **nonempty provider text deltas**, not tokenizer tokens or
SSE events. Each window is checked before its new text is released; the last
partial window is checked at completion. Short responses may therefore be held
until completion. Smaller windows reduce buffering delay but increase check
frequency; these example values are not a deployment-specific tuning recommendation.
The proxy requires `stream_first: false` when output checking is enabled.

Context lets a later check detect patterns spanning windows, but cannot retract
an already-released prefix. Finite context can also miss sufficiently long or
fragmented patterns. Do not treat streaming regex checks as a guarantee against
partial disclosure. Use buffered responses when the entire answer must be
checked before any of its text is returned; the regex's detection limits still apply.

With a compatible upstream running, replace `YOUR_DEPLOYED_MODEL` with its model
ID. Add an `Authorization` header from a locally managed environment variable if
the upstream requires one; never put credentials in the manifest.

```bash
curl -sS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"YOUR_DEPLOYED_MODEL","messages":[{"role":"user","content":"hello"}]}'

curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"YOUR_DEPLOYED_MODEL","messages":[{"role":"user","content":"hello"}],"stream":true}'

curl -sS http://127.0.0.1:8000/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{"model":"YOUR_DEPLOYED_MODEL","input":"hello"}'

curl -sS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"YOUR_DEPLOYED_MODEL","messages":[{"role":"user","content":"Contact person@example.com"}]}'
```

The last request returns `content_blocked` before upstream dispatch. To check
output blocking deterministically, use a synthetic upstream that returns
`person@example.com`: buffered output is rejected; a streamed window containing
the address is withheld and an error event is returned after streaming headers
have been sent. Earlier allowed windows may already have been released. An LLM
prompt asking it to emit an address is not a deterministic regression test.

Invalid guarded content does not fall back to a weaker binding. Unselected
operations may still pass through, and a disabled input/output check does not
become enabled merely by selecting an operation.

Profile/variant selection is not exposed. Inspection reads current profiles
from the bindings. Future variants would select shipped, tested combinations
per operation at startup; Rails configuration alone cannot enable multimodal
extraction or inspection.

The launcher reuses the existing lifespan, including catalogue preloading.
Known incomplete-cleanup/retry prerequisites in the core and shared HTTP owners
remain separate. Successful launch or disconnect smoke tests do not establish
that interrupted cleanup is complete; run those real-owner regressions before
claiming lifecycle readiness.

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
overrides. Put data settings in `ProxyConfig`; `config_resolver` and
`http_transport_factory` remain separate Python callback arguments in either
mode. See [HTTP transport](http-transport.md) for transport ownership.

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
