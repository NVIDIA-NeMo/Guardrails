# Outbound HTTP transport extension

The private staged application assembly accepts an optional HTTPX transport
factory. This is a low-level extension point for changing how provider requests
are dispatched without adding routing, networking, or deployment policy to the
Guardrails pipeline.

```python
import httpx

from nemoguardrails.server.experimental._application import create_proxy_application

application = create_proxy_application(
    providers=providers,
    rails_configs=rails_configs,
    config_resolver=config_resolver,
    http_transport_factory=lambda: httpx.MockTransport(handler),
)
```

The leading underscores are intentional while this stack is staged. This work
restores the exploratory capability without committing to a public provider
extension API. Trusted Rails configuration resolution selects from the catalog
preloaded before application readiness; it does not change transport ownership
or routing semantics.

When no factory is supplied, HTTPX uses its normal asynchronous network
transport. When one is supplied, the application invokes it once per lifespan.
Every provider instance uses the resulting shared client. The runtime closes
the client and transport during shutdown. If startup fails after transport
creation but before runtime ownership succeeds, the lifespan closes the
transport while propagating the error.

If Rails preloading fails after runtime construction, the runtime closes both
the already-started checkers and this shared HTTP client before startup fails.

HTTP client shutdown and startup rollback use a shielded five-second
cooperative cleanup budget. Timeouts and ordinary close failures are logged
without masking the original failure. Blocking or cancellation-suppressing
custom transports cannot be forcibly stopped by this budget.

## Position in the request flow

| Component | Responsibility |
| --- | --- |
| `ProviderIntegration` | Declares guarded endpoints, projections, renderers, an upstream, and provider-specific proxy control headers. |
| `UpstreamSettings` | Defines one provider instance's local mount and initial base URL. |
| `ProviderTransport` | Builds provider requests, applies guarded representation and response-buffer rules, and relays buffered or streaming responses. |
| Pass-through router | Adapts unbuffered HTTP exchanges to ASGI responses and owns cleanup through disconnect or completion. |
| HTTPX transport | Performs or replaces the final HTTPX dispatch operation. |

The provider transport strips hop-by-hop headers and the integration's declared
`control_headers` before requests reach the upstream. Provider authentication
and API-version headers are not control headers and continue upstream normally.

The shared client pools connections, not caller sessions. It does not persist
upstream cookies, and outbound requests do not inherit client-level cookies,
default headers, or authentication. Explicit caller `Cookie` and upstream
`Set-Cookie` headers still pass through normally. Trusted credential injection
belongs in the application transport, not in shared client defaults. Expected
network failures before response handoff use the provider-native error mapping;
cancellation and programming errors are not rewritten as network failures.

`provider_type` identifies a protocol family. Each provider instance explicitly
selects its own reviewed endpoints, so strict OpenAI and the LiteLLM dialect can
coexist on different mounts. `create_openai_chat_integration(endpoint=...)`
selects a dialect without changing other instances. Buffered-only endpoints do
not receive streaming bindings and reject streaming requests.

The configured request and response body limits apply to guarded traffic whose
JSON or SSE representation must be inspected. Pass-through request and response
bodies for unrelated provider operations stream without those guarded limits.
Only guarded operations with output inspection request identity encoding;
pass-through traffic and uninspected guarded output preserve the caller's
`Accept-Encoding` value.

For guarded requests, the extension point is reached after input inspection
and before output inspection:

```text
caller
  -> guarded endpoint
  -> input inspection
  -> ProviderTransport
  -> HTTPX transport
  -> selected upstream
  -> HTTPX transport
  -> output inspection
  -> caller
```

It is not FastAPI middleware. FastAPI middleware surrounds requests entering
the ASGI application and responses leaving it; an HTTPX transport surrounds
the nested provider exchange initiated by the application.

## Relationship to `UpstreamSettings`

`UpstreamSettings.base_url` is fixed for one provider instance during the
application lifetime. It supplies the initial URL used for each outbound
request. The HTTPX transport receives that complete request and may send it
unchanged, route it to a trusted configured origin, or handle it without
network I/O.

A routing transport can therefore use a virtual base URL:

```python
from examples.scripts.model_routing_transport import create_model_routing_transport
from nemoguardrails.server.experimental._application import create_proxy_application
from nemoguardrails.server.experimental.providers.openai.integration import (
    create_openai_chat_integration,
)

openai = create_openai_chat_integration(
    base_url="http://model-router.internal/v1",
)

application = create_proxy_application(
    providers=(openai,),
    rails_configs=rails_configs,
    config_resolver=config_resolver,
    http_transport_factory=create_model_routing_transport,
)
```

`model-router.internal` is a logical address. The routing transport sees the
fully formed request, chooses a destination from application-owned tables, and
delegates to a network transport. A dedicated logical hostname makes the
boundary explicit and prevents accidental direct provider access if the
routing transport is removed.

If routing runs as a separate MaaS or gateway service, its reachable URL is the
fixed provider base URL and no custom transport is necessary:

```python
openai = create_openai_chat_integration(
    base_url="http://model-routing-gateway:9000/v1",
)
```

The external gateway can then own model selection, aliases, credentials,
retries, and provider destinations independently of Guardrails.

## Model-routing example

The tested
[`model_routing_transport.py`](../../../examples/scripts/model_routing_transport.py)
example declares canonical model routes and public aliases. A Chat Completions
request containing `"model": "model-1"` is inspected, then routed to the
configured `model-2` destination with its model changed to `model-2`. The
upstream response returns to the same guarded handler and passes through output
inspection. Model-less requests use the explicit default route.

The example removes the caller's authorization value and loads the credential
named by the selected route, preventing credentials from crossing provider
origins. Destination URLs come only from validated application-owned
`ModelRoute` values. Payload fields and request headers cannot select arbitrary
URLs. Route URLs must be absolute HTTP or HTTPS URLs without embedded
credentials, query, or fragment.

The example exposes `create_model_routed_application(rails_configs,
config_resolver)` so the embedding application owns both routing and trusted
Rails policy selection without coupling the two mechanisms.

## Other uses

The transport extension can support:

- deterministic provider tests with `httpx.MockTransport`;
- in-process ASGI composition with `httpx.ASGITransport` where its buffering is acceptable;
- model aliases, deployment-name translation, and model-to-origin routing;
- client-side load balancing, failover, or service discovery;
- controlled egress through a proxy, service mesh, or Unix domain socket;
- custom TLS and mutual-TLS configuration;
- traffic recording, replay, latency simulation, and fault injection;
- outbound request tracing and metrics when HTTPX event hooks are insufficient.

Use `UpstreamSettings` when only one provider instance's fixed base URL
changes. Use FastAPI middleware for inbound concerns such as request IDs and
authentication. Use provider contracts and projections for guardable payload
semantics.

## Constraints and ownership

- The transport is application-wide and sees requests for every configured provider.
- A multi-provider transport must dispatch using logical hosts, paths, or other server-controlled metadata.
- Provider catch-all routes stream request and response bodies without applying guarded-route buffer limits.
- The factory must return a new transport for each lifespan; the application runtime closes it.
- A production transport must preserve HTTPX streaming, cancellation, and response-close behavior.
- Dispatch owns each response until handoff; the returned response then owns cleanup, including failure before iteration. Cooperative cleanup attempts are shielded and bounded, and failures are logged without replacing the primary outcome. Closing must tolerate retries; a timeout does not establish that resources were freed.
- Response cleanup budgets are separate from transport read timeouts and application shutdown.
- `httpx.ASGITransport` buffers ASGI responses and is not the production choice for guarded streaming.
- Dynamic destinations must come from trusted configuration; arbitrary client-provided URLs would introduce SSRF risk.
- Cross-origin routing must apply the credential belonging to the selected origin.
- The transport must not apply Guardrails or interpret provider contracts.
