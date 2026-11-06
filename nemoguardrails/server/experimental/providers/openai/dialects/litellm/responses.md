# LiteLLM Responses dialect

This experimental [dialect](responses.py) adds optional, opaque `model` metadata
at the root of SSE events. Requests, buffered responses, and stream lifecycle
checks use the [native Responses policy](../../responses/README.md).

## Supported boundary

Requests require one nonempty `input` string and explicit `store: false`.
Generation must be foreground; tools, conversation history, additional plaintext
instructions, and replacement are unsupported. Buffered and streaming output
support one assistant text response under the native content restrictions.

Reviewed response metadata, including `store` and `tool_usage`, remains opaque:
its values and nested structure are not inspected. The event's `model` field
does not select events or supply guarded text. Unknown root fields and events,
additional content channels, and nested LiteLLM SSE errors fail closed when
output inspection is enabled. Disabling output inspection retains raw response
relay; request restrictions still apply.

## Explicit selection

Select the endpoint when constructing the proxy integration:

```python
from dataclasses import replace

from nemoguardrails.server.experimental.providers.openai.integration import create_openai_integration
from nemoguardrails.server.experimental.providers.openai.dialects.litellm.responses import (
    LITELLM_RESPONSES_ENDPOINT,
)

integration = replace(
    create_openai_integration(base_url="https://your-gateway.example/v1"),
    endpoints=(LITELLM_RESPONSES_ENDPOINT,),
)
```

Pass this integration to the proxy application with its Rails configuration.
The transparent HTTP proxy uses the `server` extra and the application's existing
upstream authorization; no LiteLLM SDK or additional dialect-specific key is
required. The gateway URL and request model do not select the dialect. Selecting
Responses alone leaves other routes on the application's unguarded passthrough
path; restrict exposed routes as described in the native policy.

## Compatibility evidence

The source baseline is LiteLLM `v1.105.1`, pinned to
[`d665dc1a6315847fcad6cc2c1264ceb47c06abce`](https://github.com/BerriAI/litellm/tree/d665dc1a6315847fcad6cc2c1264ceb47c06abce).
The reviewed sources are the
[Responses types](https://github.com/BerriAI/litellm/blob/d665dc1a6315847fcad6cc2c1264ceb47c06abce/litellm/types/llms/openai.py)
and [OpenAI response transformation](https://github.com/BerriAI/litellm/blob/d665dc1a6315847fcad6cc2c1264ceb47c06abce/litellm/llms/openai/responses/transformation.py).
A deployment's advertised version does not establish the absence of patches.

Two buffered/streaming cassettes in
`tests/recorded/server/experimental/openai/cassettes/test_litellm_responses/`
capture `openai/openai/gpt-5-nano` through NVIDIA's inference gateway with
`store: false`. They cover shared response metadata, the SSE `model` annotation,
text deltas, and repeated snapshots. This evidence is specific to that route;
Chat-to-Responses bridging and other LiteLLM backends require separate validation.
Other inherited events and rejection cases use synthetic unit fixtures.

The shared recording sanitizer retains content and nested item identifiers,
subject to credential filtering and metadata normalization. Replay compares
sanitized bytes, not original network chunk boundaries. Unsupported fields must
not be removed to make replay pass.

## Validation and export

```bash
NVIDIA_INFERENCE_KEY=... make record-cassettes \
  RECORDED_TESTS=tests/recorded/server/experimental/openai/test_litellm_responses.py \
  RECORDED_REQUIRED_KEYS=NVIDIA_INFERENCE_KEY

make replay-cassettes \
  RECORDED_TESTS=tests/recorded/server/experimental/openai/test_litellm_responses.py

uv run --locked python -m nemoguardrails.server.experimental.provider.contract_export \
  nemoguardrails.server.experimental.providers.openai.dialects.litellm.responses:LITELLM_RESPONSES_ENDPOINT \
  --output nemoguardrails/server/experimental/contracts/openai/_generated/litellm-responses.guard.yaml
```

Capture saves the complete sanitized response; offline replay validates policy,
passthrough, and blocking with synthetic checker decisions. Replay is required
after a refresh and uses no live credentials. The exported YAML documents the
bound declarations and is not a runtime input; use `--check` instead of
`--output` to detect drift.
