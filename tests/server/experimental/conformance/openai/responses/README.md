# Responses conformance

This suite checks field policies against provider source, examples, and recorded
exchanges. A field's presence is evidence to review, not permission to treat its
contents as opaque.

## Evidence and coverage

- [Pinned inventory](fixtures/README.md): 21 Responses components, inherited
  `allOf` properties, and the official text-input response example.
- [Native Responses recordings](../../../../../recorded/server/experimental/openai/test_responses.py):
  buffered/streaming success and invalid-model HTTP errors from OpenAI.
- [LiteLLM Responses recordings](../../../../../recorded/server/experimental/openai/test_litellm_responses.py)
  and [dialect source evidence](../../../../../../nemoguardrails/server/experimental/providers/openai/dialects/litellm/responses.md):
  gateway success in both modes and the reviewed root SSE `model` annotation.

`test_schema.py` audits the bound request, response, and event models and their
field classifications. `test_http.py` mutates copies of native recordings to
check admission, additional text channels, and snapshot consistency.
`dialects/test_litellm.py` checks the gateway boundary and shared native metadata.
Unmodified recordings are positive controls for the same injection path;
mutations stay in memory and are never recorded as provider evidence.

The pinned `Response` schema omits `store`, while its example includes it. Native
captures also contain `access_programs`, `billing`, `frequency_penalty`,
`presence_penalty`, and `tool_usage`. These six response fields belong to the
native policy; only the root SSE `model` annotation requires the LiteLLM dialect.
The native delta `obfuscation` field is also accounted for explicitly.

## Policy boundaries

| Surface | Policy |
| --- | --- |
| Request `input` | One guarded text string. |
| Request `store` | Required boolean `false`; guarded creation cannot request retrievable storage. |
| Instructions, history, tools, and other additional input | Rejected before dispatch, including with output inspection disabled. |
| Output text and deltas | One assistant text target. |
| Done, part, item, and lifecycle snapshots | Checked against accumulated text; altered snapshots must not escape as metadata. |
| Reasoning summaries/content, citations, and logprobs | Nonempty additional text channels are rejected. |
| Encrypted reasoning and reviewed response metadata | Opaque, including nested values; no claim of inspection. |

The [native policy](../../../../../../nemoguardrails/server/experimental/providers/openai/responses/README.md)
defines accepted forms, trusted overrides, and storage/retrieval limitations.
Request `store` restrictions do not change the opaque response echo.

## Run and extend

```bash
make test WORKERS=0 \
  TEST="tests/server/experimental/conformance/openai/responses tests/recorded/server/experimental/openai/test_responses.py tests/recorded/server/experimental/openai/test_litellm_responses.py" \
  ARGS="--block-network --record-mode=none"
```

For capture and replay, follow the [recording workflow](../../../../../recorded/README.md).
Select `test_responses.py` with `OPENAI_API_KEY` or `test_litellm_responses.py`
with `NVIDIA_INFERENCE_KEY`; offline replay needs no live credentials.
The [fixture guide](fixtures/README.md) gives inventory regeneration commands.

Follow the [shared reviewer probes](../../README.md#reviewer-probes), especially
lifecycle snapshots, changed final text after approved deltas, encrypted
reasoning state, and native metadata versus dialect event annotations.
