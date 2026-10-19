# LiteLLM Chat dialect

This experimental dialect applies the `single_text.v1` guardrail profile to
reviewed LiteLLM Chat Completions wire shapes. Typed Python declarations in
[chat_completions.py](chat_completions.py) and [stream.py](stream.py) are
authoritative. Dialect identity, guardrail capability, and tested deployment
are distinct: this is an upstream LiteLLM dialect, not a gateway-specific one
or a promise of compatibility with every LiteLLM release and backend.

## Supported boundary

| Location | Policy |
| --- | --- |
| Request | Reuse the OpenAI single-user-text request binding, including disabled tools, multimodal content, and log probabilities. No additional request fields are enabled. |
| Buffered `choices[0].message.provider_specific_fields` | Optional and nullable, with optional `refusal: null` and absent/null/empty `thinking_blocks`. Other members are rejected. |
| Buffered `choices[0].provider_specific_fields` | Optional and nullable; reviewed `native_finish_reason` metadata is opaque. Other members are rejected. |
| Buffered message `images`, `thinking_blocks`, `reasoning_items` | Absent, null, or empty arrays only. Nonempty content is unsupported. |
| Remaining buffered fields | Inherit the OpenAI response constraints and reviewed opaque metadata. The dialect's choice and message remain closed even under a trusted unknown-field override. |
| Streaming root and delta `provider_specific_fields` | Absent, null, or empty objects only. These are separate from buffered choice extensions. |
| Streaming delta `images`, `thinking_blocks`, `reasoning_items`, `annotations` | Absent, null, or empty arrays only. |
| Streaming event selection and lifecycle | Rebind the OpenAI content rule to the LiteLLM projection; retain error selection, metadata roles, sentinel/keepalive behavior, and fresh OpenAI lifecycle hooks. Choice index remains strict integer zero. |

Extension containers are **constrained**, not wholly opaque. Their members have
individual policies: `native_finish_reason` is opaque metadata, null refusal
is disabled, and thinking blocks are constrained to empty values. Reasoning or
refusal text, token arrays, and unreviewed names are rejected. Reviewed opaque
fields such as usage remain uninspected, including their contents; this dialect
does not claim to inspect all JSON text or validate all metadata value types.

As in the base integration, disabling output inspection relays response bytes
without applying response projections or stream classification. Guarded request
validation still applies. This dialect does not change that distinction or
enable replacement outcomes. Empty/null assistant content retains the base
no-text behavior. The selected bindings preserve original bytes on allowed paths.

## Explicit selection

The private router accepts a trusted endpoint declaration:

```python
from nemoguardrails.server.experimental.providers.openai.dialects.litellm.chat_completions import (
    LITELLM_CHAT_COMPLETIONS_ENDPOINT,
)
from nemoguardrails.server.experimental.providers.openai.integration import create_openai_chat_router

router = create_openai_chat_router(
    checker=checker,
    dispatch=dispatcher.dispatch,
    stream_dispatch=dispatcher.stream_dispatch,
    endpoint=LITELLM_CHAT_COMPLETIONS_ENDPOINT,
)
```

`checker` and `dispatcher` are application-owned dependencies, not constructed
by this snippet. Use the existing server dependencies and HTTP transport; no
LiteLLM SDK, optional LiteLLM extra, or dialect-specific API key is required.
Upstream credentials follow the application's existing HTTP authorization path.
This uses the experimental transparent HTTP proxy, not LangChain or a LiteLLM
client wrapper. Omitting `endpoint` retains the OpenAI declaration. Neither the
hostname, request model string, response fields, nor exported YAML selects it.

Selection assumes the reviewed wire behavior matches the configured deployment.
Unknown shapes fail closed when inspected; accepted names and opaque fields do
not prove backend semantics. A gateway in front of vLLM needs separate evidence
for what that gateway actually exposes. Do not automatically union dialects.

## Evidence and scope

The source baseline is LiteLLM tag `v1.102.1`, resolved to immutable commit
[`d09bbae1c6df463e425558f60d460437193635da`](https://github.com/BerriAI/litellm/tree/d09bbae1c6df463e425558f60d460437193635da).
This identifies the source reviewed for the field policy, not the build running
in any deployment. See the [LiteLLM API stability policy](https://docs.litellm.ai/docs/api_stability_policy)
for upstream compatibility guidance.

| File | Observed behavior | SHA-256 of source bytes |
| --- | --- | --- |
| [`litellm/types/utils.py`](https://github.com/BerriAI/litellm/blob/d09bbae1c6df463e425558f60d460437193635da/litellm/types/utils.py) | Message/choice and stream root/delta declarations; buffered `Choices` stores the original finish reason when normalizing it. | `a1698ae315ba0c618aa25ed037338c8b47b8f821680176e31f95f588ff09b26f` |
| [`convert_dict_to_response.py`](https://github.com/BerriAI/litellm/blob/d09bbae1c6df463e425558f60d460437193635da/litellm/litellm_core_utils/llm_response_utils/convert_dict_to_response.py) | Message conversion preserves existing extensions, collects unknown message members, and copies thinking blocks into the bag. Choice conversion collects unknown choice members. | `303a5f0b82318e8f87d0c3b31db791e8370d41cc069741dd503b8a6e2b31948b` |

### Inventory decisions

This table accounts for the reviewed Chat surface, not every backend-specific
key that LiteLLM's open dictionaries could contain.

| Source fields | Policy decision |
| --- | --- |
| Chat request `model`, `messages`, `n`, `stream`, OpenAI sampling/usage controls | Inherit existing OpenAI policy; this dialect does not promise full provider-schema validation. |
| Request `caching`, `num_retries`, `guardrails`, `fallbacks`, `context_window_fallback_dict` | Reject as undeclared. These affect execution/routing or upstream guardrail selection, not just response metadata; enabling them requires a separate operational policy decision. |
| Message `content`, `role`, `audio`, `function_call`, `tool_calls`, `reasoning_content`, `annotations` | Inherit guarded text and existing role/content constraints. |
| Message `images`, `thinking_blocks`, `reasoning_items` | Explicit empty-only arrays; they are additional content channels, not accounting metadata. |
| Message extension `refusal`, `thinking_blocks` | Null-only refusal and empty-only thinking blocks. Other extension names require review. |
| Choice `message`, `finish_reason`, `index`, `logprobs` | Inherit OpenAI policy. Buffered index/finish reason are opaque; log probabilities remain disabled. |
| Choice extension `native_finish_reason` | Opaque provider termination metadata. It is not used to select or end stream events. |
| Stream root/delta extension bags | Recognize containers; reject nonempty bags pending field-specific review. Buffered extension membership is not assumed to apply to streams. |
| Stream delta content arrays and annotations | Explicit empty-only arrays, as above. |
| Root response/stream identity, usage and other existing opaque fields | Preserve unchanged. Reasoning-token counts and cost inside usage are not reasoning text. |
| Source constructor extras, including choice `enhancements`, and unknown backend fields | Reject unless already in the inherited opaque inventory. Open upstream dictionaries are not an automatic passthrough policy. |

Recorded buffered and streaming exchanges are covered separately in
[#2455](https://github.com/NVIDIA-NeMo/Guardrails/pull/2455). They exercise this
dialect through the concrete HTTP transport against recorded gateway traffic;
they do not establish direct vLLM compatibility.

Recordings do not establish a complete provider schema. The unit fixtures cover
additional source-backed shapes, including native finish metadata and empty
stream bags; they are not provider recordings. Neither a source pin nor replay
proves compatibility with all model routes. No live calls are needed for tests.

## Application-owned extensions

A trusted application can bind a reviewed Python projection to its own endpoint
and pass that endpoint to the router. For harmless additional metadata, declare
the specific member opaque and rebind the containing models; do not disable
output inspection just to preserve metadata. Regenerate the exported contract
and test both passthrough and rejection after changing a boundary. Additional
content needs supported inspection or an explicit decision to leave it opaque.

The default dialect remains closed. The existing unknown-field override does
not relax declared constraints or this dialect's closed response objects.
There is no request-controlled opt-out, YAML loader, wildcard metadata allowance,
or new deployment configuration system. Broader operator configuration can be
added separately if needed.

## Contract export

Generate the complete effective document using the shared exporter:

```bash
uv run --locked python -m nemoguardrails.server.experimental.provider.contract_export \
  nemoguardrails.server.experimental.providers.openai.dialects.litellm.chat_completions:LITELLM_CHAT_COMPLETIONS_ENDPOINT \
  --output nemoguardrails/server/experimental/contracts/openai/_generated/litellm-chat-completions.guard.yaml
```

Use `--check` instead of `--output` to detect drift. The
[exported contract](../../../../contracts/openai/_generated/litellm-chat-completions.guard.yaml)
is a resolved inspection document: inherited fields are included, not a YAML
inheritance program. Runtime construction does not import the exporter or load
this artifact. Export retains the base format's documented limitations, including
Python strictness, case-alias checks, and handwritten stream lifecycle behavior.
