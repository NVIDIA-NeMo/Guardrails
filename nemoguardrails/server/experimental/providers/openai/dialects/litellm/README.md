# Reviewed LiteLLM Chat dialect

This experimental dialect supports a narrow OpenAI Chat Completions subset
observed through a LiteLLM-facing gateway. Typed Python declarations in
[chat_completions.py](chat_completions.py) are authoritative. It is not a claim
of compatibility with every LiteLLM release, backend, model route, or raw vLLM
server. The framework capability remains `single_text.v1`.

## Supported boundary

| Location | Policy |
| --- | --- |
| Request | Reuse the OpenAI single-user-text request binding, including disabled tools, multimodal content, and log probabilities. No additional request fields are enabled. |
| Buffered `choices[0].message.provider_specific_fields` | Optional and nullable. If an object, it is closed and permits only optional `refusal: null`. |
| Buffered `choices[0].provider_specific_fields` | Optional and nullable. If an object, it must be empty. |
| Remaining buffered fields | Inherit the OpenAI response constraints and reviewed opaque metadata. The dialect's choice and message remain closed even under a trusted unknown-field override. |
| Streaming | Reuse the exact OpenAI classifier, event projections, and fresh per-request hooks. No extra root, choice, or delta fields are enabled, including `provider_specific_fields`. |

Nested extension containers are **constrained**, not opaque. Extra reasoning,
refusal text, token arrays, and unreviewed names are rejected. Absence, null, and
empty objects carry no additional content; the optional null refusal is disabled.
Do not change these containers into arbitrary dictionaries to accommodate a
different backend. Reviewed opaque fields such as usage remain uninspected,
including their contents; this dialect does not claim to inspect all JSON text.

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

Public LiteLLM source was reviewed at immutable commit
[`7efce7f8e05e58fb86e006a6427ab39b6db467fb`](https://github.com/BerriAI/litellm/tree/7efce7f8e05e58fb86e006a6427ab39b6db467fb):

| File | Observed behavior | SHA-256 of source bytes |
| --- | --- | --- |
| [`litellm/types/utils.py`](https://github.com/BerriAI/litellm/blob/7efce7f8e05e58fb86e006a6427ab39b6db467fb/litellm/types/utils.py) | `Message` and `Choices` accept optional nullable provider-specific dictionaries. Their full supported fields are broader than this dialect. | `124883cd7db4fbeaf454ebf5fa597bb5570b746a6464959419a829c2344b6079` |
| [`litellm/litellm_core_utils/llm_response_utils/convert_dict_to_response.py`](https://github.com/BerriAI/litellm/blob/7efce7f8e05e58fb86e006a6427ab39b6db467fb/litellm/litellm_core_utils/llm_response_utils/convert_dict_to_response.py) | Conversion preserves message extensions and collects fields outside `Message`/`Choices` into provider-specific containers. `refusal` is outside the pinned `Message` field inventory. | `cee5f74e2831753817a9d5550ae0da0fd47606dc71e4d2452314f18e561ceada` |

Separate deployment evidence is in the D-2 commit `1607aa70a464f2bb6fa4b650473c80a87f43f4e3`,
under `tests/recorded/server/experimental/openai/cassettes/test_chat_completions/`:
`test_buffered_chat_preserves_guarded_constrained_and_opaque_contract_fields.yaml`
and `test_nvidia_chat_source_profile_preserves_allowed_relay[buffered].yaml`.
Both contain message `{refusal: null}` and empty choice extension objects. The
recorded streaming successes use the existing OpenAI event shapes. These
recordings identify an OpenAI model route through the gateway, not raw vLLM.

The source pin does not identify the gateway's deployed LiteLLM version, and
recordings do not establish a complete provider schema. This deliberately
restricted policy is based on both observations, not advertised as universal
source/runtime parity. The unit fixtures are small synthetic shape examples,
not new provider recordings. No live calls are needed for tests.

## Export and downstream handoff

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

This change belongs after D-1 and before D-2. D-2 should explicitly select this
endpoint for the reviewed gateway recordings and replay them without editing
recorded payloads to fit the policy. D-3 may add trusted deployment-level dialect
selection while retaining the OpenAI protocol family; no registry or deployment
configuration format is added here. Direct vLLM, other LiteLLM routes, broader
conformance, and compiler preparation remain separate work.
