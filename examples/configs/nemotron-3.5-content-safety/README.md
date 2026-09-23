# Nemotron 3.5 Content Safety Usage Example

Input and output content-safety rails backed by
[`nvidia/nemotron-3.5-content-safety`](https://build.nvidia.com/nvidia/nemotron-3.5-content-safety).

```bash
export NVIDIA_API_KEY=nvapi-...
nemoguardrails chat --config=examples/configs/nemotron-3.5-content-safety --verbose
```

`--verbose` prints each LLM call, so the content-safety call and its raw verdict are visible alongside the
conversation.

## How this differs from the NemoGuard 8B example

The older `nvidia/llama-3.1-nemoguard-8b-content-safety` returns a JSON object while this model returns
flat text strings.

Its chat template builds the whole safety prompt itself: the taxonomy, the instructions and the output format are
all injected server-side, and whatever the prompt sends is inserted between `<BEGIN CONVERSATION>` and
`<END CONVERSATION>` as the material to classify. The reply is plain lines, not JSON:

```text
User Safety: unsafe
Response Safety: safe
Safety Categories: Criminal Planning/Confessions, Violence
```

So `prompts.yml` uses `messages:` carrying only the turns to classify, and the
`nemotron_content_safety_parse_*` output parsers. Pasting a taxonomy into the prompt, as the 8B example does,
would have the model classify the taxonomy as if the user had written it.

## Running against a local model

The chat template ships inside the model repository, so any stack that applies it -- vLLM, SGLang, a locally
downloaded NIM -- behaves the same way as build.nvidia.com. The prompts do not change. Point the model entry at
the local server:

```yaml
  - type: content_safety
    engine: openai
    model: nvidia/Nemotron-3.5-Content-Safety
    parameters:
      base_url: http://localhost:8000/v1
      api_key: EMPTY
      chat_template_kwargs:
        enable_thinking: false
        request_categories: "/categories"
```

```bash
vllm serve nvidia/Nemotron-3.5-Content-Safety
```

`engine: openai` plus `parameters.base_url` reaches any OpenAI-compatible server through the built-in client,
with no LangChain dependency.

Two things to be aware of:

- **If a proxy strips unknown body fields, `chat_template_kwargs` never arrives** and the template falls back to
  its defaults, `/no_think` and `/no_categories`. Verdicts still parse, but the `Safety Categories` line silently
  disappears and `policy_violations` stays empty. Nothing errors, so check a known-unsafe prompt returns categories
  before trusting them.
- **A server that does not apply the chat template needs a different prompt**, one that reproduces the taxonomy
  and the output-format instructions itself. This example does not ship that variant: it would hand-copy roughly
  fifty lines out of the model repository's `chat_template.jinja`, and the copy drifts silently when NVIDIA
  revises the checkpoint. If you need it, add it as a second prompt with `mode:` set and select it with the
  config-level `prompting_mode`; do not edit the prompt below in place, because a self-contained prompt sent to a
  server that *does* template gets wrapped a second time and the model then classifies your taxonomy as the user's
  input.

Note that the prompt cannot bypass templating from this side: the default framework always calls
`/v1/chat/completions`, and `mode: text` on a model entry is ignored on that path.

## Notes

- **Do not add a system message.** The chat template reads `messages[0]` when it is a system message and then never
  emits it, so its content is silently dropped.
- **Roles must alternate, starting with `user`.** An output rail invoked with no user message in context renders an
  empty first message, which `_render_messages` drops, leaving an assistant-only list that the endpoint rejects with
  `Conversation roles must alternate`.
- **`request_categories: "/categories"` is required** for a `Safety Categories` line to appear at all. Without it
  the model defaults to `/no_categories` and `policy_violations` is always empty.
- **`rails.config.content_safety.reasoning.enabled` does not control this model.** That flag only feeds the
  `reasoning_enabled` Jinja variable used by the Nemotron Content Safety Reasoning 4B text prompt. Here reasoning is
  `parameters.chat_template_kwargs.enable_thinking`.
- **Raise `max_tokens` before enabling thinking.** With `enable_thinking: true` the reasoning phase consumes the
  budget; measured completions ranged 38-184 tokens, so use ~512. Too small a budget returns empty content with
  `finish_reason="length"`, which the parser reports as an unparseable verdict and the rail fails closed.
- **`Safety Categories` is one flat list with no turn attribution.** On a mixed verdict the output rail's
  `policy_violations` may carry categories triggered by the user turn. Read them as "categories seen in this
  exchange", not "categories the bot response violated".
- **On the NIM the reasoning trace arrives in `reasoning_content`, never inline in `content`.** A raw vLLM or
  `transformers` deployment may emit `<think>...</think>` inline instead; the parsers handle both.

## Files

- `config.yml` - the main and content-safety models, the `chat_template_kwargs`, and the rail flows.
- `prompts.yml` - the message-based prompts and the output parsers.
