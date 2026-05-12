---
title: Speculative Generation
description: Run input rails and main LLM generation concurrently to reduce end-to-end latency.
---

# Speculative Generation

Speculative generation runs input-rail and main LLM response generation in parallel, rather than sequentially.
If response generation takes longer than the input-rail latency, this hides the latency of the input-rail check.
The tradeoff is that the main LLM will begin generating a response for unsafe requests, with a corresponding token cost.
However, responses are always checked by output rails before being returned to the client so no unsafe responses will be seen.

## When to use Speculative Generation

In many applications, safe requests are much more likely than unsafe requests.
Speculative generation takes advantage of this by assuming all requests are safe for generation.
Assuming a 2% rate of unsafe requests, the remaining 98% of safe requests will hide the input-rail latency by running in parallel with response generation.
The cost of this latency saving is that tokens for the 2% of unsafe requests will be generated and then discarded.
To decide whether Speculative Generation makes sense for your use-case, explore the unsafe request rate and potential latency savings.

:::{admonition} Experimental Feature
Speculative generation currently requires the opt-in IORails engine.
To enable IORails, set `NEMO_GUARDRAILS_IORAILS_ENGINE=1`.
Speculative generation is supported only for non-streaming requests (`generate_async`).
When speculative generation is enabled, streaming requests (`stream_async`) fall back to sequential execution and emit a warning.
:::

## How It Works

Without speculative generation, the IORails engine runs the input rails first and only starts the main LLM call once the input is determined to be safe:

1. Run input rails on the user message. If the input is unsafe, return the refusal message and stop.
2. If the input is safe, generate a response from the main LLM.
3. Run output rails on the LLM response. If the output is unsafe, return the refusal message and stop.
4. Return the response.

With speculative generation enabled, the input rails and the main LLM call start at the same time and race to completion:

1. Start the input rails and the main LLM call in parallel.
2. Wait for whichever finishes first, then resolve the race:
   - If the input rails finish first and the input is unsafe, cancel the LLM call and return the refusal message.
   - If the input rails finish first and the input is safe, wait for the LLM call to finish.
   - If the LLM call finishes first, wait for the input-rail verdict; discard the response and return the refusal message if the input is unsafe.
3. Run output rails on the LLM response.
4. Return the response, or the refusal message if output rails blocked it.

The engine handles three outcomes:

| Outcome | Behavior |
|---|---|
| Input rails finish first, input is **unsafe** | The main LLM call is cancelled. The user receives the refusal message. |
| Input rails finish first, input is **safe** | The engine waits for the main LLM call to finish, then runs output rails. |
| Main LLM finishes first | The engine waits for the input-rail verdict. If unsafe, the generated response is discarded and the user receives the refusal message. |

Output rails always run after the main LLM completes.
Speculative generation does not change the output-rail path.

## Configuration Example

To enable speculative generation, set `speculative_generation: True` under `rails.input` in the `config.yml` file.
Speculative generation requires the IORails engine; see [IORails Engine](parallel-rails.md#iorails-engine) for how to enable it.

```yaml
models:
  - type: main
    engine: nim
    model: meta/llama-3.1-70b-instruct
  - type: content_safety
    engine: nim
    model: nvidia/llama-3.1-nemoguard-8b-content-safety

rails:
  input:
    speculative_generation: True
    flows:
      - content safety check input $model=content_safety
  output:
    flows:
      - content safety check output $model=content_safety
```

`speculative_generation` and `parallel` can be combined.
Input rails will run in parallel with each other and concurrently with the main LLM call.
