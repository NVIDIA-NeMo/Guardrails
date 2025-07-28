# Guardrails Configuration

Guardrails (or rails) implement *flows* based on their role. Rails fall into five main categories:

1. **Input rails**: Trigger when the system receives new user input.
2. **Output rails**: Trigger when the system generates new output for the user.
3. **Dialog rails**: Trigger after the system interprets a user message and identifies its canonical form.
4. **Retrieval rails**: Trigger after the system completes the retrieval step (when the `retrieve_relevant_chunks` action finishes).
5. **Execution rails**: Trigger before and after the system invokes an action.

You can configure active rails using the `rails` key in `config.yml` as shown in the following example:

```yaml
rails:
  # Input rails trigger when the system receives a new user message.
  input:
    flows:
      - check jailbreak
      - check input sensitive data
      - check toxicity
      - ... # Other input rails

  # Output rails trigger after the system generates a bot message.
  output:
    flows:
      - self check facts
      - self check hallucination
      - check output sensitive data
      - ... # Other output rails

  # Retrieval rails trigger when the system computes `$relevant_chunks`.
  retrieval:
    flows:
      - check retrieval sensitive data
```

Flows that aren't input, output, or retrieval rails become dialog rails and execution rails. These flows control dialog flow and action invocation timing. Dialog/execution rail flows don't require explicit enumeration in the config. Several configuration options control their behavior.

```yaml
rails:
  # Dialog rails trigger after the system interprets a user message and computes its canonical form.
  dialog:
    # Whether to use a single LLM call for generating user intent, next step, and bot message.
    single_call:
      enabled: False

      # Whether to fall back to multiple LLM calls if a single call fails.
      fallback_to_multiple_calls: True

    user_messages:
      # Whether to use only embeddings when interpreting user messages.
      embeddings_only: False
```

## Input Rails

Input rails process user messages. For example:

```colang
define flow self check input
  $allowed = execute self_check_input

  if not $allowed
    bot refuse to respond
    stop
```

Input rails can alter input by modifying the `$user_message` context variable.

## Output Rails

Output rails process bot messages. The `$bot_message` context variable contains the message to process. Output rails can modify the `$bot_message` variable, for example, to mask sensitive information.

To temporarily deactivate output rails for the next bot message, set the `$skip_output_rails` context variable to `True`.

### Streaming Output Configuration

Output rails provide synchronous responses by default. Enable streaming to receive responses sooner.

Set the top-level `streaming: True` field in your `config.yml` file.

For the output rails, add the `streaming` field and configuration parameters.

```yaml
rails:
  output:
    - rail name
  streaming:
    enabled: True
    chunk_size: 200
    context_size: 50
    stream_first: True
streaming: True
```

When streaming is enabled, the toolkit applies output rails to token chunks. If a rail blocks a token chunk, the toolkit returns a JSON error object in the following format:

```output
{
  "error": {
    "message": "Blocked by <rail-name> rails.",
    "type": "guardrails_violation",
    "param": "<rail-name>",
    "code": "content_blocked"
  }
}
```

When integrating with the OpenAI Python client, server code catches this JSON error and converts it to an API error following the OpenAI SSE format.

The following table describes the subfields for the `streaming` field:

```{list-table}
:header-rows: 1

* - Field
  - Description
  - Default Value

* - streaming.chunk_size
  - Specifies the number of tokens per chunk. The toolkit applies output guardrails to each token chunk.

    Larger values provide more meaningful information for rail assessment but add latency while accumulating tokens for a full chunk. Higher latency risk occurs when you specify `stream_first: False`.
  - `200`

* - streaming.context_size
  - Specifies the number of tokens to keep from the previous chunk for context and processing continuity.

    Larger values provide continuity across chunks with minimal latency impact. Small values might fail to detect cross-chunk violations. Specifying approximately 25% of `chunk_size` provides a good compromise.
  - `50`

* - streaming.enabled
  - When set to `True`, the toolkit executes output rails in streaming mode.
  - `False`

* - streaming.stream_first
  - When set to `False`, the toolkit applies output rails to chunks before streaming them to the client. Setting this field to `False` avoids streaming blocked content chunks.

    By default, the toolkit streams chunks as soon as possible and before applying output rails to them.
  - `True`
```

The following table shows how token count, chunk size, and context size interact to determine the number of rails invocations.

```{csv-table}
:header: Input Length, Chunk Size, Context Size, Rails Invocations

512,256,64,3
600,256,64,3
256,256,64,1
1024,256,64,5
1024,256,32,5
1024,256,32,5
1024,128,32,11
512,128,32,5
```

Refer to [](../getting-started/5-output-rails/README.md#streaming-output) for a code sample.

(parallel-rails)=

## Parallel Execution of Input and Output Rails

You can configure input and output rails to run in parallel. This can improve latency and throughput.

### When to Use Parallel Rails Execution

- Use parallel execution for I/O-bound rails such as external API calls to LLMs or third-party integrations.
- Enable parallel execution if you have two or more independent input or output rails without shared state dependencies.
- Use parallel execution in production environments where response latency affects user experience and business metrics.

### When Not to Use Parallel Rails Execution

- Avoid parallel execution for CPU-bound rails; it might not improve performance and can introduce overhead.
- Use sequential mode during development and testing for debugging and simpler workflows.

### Configuration Example

To enable parallel execution, set `parallel: True` in the `rails.input` and `rails.output` sections in the `config.yml` file. The following configuration example is tested by NVIDIA and shows how to enable parallel execution for input and output rails.

```{note}
Input rail mutations can lead to erroneous results during parallel execution because of race conditions arising from the execution order and timing of parallel operations. This can result in output divergence compared to sequential execution. For such cases, use sequential mode.
```

The following is an example configuration for parallel rails using models from NVIDIA Cloud Functions (NVCF). When you use NVCF models, make sure that you export `NVIDIA_API_KEY` to access those models.

```yaml
models:
  - type: main
    engine: nim
    model: meta/llama-3.1-70b-instruct
  - type: content_safety
    engine: nim
    model: nvidia/llama-3.1-nemoguard-8b-content-safety
  - type: topic_control
    engine: nim
    model: nvidia/llama-3.1-nemoguard-8b-topic-control

rails:
  input:
    parallel: True
    flows:
      - content safety check input $model=content_safety
      - topic safety check input $model=topic_control
  output:
    parallel: True
    flows:
      - content safety check output $model=content_safety
      - self check output
```

## Retrieval Rails

Retrieval rails process retrieved chunks stored in the `$relevant_chunks` variable.

## Dialog Rails

Dialog rails enforce predefined conversational paths. Define canonical forms for various user messages to trigger dialog flows. See the [Hello World](https://github.com/NVIDIA/NeMo-Guardrails/tree/develop/examples/bots/hello_world/README.md) bot for a basic example. The [ABC bot](https://github.com/NVIDIA/NeMo-Guardrails/tree/develop/examples/bots/abc/README.md) demonstrates dialog rails preventing the bot from discussing specific topics.

Dialog rails require a three-step process:

1. Generate canonical user message.
2. Decide next step(s) and execute them.
3. Generate bot utterance(s).

See [The Guardrails Process](../architecture/README.md#the-guardrails-process) for detailed description.

Each step may require an LLM call.

### Single Call Mode

NeMo Guardrails supports "single call" mode since version `0.6.0`. This mode performs all three steps using a single LLM call. Set the `single_call.enabled` flag to `True` to enable it.

```yaml
rails:
  dialog:
    # Whether to try to use a single LLM call for generating the user intent, next step and bot message.
    single_call:
      enabled: True

      # If a single call fails, whether to fall back to multiple LLM calls.
      fallback_to_multiple_calls: True
```

In typical RAG (Retrieval Augmented Generation) scenarios, this option provides latency improvement and uses fewer tokens.

```{important}
Currently, single call mode only predicts bot messages as next steps. The LLM cannot generalize and execute actions on dynamically generated user canonical form messages.
```

### Embeddings Only

Use embeddings of pre-defined user messages to determine the canonical form for user input. This speeds up dialog rails. Set the `embeddings_only` flag to enable this option.

```yaml
rails:
  dialog:
    user_messages:
      # Whether to use only embeddings when interpreting user messages.
      embeddings_only: True
      # Use only embeddings when similarity exceeds the specified threshold.
      embeddings_only_similarity_threshold: 0.75
      # When fallback is None, similarity below threshold triggers normal LLM user intent computation.
      # When set to a string value, that string becomes the intent.
      embeddings_only_fallback_intent: None
```

```{important}
Use this only when you provide sufficient examples. The 0.75 threshold triggers LLM calls for user intent generation when similarity falls below this value. Increase the threshold to 0.8 if you encounter false positives. Threshold values are model dependent.
```
