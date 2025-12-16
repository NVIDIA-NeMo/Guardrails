# Parallel Execution of Input and Output Rails

You can configure input and output rails to run in parallel. This can improve latency and throughput.

### When to Use Parallel Rails Execution

Use parallel execution:
- For I/O-bound rails such as external API calls to LLMs or third-party integrations.
- If you have two or more independent input or output rails without shared state dependencies.
- In production environments where response latency affects user experience and business metrics.

### When Not to Use Parallel Rails Execution

Avoid parallel execution:
- For CPU-bound rails; it might not improve performance and can introduce overhead.
- During development and testing for debugging and simpler workflows.

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
    streaming:
      enabled: True
      chunk_size: 200
      context_size: 50
      stream_first: True
streaming: True
```
