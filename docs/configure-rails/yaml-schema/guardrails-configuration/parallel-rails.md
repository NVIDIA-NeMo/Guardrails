---
title: Parallel Execution of Rails
description: Configure input and output rails to run in parallel for improved latency and throughput.
---

# Parallel Execution of Input and Output Rails

You can configure input and output rails to run in parallel. This can improve latency and throughput.

## IORails Engine

The IORails engine is an optimized execution engine that runs NemoGuard input and output rails in
parallel with dedicated model management. When all configured flows are compatible, the NeMo
Guardrails library automatically selects the IORails engine instead of the default LLMRails engine.

### Supported Flows

The IORails engine supports the following flows:

- `content safety check input` / `content safety check output`
- `topic safety check input`
- `jailbreak detection model`

If your configuration uses only these flows, the IORails engine is selected automatically.
Configurations that include custom flows, dialog rails, or other unsupported flows
fall back to the LLMRails engine.

### Opting Out

To disable the IORails engine and use LLMRails, set `use_iorails=False`:

```python
from nemoguardrails import RailsConfig
from nemoguardrails.guardrails.guardrails import Guardrails

config = RailsConfig.from_path("./config")
guardrails = Guardrails(config, use_iorails=False)
```

When using the CLI, the IORails engine is controlled by the `NEMO_GUARDRAILS_IORAILS_ENGINE`
environment variable:

```bash
NEMO_GUARDRAILS_IORAILS_ENGINE=1 nemoguardrails chat --config examples/configs/content_safety
```

## YAML-Based Parallel Execution

You can also configure existing LLMRails flows to run in parallel using the `parallel: True`
option in the `config.yml` file. This approach works with any flow type and does not require
the IORails engine.

### When to Use

Use YAML-based parallel execution:

- For I/O-bound rails such as external API calls to LLMs or third-party integrations.
- If you have two or more independent input or output rails without shared state dependencies.
- In production environments where response latency affects user experience and business metrics.

### When Not to Use

Avoid parallel execution:

- For CPU-bound rails; it might not improve performance and can introduce overhead.
- During development and testing for debugging and simpler workflows.

### Configuration Example

To enable parallel execution, set `parallel: True` in the `rails.input` and `rails.output` sections in the `config.yml` file.

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
