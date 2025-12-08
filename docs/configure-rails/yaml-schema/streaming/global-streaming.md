---
title: Global Streaming
description: Enable streaming mode for LLM token generation in config.yml.
---

# Global Streaming

Enable streaming mode for the main LLM generation at the top level of `config.yml`.

## Configuration

```yaml
streaming: True
```

## What It Does

When enabled, global streaming:

- Sets `streaming = True` on the underlying LLM model
- Enables `stream_usage = True` for token usage tracking
- Allows using the `stream_async()` method on `LLMRails`
- Makes the LLM produce tokens incrementally instead of all at once

## Default

`False`

---

## When to Use

### Streaming Without Output Rails

If you do not have output rails configured, only global streaming is needed:

```yaml
streaming: True
```

### Streaming With Output Rails

When using output rails with streaming, you must also configure [output rail streaming](output-rail-streaming.md):

```yaml
streaming: True

rails:
  output:
    flows:
      - self check output
    streaming:
      enabled: True
```

---

## Python API Usage

### Simple Streaming

```python
from nemoguardrails import LLMRails, RailsConfig

config = RailsConfig.from_path("./config")
rails = LLMRails(config)

messages = [{"role": "user", "content": "Hello!"}]

async for chunk in rails.stream_async(messages=messages):
    print(chunk, end="", flush=True)
```

### Streaming With Handler

For more control, use a `StreamingHandler`:

```python
from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.streaming import StreamingHandler
import asyncio

config = RailsConfig.from_path("./config")
rails = LLMRails(config)

streaming_handler = StreamingHandler()

async def process_tokens():
    async for chunk in streaming_handler:
        print(chunk, end="", flush=True)

asyncio.create_task(process_tokens())

result = await rails.generate_async(
    messages=[{"role": "user", "content": "Hello!"}],
    streaming_handler=streaming_handler
)
```

---

## Server API

Enable streaming in the request body by setting `stream` to `true`:

```json
{
    "config_id": "my_config",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
}
```

---

## Token Usage Tracking

When streaming is enabled, NeMo Guardrails automatically enables token usage tracking by setting `stream_usage = True` for the underlying LLM model.

Access token usage through the `log` generation option:

```python
response = rails.generate(messages=messages, options={
    "log": {
        "llm_calls": True
    }
})

for llm_call in response.log.llm_calls:
    print(f"Total tokens: {llm_call.total_tokens}")
    print(f"Prompt tokens: {llm_call.prompt_tokens}")
    print(f"Completion tokens: {llm_call.completion_tokens}")
```

---

## HuggingFace Pipeline Streaming

For LLMs deployed using `HuggingFacePipeline`, additional configuration is required:

```python
from nemoguardrails.llm.providers.huggingface import AsyncTextIteratorStreamer

# Create streamer with tokenizer
streamer = AsyncTextIteratorStreamer(tokenizer, skip_prompt=True)
params = {"temperature": 0.01, "max_new_tokens": 100, "streamer": streamer}

pipe = pipeline(
    # other parameters
    **params,
)

llm = HuggingFacePipelineCompatible(pipeline=pipe, model_kwargs=params)
```

---

## Related Topics

- [Output Rail Streaming](output-rail-streaming.md) - Configure streaming for output rails
- [Model Configuration](../model-configuration.md) - Configure the main LLM
