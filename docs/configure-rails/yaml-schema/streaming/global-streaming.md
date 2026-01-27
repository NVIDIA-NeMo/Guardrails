---
title:
  page: Streaming LLM Responses in Real-Time
  nav: Streaming LLM Responses
description: Enable and use streaming mode for LLM responses in real-time in the NeMo Guardrails library.
topics:
- Configuration
- Streaming
tags:
- Streaming
- LLM
- stream_async
- Python API
content:
  type: how_to
  difficulty: technical_beginner
  audience:
  - engineer
  - AI Engineer
---

# Streaming LLM Responses in Real-Time

The NeMo Guardrails library supports streaming LLM responses in real-time through the `stream_async()` method. No configuration is required to enable streaming—simply use `stream_async()` instead of `generate_async()`.

## Basic Usage

```python
from nemoguardrails import LLMRails, RailsConfig

config = RailsConfig.from_path("./config")
rails = LLMRails(config)

messages = [{"role": "user", "content": "Hello!"}]

async for chunk in rails.stream_async(messages=messages):
    print(chunk, end="", flush=True)
```

---

## Streaming With Output Rails

When using output rails with streaming, you must configure [output rail streaming](output-rail-streaming.md):

```yaml
rails:
  output:
    flows:
      - self check output
    streaming:
      enabled: True
```

If output rails are configured but `rails.output.streaming.enabled` is not set to `True`, calling `stream_async()` will raise an `StreamingNotSupportedError`.

---

## Streaming With Handler (Deprecated)

> **Warning:** Using `StreamingHandler` directly is deprecated and will be removed in a future release. Use `stream_async()` instead.

For advanced use cases requiring more control over token processing, you can use a `StreamingHandler` with `generate_async()`:

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

## CLI Usage

Use the `--streaming` flag with the chat command:

```bash
nemoguardrails chat path/to/config --streaming
```

---

## Token Usage Tracking

When using `stream_async()`, NeMo Guardrails automatically enables token usage tracking by setting `stream_usage = True` on the underlying LLM model.

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
