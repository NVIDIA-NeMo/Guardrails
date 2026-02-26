---
title:
  page: "Streaming Responses"
  nav: "Streaming"
description: "Stream LLM responses in real-time with the stream_async method."
keywords: ["stream_async", "streaming responses", "real-time LLM", "StreamingHandler", "async streaming"]
topics: ["generative_ai", "developer_tools"]
tags: ["llms", "ai_inference", "ai_platforms"]
content:
  type: tutorial
  difficulty: technical_intermediate
  audience: ["data_scientist", "engineer"]
---

# Streaming Generated Responses in Real-Time

If the application LLM supports streaming, the NeMo Guardrails library can stream tokens as well. Streaming is automatically enabled when you use the `stream_async()` method - no configuration is required.

For information about configuring streaming with output guardrails, refer to the following:

- For configuration, refer to [](../../configure-rails/yaml-schema/streaming/output-rail-streaming.md).
- For sample Python client code, refer to [](../../getting-started/tutorials/index.md).

## Usage

### Chat CLI

You can enable streaming when launching the NeMo Guardrails library chat CLI by using the `--streaming` option:

```bash
nemoguardrails chat --config=examples/configs/streaming --streaming
```

### Python API

You can use the streaming directly from the python API in two ways:

1. Simple: receive just the chunks (tokens).
2. Full: receive both the chunks as they are generated and the full response at the end.

For the simple usage, you need to call the `stream_async` method on the `LLMRails` instance:

```python
from nemoguardrails import LLMRails

app = LLMRails(config)

history = [{"role": "user", "content": "What is the capital of France?"}]

async for chunk in app.stream_async(messages=history):
    print(f"CHUNK: {chunk}")
    # Or do something else with the token
```

For the full usage, you need to provide a `StreamingHandler` instance to the `generate_async` method on the `LLMRails` instance:

```python
from nemoguardrails import LLMRails
from nemoguardrails.streaming import StreamingHandler

app = LLMRails(config)

history = [{"role": "user", "content": "What is the capital of France?"}]

streaming_handler = StreamingHandler()

async def process_tokens():
    async for chunk in streaming_handler:
        print(f"CHUNK: {chunk}")
        # Or do something else with the token

asyncio.create_task(process_tokens())

result = await app.generate_async(
    messages=history, streaming_handler=streaming_handler
)
print(result)
```

> **Warning:** Using `StreamingHandler` directly is deprecated and will be removed in a future release. Use `stream_async()` instead.

(external-async-token-generators)=

### Using External Async Token Generators

You can also provide your own async generator that yields tokens, which is useful when:

- You want to use a different LLM provider that has its own streaming API.
- You have pre-generated responses that you want to stream through guardrails.
- You want to implement custom token generation logic.
- You want to test your output rails or its config in streaming mode on predefined responses without actually relying on an actual LLM generation.

To use an external generator, pass it to the `generator` parameter of `stream_async`:

```python
from nemoguardrails import LLMRails
from typing import AsyncIterator

app = LLMRails(config)

async def my_token_generator() -> AsyncIterator[str]:
    # This could be from OpenAI API, Anthropic API, or any other LLM API that already has a streaming token generator. Mocking the stream here, for a simple example.
    tokens = ["Hello", " ", "world", "!"]
    for token in tokens:
        yield token

messages = [{"role": "user", "content": "The most famous program ever written is"}]

# use the external generator with guardrails
async for chunk in app.stream_async(
    messages=messages,
    generator=my_token_generator()
):
    print(f"CHUNK: {chunk}")
```

When using an external generator:

- The internal LLM generation is completely bypassed.
- Output rails are still applied to the LLM responses returned by the external generator, if configured.
- The generator should yield string tokens.

Example with a real LLM API:

```python
async def openai_streaming_generator(messages) -> AsyncIterator[str]:
    """Example using OpenAI's streaming API."""
    import openai

    stream = await openai.ChatCompletion.create(
        model="gpt-4o",
        messages=messages,
        stream=True
    )

    # Yield tokens as they arrive
    async for chunk in stream:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

config = RailsConfig.from_path("config/with_output_rails")
app = LLMRails(config)

async for chunk in app.stream_async(
    messages=[{"role": "user", "content": "Tell me a story"}],
    generator=openai_streaming_generator(messages)
):
    # output rails will be applied to these chunks
    print(chunk, end="", flush=True)
```

This feature enables seamless integration of the NeMo Guardrails library with any streaming LLM or token source while maintaining all the safety features of output rails.

## Streaming Metadata

When using `stream_async()`, you can receive per-chunk metadata (e.g., token usage, finish reason) by setting `include_metadata=True`:

```python
async for chunk in rails.stream_async(messages=messages, include_metadata=True):
    print(chunk)
```

With `include_metadata=True`, each chunk is a `dict` with a mandatory `"text"` key. The final chunk also includes a `"metadata"` key containing `response_metadata` (finish reason, model name) and `usage_metadata` (token counts):

```python
{"text": "Hello"}
{"text": "!"}
{"text": "", "metadata": {
    "response_metadata": {"finish_reason": "stop", "model_name": "gpt-4o"},
    "usage_metadata": {"input_tokens": 75, "output_tokens": 9, "total_tokens": 84}
}}
```

Without `include_metadata`, chunks are plain strings (default behavior).

```{deprecated}
The `include_generation_metadata` parameter is deprecated. Use `include_metadata` instead. It will be removed in version 0.22.0.
```

## Token Usage Tracking

Token usage statistics are available when streaming responses, depending on provider support. When the provider does not return token usage statistics, the final chunk's `metadata` will contain `response_metadata` and `usage_metadata` set to `None`.

### Accessing Token Usage Information

You can access token usage statistics through the detailed logging capabilities of the NeMo Guardrails library. Use the `log` generation option to capture comprehensive information about LLM calls, including token usage:

```python
response = rails.generate(messages=messages, options={
    "log": {
        "llm_calls": True,
        "activated_rails": True
    }
})

for llm_call in response.log.llm_calls:
    print(f"Task: {llm_call.task}")
    print(f"Total tokens: {llm_call.total_tokens}")
    print(f"Prompt tokens: {llm_call.prompt_tokens}")
    print(f"Completion tokens: {llm_call.completion_tokens}")
```

Alternatively, you can use the `explain()` method to get a summary of token usage:

```python
info = rails.explain()
info.print_llm_calls_summary()
```

For more information about streaming token usage support across different providers, refer to the [LangChain documentation on token usage tracking](https://python.langchain.com/docs/how_to/chat_token_usage_tracking/#streaming). For detailed information about accessing generation logs and token usage, see [](generation-options.md#detailed-logging-information) and [](../../observability/logging/index.md).

```{note}
For streaming while using the Guardrails API server, refer to [](../using-fastapi-server/chat-with-guardrailed-model.md#streaming-responses).
```

### Streaming for LLMs Deployed Using HuggingFacePipeline

We also support streaming for LLMs deployed using `HuggingFacePipeline`.
One example is provided in the [HF Pipeline Dolly](https://github.com/NVIDIA-NeMo/Guardrails/tree/develop/examples/configs/llm/hf_pipeline_dolly/README.md) configuration.

To use streaming for HF Pipeline LLMs, you need to create an `nemoguardrails.llm.providers.huggingface.AsyncTextIteratorStreamer` streamer object,
add it to the `kwargs` of the pipeline and to the `model_kwargs` of the `HuggingFacePipelineCompatible` object.

```python
from nemoguardrails.llm.providers.huggingface import AsyncTextIteratorStreamer

# instantiate tokenizer object required by LLM
streamer = AsyncTextIteratorStreamer(tokenizer, skip_prompt=True)
params = {"temperature": 0.01, "max_new_tokens": 100, "streamer": streamer}

pipe = pipeline(
    # all other parameters
    **params,
)

llm = HuggingFacePipelineCompatible(pipeline=pipe, model_kwargs=params)
```
