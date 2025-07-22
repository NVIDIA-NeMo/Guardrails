# Streaming

If the application LLM supports streaming, you can configure NeMo Guardrails to stream tokens as well.

For information about configuring streaming with output guardrails, refer to the following:

- For configuration, refer to [streaming output configuration](../../user-guides/configuration-guide.md#streaming-output-configuration).
- For sample Python client code, refer to [streaming output](../../getting-started/5-output-rails/README.md#streaming-output).

## Configuration

To activate streaming on a guardrails configuration, add the following to your `config.yml`:

```yaml
streaming: True
```

## Usage

### Chat CLI

You can enable streaming when launching the NeMo Guardrails chat CLI by using the `--streaming` option:

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

### Using External Token Generators

You can also provide your own async generator that yields tokens, which is useful when:

- You want to use a different LLM provider that has its own streaming API
- You have pre-generated responses that you want to stream through guardrails
- You want to implement custom token generation logic
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

messages = [{"role": "user", "content": "The most famous program ever written is"}]"}]

# use the external generator with guardrails
async for chunk in app.stream_async(
    messages=messages,
    generator=my_token_generator()
):
    print(f"CHUNK: {chunk}")
```

When using an external generator:

- The internal LLM generation is completely bypassed
- Output rails are still applied to the LLM responses returned by the external generator, if configured
- The generator should yield string tokens

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

This feature enables seamless integration of NeMo Guardrails with any streaming LLM or token source while maintaining all the safety features of output rails.

### Server API

To make a call to the NeMo Guardrails Server in streaming mode, you have to set the `stream` parameter to `True` inside the JSON body. For example, to get the completion for a chat session using the `/v1/chat/completions` endpoint:

```
POST /v1/chat/completions
```

```json
{
    "config_id": "some_config_id",
    "messages": [{
      "role":"user",
      "content":"Hello! What can you do for me?"
    }],
    "stream": true
}
```

### Streaming for LLMs deployed using HuggingFacePipeline

We also support streaming for LLMs deployed using `HuggingFacePipeline`.
One example is provided in the [HF Pipeline Dolly](https://github.com/NVIDIA/NeMo-Guardrails/tree/develop/examples/configs/llm/hf_pipeline_dolly/README.md) configuration.

To use streaming for HF Pipeline LLMs, you first need to set the streaming flag in your `config.yml`.

```yaml
streaming: True
```

Then you need to create an `nemoguardrails.llm.providers.huggingface.AsyncTextIteratorStreamer` streamer object,
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
