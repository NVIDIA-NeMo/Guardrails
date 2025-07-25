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

For the complete working example, check out this [demo script](https://github.com/NVIDIA/NeMo-Guardrails/tree/develop/examples/scripts/demo_streaming.py).

## Token Usage Tracking

When streaming is enabled, NeMo Guardrails automatically enables token usage tracking by setting the `stream_usage` parameter to `True` for the underlying LLM model. This feature:

- Provides token usage statistics even when streaming responses
- Is primarily supported by OpenAI, AzureOpenAI, and others. The NIM provider always supports it.
- Allows to safely pass token usage statistics to LLM providers. If the LLM provider you use don't support it, the parameter is ignored.

### Version Requirements

For optimal token usage tracking with streaming, ensure you're using recent versions of LangChain packages:

- `langchain-openai >= 0.1.0` for basic streaming token support (minimum requirement)
- `langchain-openai >= 0.2.0` for enhanced features and stability
- `langchain >= 0.2.14` and `langchain-core >= 0.2.14` for universal token counting support

**Note**: NeMo Guardrails requires `langchain-openai >= 0.1.0` as an optional dependency, which provides basic streaming token usage support. For enhanced features and stability, consider upgrading to `langchain-openai >= 0.2.0` in your environment.

### Accessing Token Usage Information

You can access token usage statistics through the detailed logging capabilities of NeMo Guardrails. Use the `log` generation option to capture comprehensive information about LLM calls, including token usage:

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

For more information about streaming token usage support across different providers, refer to the [LangChain documentation on token usage tracking](https://python.langchain.com/docs/how_to/chat_token_usage_tracking/#streaming). For detailed information about accessing generation logs and token usage, see the [Generation Options](generation-options.md#detailed-logging-information) and [Detailed Logging](../detailed-logging/README.md) documentation.

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
