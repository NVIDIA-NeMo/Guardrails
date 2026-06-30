# OpenAI Responses API Example

> This example requires the NVIDIA NeMo Guardrails library to use the LangChain framework. The built-in default framework does not support the Responses API because its OpenAI-compatible client only calls `/v1/chat/completions`. The `langchain_openai.ChatOpenAI` class honors the `use_responses_api` parameter.

This configuration shows how to route an OpenAI model through the [Responses API](https://platform.openai.com/docs/api-reference/responses) (`/v1/responses`) instead of Chat Completions. Set `use_responses_api: true` in the model's `parameters` block.

## Requirements

Set the framework and install the LangChain packages:

```bash
export NEMOGUARDRAILS_LLM_FRAMEWORK=langchain
pip install langchain langchain-openai
export OPENAI_API_KEY=sk-...
```

Without `NEMOGUARDRAILS_LLM_FRAMEWORK=langchain`, the library forwards `use_responses_api` as an unknown field to `/v1/chat/completions`. The call does not switch to the Responses API.

## Self-hosted models (vLLM and other OpenAI-compatible servers)

Some inference servers, such as [vLLM](https://docs.vllm.ai/), expose an OpenAI-compatible `/v1/responses` endpoint in addition to `/v1/chat/completions`. To target that deployment, keep `engine: openai`, set `use_responses_api: true`, and point `base_url` at your server:

```yaml
models:
  - type: main
    engine: openai
    model: openai/gpt-oss-20b
    api_key_env_var: ANY_KEY_CAN_BE_USED_HERE
    parameters:
      base_url: http://localhost:8000/v1
      use_responses_api: true
```

This configuration still requires `NEMOGUARDRAILS_LLM_FRAMEWORK=langchain` because `langchain_openai.ChatOpenAI` provides the `use_responses_api` switch. Confirm that your server implements `/v1/responses` before you enable the flag. Not all OpenAI-compatible servers support it.

## Tool calling (passthrough only)

Function tool calls work over the Responses API in streaming and non-streaming modes. The adapter surfaces the assistant's tool calls on `LLMResponse.tool_calls` and `LLMResponseChunk.delta_tool_calls`, and reports `finish_reason` as `tool_calls`.

Built-in Responses API tools, such as `web_search`, `file_search`, `code_interpreter`, `computer_use`, MCP, and image generation, are not surfaced as tool calls. The API returns those tools as response output items, not function calls, so only function-tool passthrough is supported.
