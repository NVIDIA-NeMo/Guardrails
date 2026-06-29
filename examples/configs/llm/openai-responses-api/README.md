# OpenAI Responses API Example

> This example requires NeMo Guardrails' **LangChain framework**. The Responses API is **not** supported by the built-in DefaultFramework, whose OpenAI-compatible client only calls `/v1/chat/completions`. The `use_responses_api` parameter is honored only by `langchain_openai.ChatOpenAI`.

This configuration shows how to route an OpenAI model through the [Responses API](https://platform.openai.com/docs/api-reference/responses) (`/v1/responses`) instead of Chat Completions, by setting `use_responses_api: true` in the model's `parameters` block.

## Requirements

Set the framework and install the LangChain packages:

```bash
export NEMOGUARDRAILS_LLM_FRAMEWORK=langchain
pip install langchain langchain-openai
export OPENAI_API_KEY=sk-...
```

Without `NEMOGUARDRAILS_LLM_FRAMEWORK=langchain`, the `use_responses_api` parameter is silently forwarded as an unknown field to `/v1/chat/completions` and has no effect (OpenAI rejects or ignores it) — the call does **not** switch to the Responses API.

## Self-hosted models (vLLM and other OpenAI-compatible servers)

Some inference servers — for example [vLLM](https://docs.vllm.ai/) — expose an OpenAI-compatible `/v1/responses` endpoint in addition to `/v1/chat/completions`. To target such a deployment, keep `engine: openai`, set `use_responses_api: true`, and point `base_url` at your server:

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

This still requires `NEMOGUARDRAILS_LLM_FRAMEWORK=langchain`, because the `use_responses_api` switch lives in `langchain_openai.ChatOpenAI`. Confirm your server actually implements `/v1/responses` before enabling the flag; not all OpenAI-compatible servers do.

## Tool calling (passthrough only)

Function (custom) tool calls work over the Responses API in both streaming and non-streaming modes: the assistant's tool calls are surfaced on `LLMResponse.tool_calls` / `LLMResponseChunk.delta_tool_calls`, and `finish_reason` is reported as `tool_calls`.

Built-in/hosted Responses-API tools (`web_search`, `file_search`, `code_interpreter`, `computer_use`, MCP, image generation) are **not** surfaced as tool calls. They are returned by the API as response output items rather than function calls, so the adapter does not expose them. Only function-tool passthrough is supported.
