# DeepSeek R1 Example

> This example uses NeMo Guardrails' DefaultFramework. DeepSeek's hosted API at `https://api.deepseek.com/v1` is OpenAI-compatible, so the `engine: openai` plus `parameters.base_url` form routes through the built-in OpenAI-compatible HTTP client. No LangChain dependency is required.

This configuration shows how to call DeepSeek R1 (`deepseek-reasoner`) and the `reasoning_config` block used to strip `<think>...</think>` traces emitted by reasoning models.

Set `DEEPSEEK_API_KEY` in your environment before running:

```bash
export DEEPSEEK_API_KEY=sk-...
```

## NVIDIA NIM alternative

DeepSeek R1 weights are also available through NVIDIA NIM. To use the NIM hosted endpoint, replace the model entry with the `nim` engine and the canonical NIM model id, keeping the same `reasoning_config`:

```yaml
models:
  - type: main
    engine: nim
    model: deepseek-ai/deepseek-r1
    reasoning_config:
      remove_reasoning_traces: True
      start_token: "<think>"
      end_token: "</think>"
```

## LangChain fallback

If you prefer to use LangChain's `langchain-deepseek` adapter (for example to take advantage of LangChain-specific features), set `NEMOGUARDRAILS_LLM_FRAMEWORK=langchain`, install `pip install langchain langchain-deepseek`, and use `engine: deepseek` in `config.yml`. For new deployments, the DefaultFramework path above is recommended.
