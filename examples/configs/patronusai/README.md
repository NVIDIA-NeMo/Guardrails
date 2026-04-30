# Patronus AI Examples

> `lynx_config.yml` uses NeMo Guardrails' DefaultFramework with vLLM's OpenAI-compatible endpoint. No LangChain dependency is required: the `engine: openai` plus `parameters.base_url` form routes through the built-in OpenAI-compatible HTTP client.

This folder contains two example configurations:

- `lynx_config.yml` - Self-hosted [Patronus Lynx](https://huggingface.co/PatronusAI/Patronus-Lynx-70B-Instruct) hallucination check served via vLLM.
- `evaluate_api_config.yml` - Patronus Evaluate API integration.

When self-hosted vLLM does not enforce authentication, set `parameters.api_key` to any non-empty placeholder such as `EMPTY`. If your deployment requires a real token, replace it with that value or load it through `api_key_env_var`.

To switch to the 8B variant, change `model:` to `PatronusAI/Patronus-Lynx-8B-Instruct` in `lynx_config.yml`.
