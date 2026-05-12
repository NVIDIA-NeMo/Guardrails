# HuggingFace Endpoint

> **Framework requirement.** This example uses `engine: huggingface_endpoint`, which is served by the LangChain framework only.
> Set `NEMOGUARDRAILS_LLM_FRAMEWORK=langchain` and install `pip install langchain langchain-community` before running.
> The `parameters.model_kwargs` block in `config.yml` is also LangChain-specific and is rejected by the default framework.

This configuration uses the HuggingFace [Inference Endpoints](https://huggingface.co/docs/inference-endpoints/index).

First, follow the [endpoint creation guide](https://huggingface.co/docs/inference-endpoints/guides/create_endpoint). Then, update the `endpoint_url` key in the `config.yml` file.

**Disclaimer**: The `dolly-v2-3b` LLM model has only been tested on basic use cases, e.g., greetings and recognizing specific questions. On more complex queries, this model may not work correctly. Thorough testing and optimizations are needed before considering a production deployment.

## DefaultFramework alternative for OpenAI-compatible endpoints

Hugging Face Inference Endpoints can be deployed with [Text Generation Inference](https://huggingface.co/docs/text-generation-inference/index) (TGI), which optionally exposes an OpenAI-compatible `/v1/chat/completions` route. If your endpoint serves that route, you can stay on the DefaultFramework with `engine: openai` and the canonical `base_url` parameter, no LangChain required:

```yaml
models:
  - type: main
    engine: openai
    model: tgi
    parameters:
      base_url: https://xxx.aws.endpoints.huggingface.cloud/v1
```

For the standard `text-generation-inference` schema (no `/v1/chat/completions`), use the LangChain framework as described above.
