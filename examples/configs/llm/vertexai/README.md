# Vertex AI Example

> **Framework requirement.** This example uses `engine: vertexai`, which is served by the LangChain framework only.
> Set `NEMOGUARDRAILS_LLM_FRAMEWORK=langchain` and install `pip install langchain langchain-google-vertexai` before running.

This guardrails configuration is a basic example using the Vertex AI API, and it can be adapted as needed.

Calling Vertex AI APIs requires [initial Google Cloud setup](../../../../docs/user-guides/advanced/vertexai-setup.md). On top of NeMo Guardrails, install:

```
pip install "google-cloud-aiplatform>=1.38.0"
pip install langchain-google-vertexai
```

The example points at `gemini-1.0-pro` for historical continuity with this configuration; you should update the `model:` field in `config.yml` to a currently supported Gemini model for your project (for example, `gemini-2.5-flash` or `gemini-2.5-pro`). Vertex AI's model catalog rotates frequently, and older model IDs may stop accepting requests.

**Disclaimer**: This example has only been tested on basic use cases. On more complex queries, behavior depends on the Vertex AI model you choose and may not match the assertions in the guardrails flows. Thorough testing and tuning are required before any production deployment, especially for the self-check flows whose prompts assume a particular response style. Provider-side content-moderation behavior also varies across Vertex AI model generations and can surface as transient errors; production code paths should handle those gracefully.
