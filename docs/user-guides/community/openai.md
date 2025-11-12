## OpenAI API Compatibility for NeMo Guardrails

NeMo Guardrails provides server-side compatibility with OpenAI API endpoints, enabling applications that use OpenAI clients to seamlessly integrate with NeMo Guardrails for adding guardrails to LLM interactions. Point your OpenAI client to `http://localhost:8000` (or your server URL) and use the standard `/v1/chat/completions` endpoint.

## Feature Support Matrix

The following table outlines which OpenAI API features are currently supported when using NeMo Guardrails:

| Feature | Status | Notes |
| :------ | :----: | :---- |
| **Basic Chat Completion** | ✔ Supported | Full support for standard chat completions with guardrails applied |
| **Streaming Responses** | ✔ Supported | Server-Sent Events (SSE) streaming with `stream=true` |
| **Multimodal Input** | ✖ Unsupported | Support for text and image inputs (vision models) with guardrails but not yet OpenAI compatible  |
| **Function Calling** | ✖ Unsupported | Not yet implemented; guardrails need structured output support |
| **Tools** | ✖ Unsupported | Related to function calling; requires action flow integration |
| **Response Format (JSON Mode)** | ✖ Unsupported | Structured output with guardrails requires additional validation logic |
