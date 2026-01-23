---
title:
  page: "Integrate Guardrails into Your Application"
  nav: "Integrate Guardrails"
description: "Add guardrails to existing applications using the Python SDK, LangChain, or HTTP API."
topics: ["Get Started", "AI Safety"]
tags: ["Integration", "Python", "LangChain", "SDK", "API"]
content:
  type: "How-To"
  difficulty: "Beginner"
  audience: ["Developer", "AI Engineer"]
---

# Integrate Guardrails into Your Application

The NeMo Guardrails library provides the following tools to integrate guardrails into your applications.

- Use the NeMo Guardrails Python SDK to add guardrails directly into your Python application.

   ```python
   from nemoguardrails import LLMRails, RailsConfig

   config = RailsConfig.from_path("path/to/config")
   rails = LLMRails(config)

   # Use in your application
   response = rails.generate(messages=[...])
   ```

- Use the NeMo Guardrails LangChain integration capabilities to wrap guardrails around LangChain chains or use chains within guardrails.

   ```python
   from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails

   guardrails = RunnableRails(config)
   chain_with_guardrails = prompt | guardrails | model | output_parser
   ```

   For more information, refer to the [LangChain Integration Guide](../integration/langchain/langchain-integration.md).

- Integrate the NeMo Guardrails API server into your application to add protection to applications in any programming language.

   ```bash
   nemoguardrails server --config path/to/configs
   ```

   You can then use the API server in your application by sending requests to the server's endpoint.

   ```bash
   curl -X POST http://localhost:8000/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{
       "config_id": "content_safety",
       "messages": [{"role": "user", "content": "Hello!"}]
     }'
   ```

   For more information, refer to [](../run-rails/using-fastapi-server/index.md).

- Use the NeMo Guardrails Docker deployment capabilities to deploy guardrails as a containerized service.
   For more information, refer to [](../deployment/using-docker.md).

For more examples and detailed integration patterns, refer to the [examples directory](https://github.com/NVIDIA-NeMo/Guardrails/tree/develop/examples) in the NeMo Guardrails GitHub repository.
