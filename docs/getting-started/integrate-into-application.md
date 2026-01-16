---
title:
  page: "Integrate the NeMo Guardrails Library into Your Application"
  nav: "Integrate"
description: "Add guardrails to existing applications using the Python SDK, LangChain, or HTTP API."
topics: ["Get Started", "AI Safety"]
tags: ["Integration", "Python", "LangChain", "SDK", "API"]
content:
  type: "How-To"
  difficulty: "Beginner"
  audience: ["Developer", "AI Engineer"]
---

# Integrate the NeMo Guardrails Library into Your Application

If you have an existing application, you can integrate NeMo Guardrails into it using the NeMo Guardrails library.

---

## Integrate Guardrails into LLM-based Applications

The NeMo Guardrails library can be integrated into applications in multiple ways:

1. **Python SDK integration**: Add guardrails directly into your Python application.

   ```python
   from nemoguardrails import LLMRails, RailsConfig

   config = RailsConfig.from_path("path/to/config")
   rails = LLMRails(config)

   # Use in your application
   response = rails.generate(messages=[...])
   ```

2. **LangChain integration**: Wrap guardrails around LangChain chains or use chains within guardrails.

   ```python
   from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails

   guardrails = RunnableRails(config)
   chain_with_guardrails = prompt | guardrails | model | output_parser
   ```

   For more information, refer to the [LangChain Integration Guide](../integration/langchain/langchain-integration.md).

3. **HTTP API integration**: Use the guardrails server to add protection to applications in any programming language.

   ```bash
   nemoguardrails server --config path/to/configs
   ```

   For more information, refer to the [Server Guide](../deployment/local-server/index.md).

4. **Docker deployment**: Deploy guardrails as a containerized service.
   For more information, refer to the [Using Docker Guide](../deployment/using-docker.md).

For complete examples and detailed integration patterns, refer to the [examples directory](https://github.com/NVIDIA-NeMo/Guardrails/tree/develop/examples) in the GitHub repository.
