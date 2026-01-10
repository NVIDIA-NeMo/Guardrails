---
title:
  page: "Overview of the FastAPI Server"
  nav: "Overview"
description: "The Fast API server is a tool for running guardrails in a secure, isolated environment."
keywords: ["NeMo Guardrails server", "FastAPI", "REST API", "chat completions", "guardrails HTTP"]
topics: ["generative_ai", "developer_tools"]
tags: ["llms", "ai_inference", "ai_platforms"]
content:
  type: concept
  difficulty: technical_intermediate
  audience: ["data_scientist", "engineer"]
---

# Overview of the NeMo Guardrails Library FastAPI Server

The Guardrails server:

- Loads guardrails configurations at startup.
- Exposes a REST API compatible with OpenAI's chat completions format.
- Includes a built-in Chat UI for testing.
- Supports multiple configurations and combining them per-request.

## Quick Start

1. Install the NeMo Guardrails library:

   ```bash
   pip install nemoguardrails
   ```

   For more information, see [](../../getting-started/installation-guide.md).

2. **Start the server:**

   ```bash
   nemoguardrails server --config ./my-config
   ```

3. **Send a request:**

   ```bash
   curl -X POST http://localhost:8000/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{
       "config_id": "my-config",
       "messages": [{"role": "user", "content": "Hello!"}]
     }'
   ```

4. **View the Chat UI:** Open `http://localhost:8000` in your browser.

## Related Topics

- [Server Endpoints Reference](../../reference/api-server-endpoints/index.md)
- [Local Server Deployment](../../deployment/local-server/index.md)
