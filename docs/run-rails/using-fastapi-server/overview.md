---
title:
  page: "Overview of the FastAPI Server"
  nav: "Overview"
description: "The FastAPI server is a tool for running guardrails in a secure, isolated environment."
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

1. Start the server using the nemoguardrails CLI:

   ```bash
   nemoguardrails server --config ./my-config
   ```

2. Send a request to the server:

   ```bash
   curl -X POST http://localhost:8000/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{
       "config_id": "my-config",
       "messages": [{"role": "user", "content": "Hello!"}]
     }'
   ```

3. View the Chat UI by opening `http://localhost:8000` in your browser.

## Related Topics

- [Server Endpoints Reference](../../reference/api-server-endpoints/index.md)
- [Local Server Deployment](../../deployment/local-server/index.md)
