---
title:
  page: "How It Works"
  nav: "How It Works"
description: "High level explanation of how Guardrails works."
keywords: ["guardrails architecture", "event-driven runtime", "llmrails", "colang"]
topics: ["generative_ai", "developer_tools"]
tags: ["llms", "security_for_ai", "ai_inference", "nlp"]
content:
  type: concept
  difficulty: technical_advanced
  audience: [engineer, data_scientist]
---

# How It Works

The NeMo Guardrails library acts as an intermediary between application code and LLM requests and responses. Once Guardrails is integrated in an application, all LLM inference requests are first checked by Guardrails to ensure user requests are safe and not malicious. If they are, the request is passed to the LLM for inference. Guardrails also checks the LLM response once it's available, making sure it's appropriate before being passed back to the user.

```{image} ../_static/images/programmable_guardrails.png
:alt: "Programmable Guardrails Flow"
:width: 800px
:align: center
```

Each application can configure its own set of guardrails, depending on the use-case. Guardrails requests can trigger calls to third-party APIs, LLMs fine-tuned to implement Guardrail functionality, or to the Application LLM. Guardrails hides this complexity from clients, orchestrating the workflows behind-the-scenes so applications can focus on their business logic.
