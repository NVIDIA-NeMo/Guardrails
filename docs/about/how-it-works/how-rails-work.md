---
title: How Guardrails Work
description: Learn how the NeMo Guardrails toolkit applies guardrails at multiple stages of the LLM interaction.
---

# How Guardrails Work

The NeMo Guardrails toolkit applies guardrails at multiple stages of the LLM interaction.

| Stage | Rail Type | Common Use Cases |
|-------|-----------|------------------|
| **Before LLM** | Input rails | Content safety, jailbreak detection, topic control, PII masking |
| **After LLM** | Output rails | Response filtering, fact checking, sensitive data removal |
| **RAG pipeline** | Retrieval rails | Document filtering, chunk validation |
| **Tool calls** | Execution rails | Action input/output validation |
| **Conversation** | Dialog rails | Flow control, guided conversations |

```{image} ../../_static/images/programmable_guardrails_flow.png
:alt: "Programmable Guardrails Flow"
:width: 800px
:align: center
```
