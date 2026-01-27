---
title:
  page: "Other Configurations"
  nav: "Other Configurations"
description: "Additional configuration topics including knowledge base setup and embedding search providers."
keywords: ["nemo guardrails other configurations", "knowledge base", "embedding search", "RAG configuration"]
topics: ["generative_ai", "developer_tools"]
tags: ["llms", "ai_inference", "configuration"]
content:
  type: reference
  difficulty: technical_intermediate
  audience: ["engineer"]
---

# Other Configurations

This section provides additional configuration topics that are not covered in the previous sections of the configuration guide.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Knowledge Base
:link: knowledge-base
:link-type: doc

Configure the knowledge base folder for RAG-based responses using markdown documents.
+++
{bdg-secondary}`How To`
:::

:::{grid-item-card} Embedding Search Providers
:link: embedding-search-providers
:link-type: doc

Configure embedding search providers for vector similarity search using FastEmbed, OpenAI, or custom implementations.
+++
{bdg-secondary}`Reference`
:::

::::

```{toctree}
:hidden:
:maxdepth: 2

Knowledge Base <knowledge-base>
Embedding Search Providers <embedding-search-providers>
```
