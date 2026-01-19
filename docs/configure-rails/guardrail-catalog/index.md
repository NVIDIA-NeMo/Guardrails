---
title:
  page: "Guardrail Catalog"
  nav: "Guardrail Catalog"
description: "Reference for pre-built guardrails including content safety, jailbreak detection, topic control, PII handling, agentic security, and third party APIs."
topics: ["Configuration", "AI Safety"]
tags: ["Rails", "Content Safety", "Jailbreak", "Security", "YAML"]
content:
  type: "Reference"
  difficulty: "Intermediate"
  audience: ["Developer", "AI Engineer"]
---

# Guardrail Catalog

The NeMo Guardrails library comes with a set of guardrails that you can use out of the box. The following sections provide a comprehensive reference for all the guardrails and their configurations.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Content Safety
:link: content-safety
:link-type: doc

Content safety guardrails help ensure that both user inputs and LLM outputs are safe and appropriate.
:::

:::{grid-item-card} Jailbreak Protection
:link: jailbreak-protection
:link-type: doc

Jailbreak protection helps prevent adversarial attempts from bypassing safety measures and manipulating the LLM into generating harmful or unwanted content.
:::

:::{grid-item-card} Topic Control
:link: topic-control
:link-type: doc

Topic control guardrails ensure that conversations stay within predefined subject boundaries and prevent the LLM from engaging in off-topic discussions.
:::

:::{grid-item-card} PII Detection
:link: pii-detection
:link-type: doc

Personally Identifiable Information (PII) detection helps protect user privacy by detecting and masking sensitive data in user inputs, LLM outputs, and retrieved content.
:::

:::{grid-item-card} Agentic Security
:link: agentic-security
:link-type: doc

Agentic security provides specialized guardrails for LLM-based agents that use tools and interact with external systems.
:::

:::{grid-item-card} Hallucinations & Fact-checking
:link: fact-checking
:link-type: doc

Fact-checking guardrails can help ensure that LLM output is well grounded in evidence and reduce so-called hallucinations or false claims.
:::

:::{grid-item-card} LLM Self-check
:link: self-check
:link-type: doc

By prompting the LLM, self-check rails can test input or output against a simple safety policy.
:::

:::{grid-item-card} Third-Party APIs
:link: third-party
:link-type: doc

Third-party APIs connect with managed services for a wide variety of guardrail use cases.
Combine techniques across the guardrail ecosystem for a best-of-breed approach.
:::

::::
---
