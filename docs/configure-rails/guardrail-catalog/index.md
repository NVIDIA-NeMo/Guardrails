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

The NeMo Guardrails library ships with a catalog of pre-built guardrails that you can activate out of the box. These guardrails span the most common safety and security concerns in LLM-powered applications from blocking harmful content and detecting jailbreak attempts to masking personally identifiable information and grounding responses in evidence.

Each guardrail is implemented as a configurable rail flow that you add to the `input`, `output`, or `retrieval` section of your `config.yml`. You can use NVIDIA-trained safety models, open-source community models, LLM self-check prompts, or third-party managed APIs, and combine multiple approaches for defense in depth.

Browse the catalog below to find the guardrail that fits your use case.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Content Safety
:link: content-safety
:link-type: doc

Reference for pre-built content safety guardrails for protecting against violence, criminal activity, hate speech, sexually explicit content, and similar areas.
+++
{bdg-secondary}`Reference`
:::

:::{grid-item-card} Jailbreak Protection
:link: jailbreak-protection
:link-type: doc

Reference for jailbreak protection guardrails that help prevent adversarial attempts from bypassing safety measures.
+++
{bdg-secondary}`Reference`
:::

:::{grid-item-card} Topic Control
:link: topic-control
:link-type: doc

Reference for topic control guardrails that ensure conversations stay within predefined subject boundaries.
+++
{bdg-secondary}`Reference`
:::

:::{grid-item-card} PII Detection
:link: pii-detection
:link-type: doc

Reference for PII detection guardrails that protect user privacy by detecting and masking sensitive data.
+++
{bdg-secondary}`Reference`
:::

:::{grid-item-card} Agentic Security
:link: agentic-security
:link-type: doc

Reference for agentic security guardrails that protect LLM-based agents using tools and interacting with external systems.
+++
{bdg-secondary}`Reference`
:::

:::{grid-item-card} Hallucinations & Fact-Checking
:link: fact-checking
:link-type: doc

Reference for fact-checking and hallucination detection guardrails that ensure LLM output is grounded in evidence.
+++
{bdg-secondary}`Reference`
:::

:::{grid-item-card} LLM Self-Check
:link: self-check
:link-type: doc

Reference for LLM self-checking guardrails that prompt the LLM to perform input checking, output checking, or fact-checking.
+++
{bdg-secondary}`Reference`
:::

:::{grid-item-card} Third-Party APIs
:link: third-party
:link-type: doc

Reference for third-party API integrations that connect with managed services for guardrail use cases.
+++
{bdg-secondary}`Reference`
:::

::::

```{toctree}
:caption: Guardrail Catalog
:name: Guardrail Catalog
:hidden:

Content Safety <content-safety>
Jailbreak Protection <jailbreak-protection>
Topic Control <topic-control>
PII Detection <pii-detection>
Agentic Security <agentic-security>
Hallucinations & Fact-Checking <fact-checking>
LLM Self-Check <self-check>
Third-Party APIs <third-party>
```
