---
title:
  page: "NeMo Guardrails Tutorials"
  nav: "Tutorials"
description: "Follow hands-on tutorials to deploy content safety, topic control, and jailbreak detection."
topics: ["Get Started", "AI Safety"]
tags: ["Tutorial", "Content Safety", "Jailbreak", "Topic Control"]
content:
  type: "Tutorial"
  difficulty: "Beginner"
  audience: ["Developer", "AI Engineer"]
---

# Tutorials

This section contains tutorials that help you get started with the NeMo Guardrails library.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Content Safety
:link: nemotron-safety-guard-deployment
:link-type: doc

Deploy Nemotron Safety Guard to detect harmful content in multilingual text inputs and outputs.
:::

:::{grid-item-card} Topic Control
:link: nemoguard-topiccontrol-deployment
:link-type: doc

Deploy NeMo Topic Control NIM to restrict conversations to allowed topics.
:::

:::{grid-item-card} Jailbreak Detection
:link: nemoguard-jailbreakdetect-deployment
:link-type: doc

Deploy NeMo Jailbreak Detect NIM to block adversarial prompts and jailbreak attempts.
:::

:::{grid-item-card} Multimodal
:link: multimodal
:link-type: doc

Add safety checks to images and text using vision models as LLM-as-a-judge.
:::

::::

```{toctree}
:hidden:
:maxdepth: 2

Content Safety <nemotron-safety-guard-deployment>
Topic Control <nemoguard-topiccontrol-deployment>
Jailbreak Detection <nemoguard-jailbreakdetect-deployment>
Multimodal Data <multimodal>
```
