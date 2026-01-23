---
title:
  page: "NeMo Guardrails Tutorials"
  nav: "Tutorials"
description: "Follow hands-on tutorials to deploy Nemotron Content Safety, Nemotron Topic Control, and Nemotron Jailbreak Detect NIMs."
topics: ["Get Started", "AI Safety"]
tags: ["Tutorial", "Content Safety", "Jailbreak", "Topic Control", "Nemotron", "NIM"]
content:
  type: "Tutorial"
  difficulty: "Beginner"
  audience: ["Developer", "AI Engineer"]
---

# Tutorials

This section contains tutorials that help you get started with the NeMo Guardrails library.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Check Harmful Content
:link: nemotron-safety-guard-deployment
:link-type: doc

Check text inputs and outputs for harmful content using Nemotron Content Safety NIM.
:::

:::{grid-item-card} Content Safety Reasoning
:link: nemotron-content-safety-reasoning-deployment
:link-type: doc

Deploy Nemotron-Content-Safety-Reasoning-4B for customizable content safety with reasoning traces.
:::

:::{grid-item-card} Restrict Topics
:link: nemoguard-topiccontrol-deployment
:link-type: doc

Restrict conversations to allowed topics using Llama 3.1 NemoGuard 8B TopicControl NIM.
:::

:::{grid-item-card} Detect Jailbreak Attempts
:link: nemoguard-jailbreakdetect-deployment
:link-type: doc

Detect and block adversarial prompts and jailbreak attempts using NemoGuard JailbreakDetect NIM.
:::

:::{grid-item-card} Add Multimodal Content Safety
:link: multimodal
:link-type: doc

Add safety checks to images and text using a vision model as LLM-as-a-Judge.
:::

::::

```{toctree}
:hidden:
:maxdepth: 2

Check Harmful Content <nemotron-safety-guard-deployment>
Content Safety Reasoning <nemotron-content-safety-reasoning-deployment>
Restrict Topics <nemoguard-topiccontrol-deployment>
Detect Jailbreak Attempts <nemoguard-jailbreakdetect-deployment>
Add Multimodal Content Safety <multimodal>
```
