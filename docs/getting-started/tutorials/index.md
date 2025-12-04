# Tutorials

This section contains tutorials that help you get started with NeMo Guardrails Toolkit.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Nemotron Safety Guard Deployment
:link: nemotron-safety-guard-deployment
:link-type: doc

Deploy a GPU-accelerated multilingual content safety model using Llama 3.1 Nemotron Safety Guard 8B V3 to detect harmful content in multiple languages.
:::

:::{grid-item-card} Llama 3.1 NemoGuard 8B Topic Control Deployment
:link: nemoguard-topiccontrol-deployment
:link-type: doc

Deploy the TopicControl NIM microservice for low-latency optimized inference and integrate it into your NeMo Guardrails configuration.
:::

:::{grid-item-card} NemoGuard JailbreakDetect Deployment
:link: nemoguard-jailbreakdetect-deployment
:link-type: doc

Deploy the NemoGuard Jailbreak Detection NIM microservice to protect your LLM applications from adversarial jailbreak attempts.
:::

:::{grid-item-card} Multimodal Data with NeMo Guardrails
:link: multimodal
:link-type: doc

Add safety checks to multimodal content including images and text using image reasoning models as LLM-as-a-judge.
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
