---
title:
  page: "About Configuring Guardrails"
  nav: "About Configuring Guardrails"
description: "Configure YAML files, Colang flows, custom actions, and other components to control LLM behavior."
keywords: ["nemo guardrails configuration", "guardrails setup", "config.yml", "Colang flows", "custom actions"]
topics: ["generative_ai", "cybersecurity"]
tags: ["llms", "security_for_ai", "ai_inference"]
content:
  type: concept
  difficulty: technical_beginner
  audience: ["engineer"]
---

# About Configuring Guardrails

This section explains how to configure your guardrails system, from defining LLM models and guardrail flows in YAML to implementing advanced features like Colang flows and custom actions.

---

## Before You Begin with Configuring Guardrails

Before diving into configuring guardrails, ensure you have the required components ready and understand the overall structure of the guardrails system.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Prerequisites
:link: before-configuration
:link-type: doc

Prepare LLM endpoints, NemoGuard NIMs, and knowledge base documents before configuration.
+++
{bdg-secondary}`Get Started`
:::

:::{grid-item-card} Overview
:link: overview
:link-type: doc

Learn to write config.yml, Colang flows, and custom actions for guardrails.
+++
{bdg-secondary}`Concept`
:::

::::

---

## Core Configuration

Configure the essential components of your guardrails system.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Configuring YAML File
:link: yaml-schema/index
:link-type: doc

Define models, guardrails, prompts, and tracing settings in the config.yml file.
+++
{bdg-secondary}`Reference`
:::

:::{grid-item-card} YAML Schema Reference
:link: configuration-reference
:link-type: doc

Reference for all config.yml options including models, rails, prompts, and advanced settings.
+++
{bdg-secondary}`Reference`
:::

:::{grid-item-card} Guardrail Catalog
:link: guardrail-catalog
:link-type: doc

Reference for pre-built guardrails including content safety, jailbreak detection, and PII handling.
+++
{bdg-secondary}`Reference`
:::

:::{grid-item-card} Colang
:link: colang/index
:link-type: doc

Learn Colang, the event-driven language for defining guardrails flows and bot behavior.
+++
{bdg-secondary}`Concept`
:::

::::

---

## Advanced Configuration

Optional configurations for extending and optimizing your guardrails system.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Custom Actions
:link: actions/index
:link-type: doc

Create Python actions to extend guardrails with external APIs and validation logic.
+++
{bdg-secondary}`How To`
:::

:::{grid-item-card} Custom Initialization
:link: custom-initialization/index
:link-type: doc

Use config.py to register custom LLM providers, embedding providers, and shared resources at startup.
+++
{bdg-secondary}`How To`
:::

:::{grid-item-card} Other Configurations
:link: other-configurations/index
:link-type: doc

Additional configuration topics including knowledge base setup and embedding search providers.
+++
{bdg-secondary}`Reference`
:::

:::{grid-item-card} Caching
:link: caching/index
:link-type: doc

Configure in-memory caching for LLM calls and KV cache reuse to improve performance and reduce latency.
+++
{bdg-secondary}`How To`
:::

:::{grid-item-card} Exceptions
:link: exceptions
:link-type: doc

Raise and handle exceptions in guardrails flows to control error behavior and custom responses.
+++
{bdg-secondary}`Reference`
:::
