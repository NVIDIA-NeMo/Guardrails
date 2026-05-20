---
title:
  page: "Release Notes"
  nav: "Release Notes"
description: "Review new features, breaking changes, and fixed issues for each release."
keywords: ["nemo guardrails changelog", "release notes", "version history"]
topics: ["generative_ai"]
tags: ["llms", "security_for_ai"]
content:
  type: reference
  difficulty: technical_beginner
  audience: [engineer, data_scientist, it_operations]
tocdepth: 2
---
<!--
  SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Release Notes

The following sections summarize and highlight the changes for each release.
For a complete record of changes in a release, refer to the
[CHANGELOG.md](https://github.com/NVIDIA-NeMo/Guardrails/blob/develop/CHANGELOG.md) in the GitHub repository.

---

(v0-22-0)=

## 0.22.0

(v0-22-0-features)=

### Key Features

- LangChain is now optional. `pip install nemoguardrails` no longer pulls
  LangChain or any provider-specific `langchain-*` packages. The NVIDIA NeMo
  Guardrails library ships with a built-in client that talks to
  OpenAI-compatible endpoints directly over `httpx`. Engines whose API isn't
  OpenAI-compatible (Anthropic, Cohere, Vertex AI, Google Generative AI,
  in-process Hugging Face, TensorRT-LLM, and others) keep working through
  LangChain when you opt in with `NEMOGUARDRAILS_LLM_FRAMEWORK=langchain` and
  install the matching provider package. Most 0.21 configurations keep working
  unchanged; some shapes need a YAML rewrite. For recipes, see
  [Migrating to 0.22](../migration/0.22.md), the
  [Supported LLMs](./supported-llms.md) matrix, and
  [Model Configuration](../configure-rails/yaml-schema/model-configuration.md)
  ([#1854](https://github.com/NVIDIA-NeMo/Guardrails/pull/1854),
  [#1855](https://github.com/NVIDIA-NeMo/Guardrails/pull/1855),
  [#1856](https://github.com/NVIDIA-NeMo/Guardrails/pull/1856),
  [#1858](https://github.com/NVIDIA-NeMo/Guardrails/pull/1858),
  [#1881](https://github.com/NVIDIA-NeMo/Guardrails/pull/1881)).

- OpenAI-compatible service support is improved in the default framework.
  The default framework now supports OpenAI-compatible providers directly,
  includes native Azure OpenAI support through `engine: azure` and
  `engine: azure_openai`, and documents how to migrate provider-specific
  LangChain parameters to the new `base_url`-based configuration shape. For
  more information, refer to
  [Migrating to 0.22](../migration/0.22.md),
  [Model Configuration](../configure-rails/yaml-schema/model-configuration.md),
  [Configuration Reference](../configure-rails/configuration-reference.md), and
  [Using Docker](../deployment/using-docker.md)
  ([#1854](https://github.com/NVIDIA-NeMo/Guardrails/pull/1854),
  [#1855](https://github.com/NVIDIA-NeMo/Guardrails/pull/1855),
  [#1857](https://github.com/NVIDIA-NeMo/Guardrails/pull/1857),
  [#1858](https://github.com/NVIDIA-NeMo/Guardrails/pull/1858),
  [#1881](https://github.com/NVIDIA-NeMo/Guardrails/pull/1881),
  [#1896](https://github.com/NVIDIA-NeMo/Guardrails/pull/1896)).

- `IORails` observability is expanded with OpenTelemetry logging, tracing, and
  metrics guidance. The documentation now covers OTLP setup, Prometheus client
  installation, request-level and token-level metrics, and the recommended
  `Guardrails` entry point for the optimized input and output rails engine. For
  more information, refer to
  [Observability](../observability/index.md),
  [OpenTelemetry Logs](../observability/tracing/opentelemetry-logs.md),
  [OpenTelemetry Tracing](../observability/tracing/opentelemetry-integration.md),
  [OpenTelemetry Metrics](../observability/metrics/opentelemetry-integration.md),
  [Enable Metrics](../observability/metrics/enable-metrics.md), and the
  [Metrics Reference](../observability/metrics/reference.md)
  ([#1807](https://github.com/NVIDIA-NeMo/Guardrails/pull/1807),
  [#1864](https://github.com/NVIDIA-NeMo/Guardrails/pull/1864),
  [#1884](https://github.com/NVIDIA-NeMo/Guardrails/pull/1884),
  [#1892](https://github.com/NVIDIA-NeMo/Guardrails/pull/1892),
  [#1893](https://github.com/NVIDIA-NeMo/Guardrails/pull/1893),
  [#1894](https://github.com/NVIDIA-NeMo/Guardrails/pull/1894)).

- Anonymous usage reporting is documented with clear privacy boundaries and
  opt-out controls. The telemetry reference explains what fields are collected,
  what data is excluded, how local audit files work, and how to opt out with
  `NEMO_GUARDRAILS_NO_USAGE_STATS=1`, `DO_NOT_TRACK=1`, or the
  `~/.config/nemoguardrails/do_not_track` file. For more information, refer to
  [Telemetry](../telemetry.md)
  ([#1891](https://github.com/NVIDIA-NeMo/Guardrails/pull/1891)).

- The GLiNER PII connector documentation and notebook are updated for the new
  GLiNER PII NIM. The examples cover both remote and local deployment modes
  and API key configuration for the connector. For more information, refer to
  [GLiNER](../configure-rails/guardrail-catalog/community/gliner.md) and
  [PII Detection](../configure-rails/guardrail-catalog/pii-detection.md)
  ([#1845](https://github.com/NVIDIA-NeMo/Guardrails/pull/1845)).

- Public extension points for LLM integration. Two new protocols, `LLMModel`
  and `LLMFramework` in `nemoguardrails.types`, let you plug in a custom
  backend or a whole alternative framework without touching internals. For more
  information, refer to
  [Custom LLM Models](../configure-rails/custom-initialization/custom-llm-model.md)
  and
  [Custom LLM Frameworks](../configure-rails/custom-initialization/custom-llm-framework.md)
  ([#1857](https://github.com/NVIDIA-NeMo/Guardrails/pull/1857),
  [#1882](https://github.com/NVIDIA-NeMo/Guardrails/pull/1882)).

- Public testing surface. The `nemoguardrails.testing` module exposes
  `FakeLLMModel`, `TestChat`, and pytest fixtures for writing tests against a
  guardrails configuration without calling a real model.

(v0-22-0-doc-and-behavior-fixes)=

### Documentation and Behavior Fixes

- Fixed the example query and expected output in the Guardrails Agent
  Middleware integration guide so the example matches the configured blocked
  response behavior. For more information, refer to
  [Guardrails Agent Middleware](../integration/langchain/agent-middleware.md)
  ([#1784](https://github.com/NVIDIA-NeMo/Guardrails/pull/1784)).
- A warning about a missing main LLM is now emitted only when generation is
  actually attempted and the generation path needs the main LLM. Check-only
  configurations no longer emit the warning during initialization. For more
  information, refer to
  [Check Messages](../run-rails/using-python-apis/check-messages.md)
  ([#1813](https://github.com/NVIDIA-NeMo/Guardrails/pull/1813)).
- Fixed issues in the [Colang 1.0 Hello World tutorial](../configure-rails/colang/colang-1/tutorials/1-hello-world/README.md) and companion notebook.
  ([#1834](https://github.com/NVIDIA-NeMo/Guardrails/pull/1834)).

---

## Previous Release Notes

- [0.21.0](https://docs.nvidia.com/nemo/guardrails/0.21.0/release-notes.html)
- [0.20.0](https://docs.nvidia.com/nemo/guardrails/0.20.0/release-notes.html)
- [0.19.0](https://docs.nvidia.com/nemo/guardrails/0.19.0/release-notes.html)
- [0.18.0](https://docs.nvidia.com/nemo/guardrails/0.18.0/release-notes.html)
- [0.17.0](https://docs.nvidia.com/nemo/guardrails/0.17.0/release-notes.html)
- [0.16.0](https://docs.nvidia.com/nemo/guardrails/0.16.0/release-notes.html)
- [0.15.0](https://docs.nvidia.com/nemo/guardrails/0.15.0/release-notes.html)
- [0.14.1](https://docs.nvidia.com/nemo/guardrails/0.14.1/release-notes.html)
- [0.14.0](https://docs.nvidia.com/nemo/guardrails/0.14.0/release-notes.html)
