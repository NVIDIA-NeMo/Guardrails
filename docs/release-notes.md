---
tocdepth: 2
---
<!--
  SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Release Notes

The following sections summarize and highlight the changes for each release.
For a complete record of changes in a release, refer to the
[CHANGELOG.md](https://github.com/NVIDIA/NeMo-Guardrails/blob/develop/CHANGELOG.md) in the GitHub repository.

---

(v0-19-0)=

## 0.19.0

(v0-19-0-features)=

### Key Features

- Added support for LangChain 1.x, including the content blocks API for reasoning traces and tool calls.

(v0-19-0-fixed-issues)=

### Fixed Issues

- Fixed TypeError in Colang 2.x chat caused by incorrect type conversion between `State` and `dict`.
- Fixed async streaming support for the ChatNVIDIA provider patch by adding a new `async_stream_decorator`.

---

(v0-18-0)=

## 0.18.0

(v0-18-0-features)=

### Key Features

- In-memory caching of guardrail model calls for reduced latency and cost savings.
  NeMo Guardrails now supports per-model caching of guardrail responses using an LFU (Least Frequently Used) cache.
  This feature is particularly effective for safety models such as NVIDIA NemoGuard [Content Safety](https://build.nvidia.com/nvidia/llama-3_1-nemoguard-8b-content-safety), [Topic Control](https://build.nvidia.com/nvidia/llama-3_1-nemoguard-8b-topic-control), and [Jailbreak Detection](https://build.nvidia.com/nvidia/nemoguard-jailbreak-detect) where identical inputs are common.
  For more information, refer to [](model-memory-cache).
- NeMo Guardrails extracts the reasoning traces from the LLM response and emits them as `BotThinking` events before the final `BotMessage` event.
  For more information, refer to [](bot-thinking-guardrails).
- New community integration with [Cisco AI Defense](https://www.cisco.com/site/ca/en/products/security/ai-defense/index.html).
- New embedding integrations with Azure OpenAI, Google, and Cohere.

(v0-18-0-fixed-issues)=

### Fixed Issues

- Implemented validation of content safety and topic control guardrail configurations at creation time, providing prompt error reporting if required prompt templates or parameters are missing.

---

(v0-17-0)=

## 0.17.0

(v0-17-0-features)=

### Key Features

- Added support for [integrating with LangGraph and tool calling](./user-guides/langchain/langgraph-integration.md).
  This integration enables building safe and controlled multi-agent workflows.
  LangGraph enables you to create sophisticated agent architectures with state management, conditional routing, and tool calling, while NeMo Guardrails provides the safety layer to ensure responsible AI behavior.
  You can intercept, store, and forward LLM tool invocations with backward compatibility.

- Enhanced support for [integrating with LangChain `RunnableRails`](./user-guides/langchain/runnable-rails.md).
  This release supports the LangChain Runnable interface, such as synchronous and asynchronous operations, streaming, and batch processing while preserving metadata during LangChain operation.
  This enhancement enables NeMo Guardrails to plug into LangChain pipelines seamlessly.

- Trend Micro contributed support for Trend Micro Vision One AI Application Security AI Guard.
  Refer to [configuration documentation](./user-guides/community/trend-micro.md) for more information.

(v0-17-0-other-changes)=

### Other Changes

- Improved URL handling for connecting to NemoGuard JailbreakDetect NIM.
  Guardrails now tolerates the URL for `rails.config.jailbreak_detection.nim_base_url` ending with or without a trailing slash.
  Refer to [](./user-guides/advanced/nemoguard-jailbreakdetect-deployment.md) for information about using the NIM.

(v0-16-0)=

## 0.16.0

(v0-16-0-features)=

### Key Features

- Enhanced tracing system with [OpenTelemetry semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/). To learn more, refer to [](tracing). For usage examples, refer to the following notebooks
  - [Tracing Guardrails Quickstart](https://github.com/NVIDIA/NeMo-Guardrails/tree/develop/docs/getting-started/8-tracing/1_tracing_quickstart.ipynb)
  - [Tracing Guardrails with Jaeger](https://github.com/NVIDIA/NeMo-Guardrails/tree/develop/docs/getting-started/8-tracing/2_tracing_with_jaeger.ipynb)
- Community integration with [GuardrailsAI](https://www.guardrailsai.com/) and [Pangea AI Guard](https://pangea.cloud/services/ai-guard).

(v0-16-0-other-changes)=

### Other Changes

- Added documentation about using KV cache reuse for LLM-based NemoGuard NIMs. By using KV cache reuse, you can improve the performance of LLM-based NemoGuard NIMs where the system prompt is the same for all calls up to the point where user query and LLM response are injected. To learn more, refer to [](kv-cache-reuse).

(v0-15-0)=

## 0.15.0

(v0-15-0-features)=

### Key Features

- Added parallel execution for input and output rails. To learn more, refer to [](parallel-rails).
- Implemented a new way of configuring tracing. You can now use the OpenTelemetry SDK and the OpenTelemetry Protocol (OTLP) exporter while configuring the NeMo Guardrails clients in your application code directly. To learn more, refer to the [basic tracing configuration guide](tracing-configuration) and the [advanced tracing configuration guide](tracing).
- Updated the streaming capability of output rails to support parallel execution.
- Added support for external async token generators. To learn more, refer to the [](external-async-token-generators) section.

### Breaking Changes

With the new tracing configuration, the following old configuration for tracing in `config.yml` is no longer supported.

```yaml
#  No longer supported
tracing:
  enabled: true
  adapters:
    - name: OpenTelemetry
      service_name: "my-service"
      exporter: "console"
```

To find the new way of configuring tracing, refer to [](tracing-configuration).

### Deprecated Functions

- `register_otel_exporter()` is deprecated and will be removed in v0.16.0. Configure exporters directly in your application instead.

(v0-14-1)=

## 0.14.1

(v0-14-1-features)=

### Features

- Added direct API key configuration support for jailbreak detection. This change adds a new optional field `api_key` to the `JailbreakDetectionConfig` Pydantic model. This allows to provide an API Key in a `RailsConfig` object or YAML file, for use in Jailbreak NIM calls. Prior to this change, the `api_key_env_var` field used an environment variable (for example `NVIDIA_API_KEY`) to get the API Key for the Jailbreak NIM.

(v0-14-1-fixed-issues)=

### Fixed Issues

- Fixed lazy loading of jailbreak detection dependencies. Before, jailbreak detection imported unnecessary dependencies when using NIM, which led to installation of those dependencies even when not using the local model-based jailbreak detection.
- Fixed constructor LLM configuration to properly load other config models.
- Fixed content safety policy violations handling by replacing try-except with iterable unpacking.
- Fixed numpy version compatibility by pinning to version 1.23.5 for scikit-learn compatibility.
- Fixed iterable unpacking compatibility in content safety output parsers.

(v0-14-0)=

## 0.14.0

(v0-14-0-features)=

### Features

- Added support for Python 3.13.
- Enhanced working with advanced reasoning models.
  - Added support for the NVIDIA Nemotron family of advanced reasoning models, such as Llama 3.1 Nemotron Ultra 253B V1.
  - Added the `rails.output.apply_to_reasoning_traces` field.
    When this field is `True`, output rails are applied to the reasoning traces and the model output.
    For more information, refer to [](./user-guides/configuration-guide.md#using-llms-with-reasoning-traces).
  - The `reasoning_config.remove_thinking_traces` field is deprecated and replaced by the `reasoning_config.remove_reasoning_traces` field that has the same purpose and subfields.
  - Previously, if `remove_thinking_traces` was set to `True`, the reasoning traces were omitted from the final response presented to the end user.
    In this release, `remove_reasoning_traces` controls whether reasoning traces are removed from internal tasks and has no effect on the final response presented to the user.
  - Using advanced reasoning models with dialog rails is not supported.
- Simplified and broadened support for chat model providers from LangChain and
  LangChain Community chat model providers.
  You must use `langchain` version `0.2.14` or higher and `langchain-community` version `0.2.5` or higher.
  For information about using model providers, refer to [](./user-guides/configuration-guide.md#the-llm-model).
- Added support for code injection detection.
  For more information, refer to [](./user-guides/guardrails-library.md#injection-detection).
- Enhanced the `nemoguardrails` CLI with a `find-providers` argument to list chat and text completion providers.
  For more information, refer to [](./user-guides/cli.md#providers).

(v0-14-0-breaking-changes)=

### Breaking Changes

- Removed support for the NeMo LLM Service, `nemollm`.
  This provider reached end-of-life on February 5, 2025.
- The `HuggingFacePipelineCompatible` provider is refactored.
  Previously, the class was available from the `nemoguardrails.llm.providers` package.
  In this release, the class is moved to the `nemoguardrails.llm.providers.huggingface` package.

(v0-14-0-fixed-issues)=

### Fixed Issues

- Fixed an issue when tracing is enabled.
  Previously, the response was replaced when tracing is enabled and could cause a crash or exception.
  In this release, the response is not modified when tracing is enabled.
  For more information, refer to <pr:1103>.

- Fixed an issue with the self check output flow.
  Previously, the `stop` instruction was not executed when `enable_rails_exceptions` was enabled.
  In this release, the `stop` instruction correctly regardless of the `enable_rails_execptions` value.
  For more information, refer to <pr:1126>.

- Previously, the model specification in the guardrails configuration file, `config.yml`, did not validate the model name.
  In this release you must specify the model name in the `model` top-level field or as `model` or `model_name` in
  the parameters field.
  For more information, refer to <pr:1084>.
