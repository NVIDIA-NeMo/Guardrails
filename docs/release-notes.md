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

(v0-14-1)=

## 0.14.1

(v0-14-1-features)=

### Features

-

(v0-14-1-fixed-issues)=

### Fixed Issues

-

(v0-14-1-known-issues)=

### Known Issues

-

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
