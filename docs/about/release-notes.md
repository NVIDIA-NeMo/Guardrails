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

(v0-21-0)=

## 0.21.0

(v0-21-0-features)=

### Key Features

- Added the `IORails` class, a new optimized execution engine that runs NemoGuard input and output rails, such as
  content-safety, topic-safety, and jailbreak detection, in parallel. The engine is opt-in:
  set `NEMO_GUARDRAILS_IORAILS_ENGINE=1` to enable it. When enabled, the configuration is
  validated for compatibility and falls back to LLMRails if unsupported flows are detected.
  For more information, refer to [](../configure-rails/yaml-schema/guardrails-configuration/parallel-rails.md#iorails-engine).

- Added the `check_async()` and `check()` methods on `LLMRails` to enable validating messages against input and output rails without triggering full LLM generation.
  Returns a `RailsResult` with `PASSED`, `MODIFIED`, or `BLOCKED` status.
  For more information, refer to [](../run-rails/using-python-apis/check-messages.md).

- The guardrails server now exposes a fully OpenAI-compatible
  REST API. The `/v1/chat/completions` endpoint accepts standard `ChatCompletion` requests with a
  `guardrails` field for config selection. A new `/v1/models` endpoint lists available models from the
  configured provider. The `openai` package is now a required component of the optional `server` extra ([#1623](https://github.com/NVIDIA-NeMo/Guardrails/pull/1623)).
  For more information, refer to [](../run-rails/using-fastapi-server/overview.md).

- Added the `GuardrailsMiddleware` class, a new middleware that integrates with
  LangChain's Agent Middleware protocol, applying input and output rail checks before and after
  every model call in the agent loop. It includes the `InputRailsMiddleware` and `OutputRailsMiddleware`
  convenience subclasses.
  For more information, refer to [](../integration/langchain/agent-middleware.md).

- Added three new community rails:
  [PolicyAI](../configure-rails/guardrail-catalog/community/policyai.md) for policy-based content moderation,
  [CrowdStrike AIDR](../configure-rails/guardrail-catalog/community/crowdstrike-aidr.md) for AI-powered detection and response, and
  [Regex Detection](../configure-rails/guardrail-catalog/community/regex.md) for pattern-based content filtering on input, output, and retrieval.

- Jailbreak detection configuration is now validated at
  create-time. Invalid thresholds and malformed URLs raise errors immediately.
  For more information, refer to [](../configure-rails/guardrail-catalog/jailbreak-protection.md#configuration-validation).

- Embedding indexes are now initialized lazily.
  FastEmbed models are only downloaded when semantic search is needed, reducing startup time for
  configurations that use only input and output rails.

(v0-21-0-breaking-changes)=

### Breaking Changes

- Streaming metadata parameter renamed. The `include_generation_metadata` parameter on
  `LLMRails.stream_async()` and `StreamingHandler` is deprecated in favor of `include_metadata`.
  The `generation_info` field in streaming chunk dicts is renamed to `metadata`.
  The deprecated parameter still works and emits a `DeprecationWarning`.

  ```python
  # Before (deprecated)
  async for chunk in rails.stream_async(messages=messages, include_generation_metadata=True):
      info = chunk["generation_info"]

  # After
  async for chunk in rails.stream_async(messages=messages, include_metadata=True):
      info = chunk["metadata"]
  ```

- `StreamingHandler` no longer inherits from LangChain `AsyncCallbackHandler`.
  Streaming now uses `llm.astream()` with direct `push_chunk()` calls.
  If your code depends on `StreamingHandler` as a LangChain callback, update it to use the
  new `push_chunk()` interface.

- Removed the `stream_usage` parameter. The `stream_usage=True` parameter is no longer
  automatically added to LLM call kwargs. Streaming metadata is now captured through
  `response_metadata` and `usage_metadata` on final chunks.

- Server request and response format changed. The `/v1/chat/completions` endpoint now uses
  OpenAI-compatible request and response schemas. The previous `RequestBody` and `ResponseBody`
  classes are removed. For the new format, refer to
  [](../run-rails/using-fastapi-server/overview.md).

- ChatNVIDIA streaming patch removed. The custom
  `_langchain_nvidia_ai_endpoints_patch.py` module is removed.
  The standard `ChatNVIDIA` from `langchain_nvidia_ai_endpoints` is used directly.

(v0-21-0-bug-fixes)=

### Bug Fixes

- Fixed a naming mismatch where the `generate_next_step` action did not match the
  `generate_next_steps` task enum value, which prevented task-specific LLM configuration
  from working correctly ([#1603](https://github.com/NVIDIA-NeMo/Guardrails/pull/1603)).
- Added the `valid` alias to action results in the GuardrailsAI integration so that
  Colang flows checking `$result["valid"]` work as expected ([#1611](https://github.com/NVIDIA-NeMo/Guardrails/pull/1611)).
- Filtered the `stop` parameter for OpenAI reasoning models (such as GPT-5) that do not
  accept it, preventing `400` errors during dialogue rail execution ([#1653](https://github.com/NVIDIA-NeMo/Guardrails/pull/1653)).
- Fixed GLiNER PII detection to use "bot refuse to respond" instead of
  "bot inform answer unknown", which returned a misleading "I don't know" message ([#1671](https://github.com/NVIDIA-NeMo/Guardrails/pull/1671)).
- Fixed a `TypeError` when `stop=None` is passed to `StreamingHandler` by coercing
  `None` to an empty list ([#1685](https://github.com/NVIDIA-NeMo/Guardrails/pull/1685)).
- Fixed a `TypeError` in `RollingBuffer.format_chunks` when `include_metadata=True` is used
  with output rail streaming enabled. Dict chunks are now normalized to strings at the
  input boundary ([#1687](https://github.com/NVIDIA-NeMo/Guardrails/pull/1687)).
- Fixed `GuardrailsMiddleware` silently dropping content when rails return `MODIFIED` status.
  Input rails now replace the last user message and output rails replace the last AI
  message with the sanitized content ([#1714](https://github.com/NVIDIA-NeMo/Guardrails/pull/1714)).
- Cache hit statistics are now visible in the Stats log line. Cache stats are also
  visible in verbose mode ([#1666](https://github.com/NVIDIA-NeMo/Guardrails/pull/1666), [#1667](https://github.com/NVIDIA-NeMo/Guardrails/pull/1667)).

(v0-21-0-other-changes)=

### Other Changes

- Updated the Fiddler Guardrails API to match the new specification: the `prompt` field is
  renamed to `input`, faithfulness uses strings instead of lists, and a new `fdl_roleplaying`
  category is added ([#1619](https://github.com/NVIDIA-NeMo/Guardrails/pull/1619)).
- Updated the Trend Micro Vision One AI Guard integration from the beta endpoint to the
  officially released GA endpoint. A required `TMV1-Application-Name` header is added and the
  request key is changed from `guard` to `prompt` ([#1546](https://github.com/NVIDIA-NeMo/Guardrails/pull/1546)).
- Added a Locust stress-test benchmark for load testing ([#1629](https://github.com/NVIDIA-NeMo/Guardrails/pull/1629)).
- Removed the `multi_kb` example ([#1673](https://github.com/NVIDIA-NeMo/Guardrails/pull/1673)).
- Removed the AI Virtual Assistant Blueprint notebook ([#1682](https://github.com/NVIDIA-NeMo/Guardrails/pull/1682)).
- Updated the Pangea User-Agent repo URL ([#1610](https://github.com/NVIDIA-NeMo/Guardrails/pull/1610)).
- Updated dependencies for the jailbreak detection Docker container ([#1596](https://github.com/NVIDIA-NeMo/Guardrails/pull/1596)).
- Major documentation revamp with improved structure and navigation.

---

## Previous Release Notes

- [0.20.0](https://docs.nvidia.com/nemo/guardrails/0.20.0/release-notes.html)
- [0.19.0](https://docs.nvidia.com/nemo/guardrails/0.19.0/release-notes.html)
- [0.18.0](https://docs.nvidia.com/nemo/guardrails/0.18.0/release-notes.html)
- [0.17.0](https://docs.nvidia.com/nemo/guardrails/0.17.0/release-notes.html)
- [0.16.0](https://docs.nvidia.com/nemo/guardrails/0.16.0/release-notes.html)
- [0.15.0](https://docs.nvidia.com/nemo/guardrails/0.15.0/release-notes.html)
- [0.14.1](https://docs.nvidia.com/nemo/guardrails/0.14.1/release-notes.html)
- [0.14.0](https://docs.nvidia.com/nemo/guardrails/0.14.0/release-notes.html)
