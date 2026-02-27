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

(v0-20-0)=

## 0.20.0

(v0-20-0-features)=

### Key Features

- Added support for multilingual content safety models such as [NVIDIA Nemotron Safety Guard 8B v3](https://build.nvidia.com/nvidia/llama-3_1-nemotron-safety-guard-8b-v3). This feature uses the [fast-langdetect package](https://github.com/LlmKira/fast-langdetect) to detect the user's input language and return refusal messages in the appropriate language. To use this feature, install the NeMo Guardrails library with the `multilingual` extra.

  ```bash
  pip install nemoguardrails[multilingual]
  ```

- Added support for configuring custom refusal messages per language to complement multilingual content safety models. You can enable multilingual refusal messages and specify custom refusal messages in the `rails.config.content_safety` section of the `config.yml` file.

  ```yaml
  rails:
    config:
      content_safety:
        multilingual:
          enabled: true
          refusal_messages:
            en: "Sorry, I cannot help with that request."
            es: "Lo siento, no puedo ayudar con esa solicitud."
            zh: "抱歉，我无法处理该请求。"
            # Add other languages as needed
  ```

  For more information, refer to [](../configure-rails/guardrail-catalog/content-safety.md#multilingual-refusal-messages).
- Added support for [NVIDIA GLiNER-PII](https://huggingface.co/nvidia/gliner-PII) for detecting entities such as names, email addresses, phone numbers, social security numbers, and more. For more information, refer to [](../configure-rails/guardrail-catalog/community/gliner.md).

### Breaking Changes

- A breaking change removes redundant streaming configuration for output rails. Prior to the change, streaming had to be enabled in two places: `streaming` and `rails.output.streaming.enabled`. This change removes the top-level `streaming` configuration.
  - Example `config.yml` before the change:

    ```{code-block} yaml
    :emphasize-lines: 21

    models:
      - type: main
        engine: nvidia_ai_endpoints
        model: meta/llama-3.3-70b-instruct
      - type: content_safety
        engine: nvidia_ai_endpoints
        model: nvidia/llama-3.1-nemoguard-8b-content-safety

    rails:
      input:
        flows:
          - content safety check input $model=content_safety
      output:
        flows:
          - content safety check output $model=content_safety
        streaming:
          enabled: True
          chunk_size: 200
          context_size: 50

    streaming: True # No longer needed starting from v0.20.0
    ```

  - Example `config.yml` after the change:

    ```yaml
    models:
      - type: main
        engine: nvidia_ai_endpoints
        model: meta/llama-3.3-70b-instruct

      - type: content_safety
        engine: nvidia_ai_endpoints
        model: nvidia/llama-3.1-nemoguard-8b-content-safety

    rails:
      input:
        flows:
          - content safety check input $model=content_safety
      output:
        flows:
          - content safety check output $model=content_safety
        streaming:
          enabled: True
          chunk_size: 200
          context_size: 50
    ```

  For more information, refer to [](../run-rails/using-python-apis/streaming.md).

### Other Changes

- Restructured the documentation with improved navigation, clearer content organization, and updated configuration reference and user guides.

---

## Previous Release Notes

- [0.19.0](https://docs.nvidia.com/nemo/guardrails/0.19.0/release-notes.html)
- [0.18.0](https://docs.nvidia.com/nemo/guardrails/0.18.0/release-notes.html)
- [0.17.0](https://docs.nvidia.com/nemo/guardrails/0.17.0/release-notes.html)
- [0.16.0](https://docs.nvidia.com/nemo/guardrails/0.16.0/release-notes.html)
- [0.15.0](https://docs.nvidia.com/nemo/guardrails/0.15.0/release-notes.html)
- [0.14.1](https://docs.nvidia.com/nemo/guardrails/0.14.1/release-notes.html)
- [0.14.0](https://docs.nvidia.com/nemo/guardrails/0.14.0/release-notes.html)
