---
title:
  page: "Overview of the NeMo Guardrails Library"
  nav: "Overview"
description: "Add programmable guardrails to LLM applications with this open-source Python library."
keywords: ["nemo guardrails", "llm safety", "content moderation", "guardrails python"]
topics: ["generative_ai", "cybersecurity"]
tags: ["llms", "security_for_ai", "nlp", "ai_inference"]
content:
  type: concept
  difficulty: technical_beginner
  audience: [engineer, data_scientist]
---

<!--
  SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Overview of NVIDIA NeMo Guardrails Library

The NVIDIA NeMo Guardrails library ([PyPI](https://pypi.org/project/nemoguardrails/) | [GitHub](https://github.com/NVIDIA-NeMo/Guardrails)) is an open-source Python package for adding programmable guardrails to LLM-based applications. Guardrails make your LLM-based application safer and more secure by blocking inappropriate, off-topic or malicious user inputs or LLM responses.

## NeMo Guardrails Library within the NVIDIA NeMo Software Stack

[NVIDIA NeMo](https://www.nvidia.com/en-us/ai-data-science/products/nemo/) is a suite of microservices, tools, and libraries for building, deploying, and scaling LLM-based applications.

NeMo Guardrails is part of the NVIDIA NeMo software stack. It takes part in adding programmable guardrails to LLM-based applications. The NeMo Guardrails library provides tools to build guardrails and integrate them into your LLM-based applications at development time. The NeMo Guardrails microservice as part of the [NeMo microservices platform](https://docs.nvidia.com/nemo/microservices/latest/about/index.html) is a production-ready container image built on top of this library, designed for Kubernetes deployment with Helm charts.

|                  | NeMo Guardrails Library          | NeMo Guardrails Microservice     |
|------------------|----------------------------------|----------------------------------|
| Distribution     | PyPI (`pip install`)             | Container image (backed by this library) |
| Deployment       | Self-managed Python environment  | Kubernetes with Helm             |
| Scaling          | Application-level                | Managed by orchestrator          |
| Configuration    | YAML + Colang                    | Same YAML + Colang format        |

Configurations are portable between the library and microservice, so you can develop locally with the library and deploy to production with the microservice.

## Use Cases

The following are the top use cases of the NeMo Guardrails library that you can apply to protect your LLM applications.

:::{dropdown} 🛡️ Add Content Safety

Content safety guardrails help ensure that both user inputs and LLM outputs are safe and appropriate.
The NeMo Guardrails library provides multiple approaches to content safety:

- **LLM self-checking**: Use the LLM itself to check inputs and outputs for harmful content.
- **NVIDIA safety models**: Integrate with [Llama 3.1 NemoGuard 8B Content Safety](https://build.nvidia.com/nvidia/llama-3_1-nemotron-safety-guard-8b-v3) for robust content moderation.
- **Community models**: Use [LlamaGuard](../user-guides/community/llama-guard.md), [Fiddler Guardrails](../user-guides/community/fiddler.md), and other community content safety solutions.
- **Third-party APIs**: Integrate with [ActiveFence](../configure-rails/guardrail-catalog.md#activefence), [Cisco AI Defense](../user-guides/community/ai-defense.md), and other moderation services.

For practical examples, try the following tutorials:

- [Content Safety - Text](../getting-started/tutorials/nemotron-safety-guard-deployment.md)
- [Content Safety - Multimodal](../getting-started/tutorials/multimodal.md)
:::

:::{dropdown} 🔒 Add Jailbreak Protection

Jailbreak protection helps prevent adversarial attempts from bypassing safety measures and manipulating the LLM into generating harmful or unwanted content.
The NeMo Guardrails library provides multiple layers of jailbreak protection:

- **Self-check jailbreak detection**: Use the LLM to identify jailbreak attempts.
- **Heuristic detection**: Use pattern-based detection for common jailbreak techniques.
- **NVIDIA NemoGuard**: Integrate with [NemoGuard Jailbreak Detection NIM](../getting-started/tutorials/nemoguard-jailbreakdetect-deployment.md) for advanced threat detection.
- **Third-party integrations**: Use [Prompt Security](../user-guides/community/prompt-security.md), [Pangea AI Guard](../user-guides/community/pangea.md), and other services.

For practical examples, try the following tutorial:

- [](../getting-started/tutorials/nemoguard-jailbreakdetect-deployment.md)
:::

:::{dropdown} 🎯 Control Topic Conversation

Topic control guardrails ensure that conversations stay within predefined subject boundaries and prevent the LLM from engaging in off-topic discussions.
This is implemented through:

- **Dialog rails**: Pre-define conversational flows using the Colang language.
- **Topical rails**: Control what topics the bot can and cannot discuss.
- **NVIDIA NemoGuard**: Integrate with [NemoGuard Topic Control NIM](../getting-started/tutorials/nemoguard-topiccontrol-deployment.md) for semantic topic detection.

For practical examples, try the following tutorial:

- [](../getting-started/tutorials/nemoguard-topiccontrol-deployment.md)
:::

:::{dropdown} 🔐 Detect and Mask PII

Personally Identifiable Information (PII) detection helps protect user privacy by detecting and masking sensitive data in user inputs, LLM outputs, and retrieved content.
The NeMo Guardrails library supports PII detection through multiple integrations:

- **Gliner**: Use [NVIDIA GLiNER-PII](../user-guides/community/gliner.md) for detecting entities such as names, email addresses, phone numbers, social security numbers, and more.
- **Presidio-based detection**: Use [Microsoft Presidio](../user-guides/community/presidio.md) for detecting entities such as names, email addresses, phone numbers, social security numbers, and more.
- **Private AI**: Integrate with [Private AI](../user-guides/community/privateai.md) for advanced PII detection and masking.
- **AutoAlign**: Use [AutoAlign PII detection](../user-guides/community/auto-align.md) with customizable entity types.
- **GuardrailsAI**: Access [GuardrailsAI PII validators](../user-guides/community/guardrails-ai.md) from the Guardrails Hub.

PII detection can be configured to either detect and block content containing PII or to mask PII entities before processing.

For more information, refer to the [Presidio Integration](../user-guides/community/presidio.md) and [Sensitive Data Detection section](../configure-rails/guardrail-catalog.md#presidio-based-sensitive-data-detection) in the Guardrail Catalog.
:::

:::{dropdown} 🤖 Add Agentic Security

Agentic security provides specialized guardrails for LLM-based agents that use tools and interact with external systems.
This includes:

- **Tool call validation**: Execute rails that validate tool inputs and outputs before and after invocation.
- **Agent workflow protection**: Integrate with [LangGraph](../integration/langchain/langgraph-integration.md) for multi-agent safety.
- **Secure tool integration**: Review guidelines for safely connecting LLMs to external resources (refer to [Security Guidelines](../resources/security/guidelines.md)).
- **Action monitoring**: Monitor detailed logging and tracing of agent actions.

Key security considerations for agent systems:

1. Isolate all authentication information from the LLM.
2. Validate and sanitize all tool inputs.
3. Apply execution rails to tool calls.
4. Monitor agent behavior for unexpected actions.

For more information, refer to the [Tools Integration Guide](../integration/tools-integration.md), [Security Guidelines](../resources/security/guidelines.md), and [LangGraph Integration](../integration/langchain/langgraph-integration.md).
:::

:::{dropdown} 🔧 Build Your Own or Use Third-party Guardrail Solutions

The NeMo Guardrails library provides extensive flexibility for creating custom guardrails tailored to your specific requirements. You can either build your own guardrails or use third-party guardrails.
If you have a script or tool that runs a custom guardrail, you can use it in NeMo Guardrails by following one of these approaches:

1. **Python actions**: Create custom actions in Python for complex logic and external integrations. For more information, refer to the [](../configure-rails/actions/index.md).

2. **LangChain tool integration**: Register LangChain tools as custom actions. For more information, refer to the [](../integration/tools-integration.md).

3. **Third-party API integration**: Integrate external moderation and validation services. For a complete list of supported third-party guardrail services, refer to the [Third-Party APIs](../configure-rails/guardrail-catalog.md#third-party-apis) section in the Guardrail Catalog.
:::

:::{dropdown} 🔌 Integrate NeMo Guardrails Library into Your Application

You can integrate the NeMo Guardrails library into your application using the tools provided by the library.

1. **Python SDK**: Use the Python SDK to add guardrails directly into your Python application.

    ```python
    from nemoguardrails import LLMRails, RailsConfig

    config = RailsConfig.from_path("./config")
    rails = LLMRails(config)

    response = rails.generate(
        messages=[{"role": "user", "content": "Hello!"}]
    )
    ```

    The `generate` method accepts the same message format as the OpenAI Chat Completions API.

2. **API Server**: You can solely set up a guardrails server after programming guardrails using the Python SDK. You can then start a local NeMo Guardrails server with the following command.

    ```bash
    nemoguardrails server --config ./config --port 8000
    ```

    The server exposes API endpoints such as `/v1/chat/completions` for guardrailed chat completions.

    ```bash
    curl -X POST http://localhost:8000/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{
        "config_id": "my-config",
        "messages": [{"role": "user", "content": "Hello!"}]
      }'
    ```

    <!-- The server exposes HTTP APIs compatible with OpenAI's `/v1/chat/completions` endpoint. You can then use the server in your application by sending requests to the server's endpoint. -->

:::

---

## Next Steps

To get started with NeMo Guardrails, begin by [installing the library](../getting-started/installation-guide.md) and try out one of the [tutorials](../getting-started/tutorials/index.md).
