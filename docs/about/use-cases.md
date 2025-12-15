---
title: Use Cases
description: Browse the different use cases of the NeMo Guardrails Library.
---

# Use Cases

The NeMo Guardrails Library supports a wide range of use cases for protecting LLM-based applications.
The following sections describe the primary use cases.

---

## Content Safety

Content safety guardrails help ensure that both user inputs and LLM outputs are safe and appropriate.
The NeMo Guardrails Library provides multiple approaches to content safety:

- **LLM self-checking**: Use the LLM itself to check inputs and outputs for harmful content.
- **NVIDIA safety models**: Integrate with [Llama 3.1 NemoGuard 8B Content Safety](https://build.nvidia.com/nvidia/llama-3_1-nemoguard-8b-content-safety) for robust content moderation.
- **Community models**: Use [LlamaGuard](../user-guides/community/llama-guard.md), [Fiddler Guardrails](../user-guides/community/fiddler.md), and other community content safety solutions.
- **Third-party APIs**: Integrate with [ActiveFence](../user-guides/guardrails-library.md#activefence), [Cisco AI Defense](../user-guides/community/ai-defense.md), and other moderation services.

For more information, refer to the [Content Safety section](../user-guides/guardrails-library.md#content-safety) in the Guardrails Library and the [Getting Started guide](../getting-started/index.md).

## Jailbreak Protection

Jailbreak protection helps prevent adversarial attempts from bypassing safety measures and manipulating the LLM into generating harmful or unwanted content.
The NeMo Guardrails Library provides multiple layers of jailbreak protection:

- **Self-check jailbreak detection**: Use the LLM to identify jailbreak attempts.
- **Heuristic detection**: Use pattern-based detection for common jailbreak techniques.
- **NVIDIA NemoGuard**: Integrate with [NemoGuard Jailbreak Detection NIM](../getting-started/tutorials/nemoguard-jailbreakdetect-deployment.md) for advanced threat detection.
- **Third-party integrations**: Use [Prompt Security](../user-guides/community/prompt-security.md), [Pangea AI Guard](../user-guides/community/pangea.md), and other services.

For more information, refer to the [Jailbreak Detection section](../user-guides/guardrails-library.md#jailbreak-detection) in the Guardrails Library and [LLM Vulnerability Scanning](../evaluation/llm-vulnerability-scanning.md).

## Topic Control

Topic control guardrails ensure that conversations stay within predefined subject boundaries and prevent the LLM from engaging in off-topic discussions.
This is implemented through:

- **Dialog rails**: Pre-define conversational flows using the Colang language.
- **Topical rails**: Control what topics the bot can and cannot discuss.
- **NVIDIA NemoGuard**: Integrate with [NemoGuard Topic Control NIM](../getting-started/tutorials/nemoguard-topiccontrol-deployment.md) for semantic topic detection.

For more information, refer to the [Topical Rails tutorial](../getting-started/6-topical-rails/README.md) and [Colang Language Syntax Guide](../user-guides/colang-language-syntax-guide.md).

## PII Detection

Personally Identifiable Information (PII) detection helps protect user privacy by detecting and masking sensitive data in user inputs, LLM outputs, and retrieved content.
The NeMo Guardrails Library supports PII detection through multiple integrations:

- **Presidio-based detection**: Use [Microsoft Presidio](../user-guides/community/presidio.md) for detecting entities such as names, email addresses, phone numbers, social security numbers, and more.
- **Private AI**: Integrate with [Private AI](../user-guides/community/privateai.md) for advanced PII detection and masking.
- **AutoAlign**: Use [AutoAlign PII detection](../user-guides/community/auto-align.md) with customizable entity types.
- **GuardrailsAI**: Access [GuardrailsAI PII validators](../user-guides/community/guardrails-ai.md) from the Guardrails Hub.

PII detection can be configured to either detect and block content containing PII or to mask PII entities before processing.

For more information, refer to the [Presidio Integration](../user-guides/community/presidio.md) and [Sensitive Data Detection section](../configure-rails/yaml-schema/guardrails-configuration/built-in-guardrails.md#presidio-based-sensitive-data-detection) in the built-in Guardrails library.

## Agentic Security (Security Rails for Agent Systems)

Agentic security provides specialized guardrails for LLM-based agents that use tools and interact with external systems.
This includes:

- **Tool call validation**: Execute rails that validate tool inputs and outputs before and after invocation.
- **Agent workflow protection**: Integrate with [LangGraph](../integration/langchain/langgraph-integration.md) for multi-agent safety.
- **Secure tool integration**: Review guidlines for safety connecting LLMs to external resources (refer to [Security Guidelines](../security/guidelines.md)).
- **Action monitoring**: Monitor detailed logging and tracing of agent actions.

Key security considerations for agent systems:

1. Isolate all authentication information from the LLM.
2. Validate and sanitize all tool inputs.
3. Apply execution rails to tool calls.
4. Monitor agent behavior for unexpected actions.

For more information, refer to the [Tools Integration Guide](../integration/tools-integration.md), [Security Guidelines](../security/guidelines.md), and [LangGraph Integration](../integration/langchain/langgraph-integration.md).

## Custom Rails

The NeMo Guardrails Library provides extensive flexibility for creating custom guardrails tailored to your specific requirements.

### Add Custom Rails into Guardrails

If you have a script or tool that runs a custom guardrail, you can use it in NeMo Guardrails by following one of these approaches:

1. **Python actions**: Create custom actions in Python for complex logic and external integrations.

   ```python
   from nemoguardrails.actions import action

   @action()
   async def check_custom_policy(context: dict):
       # Custom validation logic
       return True
   ```

   For more information, refer to the [Python API Guide](../python-api/index.md).

2. **LangChain tool integration**: Register LangChain tools as custom actions.

   ```python
   from langchain_core.tools import tool

   @tool
   def custom_tool(query: str) -> str:
       """Custom tool implementation."""
       return result

   rails.register_action(custom_tool, "custom_action")
   ```

   For more information, refer to the [Tools Integration Guide](../integration/tools-integration.md).

3. **Third-party API integration**: Integrate external moderation and validation services.
   For examples, refer to the [Guardrails Library](../user-guides/guardrails-library.md) which includes integrations with ActiveFence, AutoAlign, Fiddler, and other services.

### Integrate Guardrails into LLM-based Applications

The NeMo Guardrails Library can be integrated into applications in multiple ways:

1. **Python SDK integration**: Add guardrails directly into your Python application.

   ```python
   from nemoguardrails import LLMRails, RailsConfig

   config = RailsConfig.from_path("path/to/config")
   rails = LLMRails(config)

   # Use in your application
   response = rails.generate(messages=[...])
   ```

2. **LangChain integration**: Wrap guardrails around LangChain chains or use chains within guardrails.

   ```python
   from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails

   guardrails = RunnableRails(config)
   chain_with_guardrails = prompt | guardrails | model | output_parser
   ```

   For more information, refer to the [LangChain Integration Guide](../integration/langchain/langchain-integration.md).

3. **HTTP API integration**: Use the guardrails server to add protection to applications in any programming language.

   ```bash
   nemoguardrails server --config path/to/configs
   ```

   For more information, refer to the [Server Guide](../deployment/local-server/index.md).

4. **Docker deployment**: Deploy guardrails as a containerized service.
   For more information, refer to the [Using Docker Guide](../deployment/using-docker.md).

For complete examples and detailed integration patterns, refer to the [examples directory](https://github.com/NVIDIA/NeMo-Guardrails/tree/develop/examples) in the GitHub repository.
