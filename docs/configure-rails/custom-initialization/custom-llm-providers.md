---
title: Custom LLM Providers
description: Register custom text completion (BaseLLM) and chat models (BaseChatModel) for use with NeMo Guardrails.
---

# Custom LLM Providers

NeMo Guardrails supports two types of custom LLM providers:

| Type | Base Class | Input | Output |
|------|------------|-------|--------|
| Text Completion | `BaseLLM` | String prompt | String response |
| Chat Model | `BaseChatModel` | List of messages | Message response |

## Text Completion Models (BaseLLM)

For models that work with string prompts:

```python
from typing import Any, List, Optional

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models import BaseLLM

from nemoguardrails.llm.providers import register_llm_provider


class MyCustomLLM(BaseLLM):
    """Custom text completion LLM."""

    @property
    def _llm_type(self) -> str:
        return "my_custom_llm"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Synchronous text completion."""
        # Your implementation here
        return "Generated text response"

    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Asynchronous text completion (recommended)."""
        # Your async implementation here
        return "Generated text response"


# Register the provider
register_llm_provider("my_custom_llm", MyCustomLLM)
```

## Chat Models (BaseChatModel)

For models that work with message-based conversations:

```python
from typing import Any, List, Optional

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from nemoguardrails.llm.providers import register_chat_provider


class MyCustomChatModel(BaseChatModel):
    """Custom chat model."""

    @property
    def _llm_type(self) -> str:
        return "my_custom_chat"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Synchronous chat completion."""
        # Convert messages to your model's format
        response_text = "Generated chat response"

        message = AIMessage(content=response_text)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Asynchronous chat completion (recommended)."""
        response_text = "Generated chat response"

        message = AIMessage(content=response_text)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])


# Register the provider
register_chat_provider("my_custom_chat", MyCustomChatModel)
```

## Using Custom Providers

After registering your custom provider in `config.py`, use it in `config.yml`:

```yaml
models:
  - type: main
    engine: my_custom_llm  # or my_custom_chat
    model: optional-model-name
```

## Required and Optional Methods

### BaseLLM Methods

| Method | Required | Description |
|--------|----------|-------------|
| `_call` | Yes | Synchronous text completion |
| `_llm_type` | Yes | Returns the LLM type identifier |
| `_acall` | Recommended | Asynchronous text completion |
| `_stream` | Optional | Streaming text completion |
| `_astream` | Optional | Async streaming text completion |

### BaseChatModel Methods

| Method | Required | Description |
|--------|----------|-------------|
| `_generate` | Yes | Synchronous chat completion |
| `_llm_type` | Yes | Returns the LLM type identifier |
| `_agenerate` | Recommended | Asynchronous chat completion |
| `_stream` | Optional | Streaming chat completion |
| `_astream` | Optional | Async streaming chat completion |

## Best Practices

1. **Implement async methods**: For better performance, always implement `_acall` (for BaseLLM) or `_agenerate` (for BaseChatModel).

2. **Choose the right base class**:
   - Use `BaseLLM` for text completion models (prompt → text)
   - Use `BaseChatModel` for chat models (messages → message)

3. **Import from langchain-core**: Always import base classes from `langchain_core.language_models`.

4. **Use correct registration function**:
   - `register_llm_provider()` for `BaseLLM` subclasses
   - `register_chat_provider()` for `BaseChatModel` subclasses
