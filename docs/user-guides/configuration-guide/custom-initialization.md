# Custom Initialization

If present, the `config.py` module is loaded before initializing the `LLMRails` instance.

If the `config.py` module contains an `init` function, it gets called as part of the initialization of the `LLMRails` instance. For example, you can use the `init` function to initialize the connection to a database and register it as a custom action parameter using the `register_action_param(...)` function:

```python
from nemoguardrails import LLMRails

def init(app: LLMRails):
    # Initialize the database connection
    db = ...

    # Register the action parameter
    app.register_action_param("db", db)
```

Custom action parameters are passed on to the custom actions when they are invoked.

## Custom Data Access

If you need to pass additional configuration data to any custom component for your configuration, you can use the `custom_data` field in your `config.yml`:

```yaml
custom_data:
  custom_config_field: "some_value"
```

For example, you can access the custom configuration inside the `init` function in your `config.py`:

```python
def init(app: LLMRails):
    config = app.config

    # Do something with config.custom_data
```

## Custom LLM Provider Registration

NeMo Guardrails supports two types of custom LLM providers:
1. **Text Completion Models** (`BaseLLM`) - For models that work with string prompts
2. **Chat Models** (`BaseChatModel`) - For models that work with message-based conversations

### Custom Text Completion LLM (BaseLLM)

To register a custom text completion LLM provider, create a class that inherits from `BaseLLM` and register it using `register_llm_provider`.

**Required methods:**
- `_call` - Synchronous text completion
- `_llm_type` - Returns the LLM type identifier

**Optional methods:**
- `_acall` - Asynchronous text completion (recommended)
- `_stream` - Streaming text completion
- `_astream` - Async streaming text completion
- `_identifying_params` - Returns parameters for model identification

```python
from typing import Any, Iterator, List, Optional

from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseLLM
from langchain_core.outputs import GenerationChunk

from nemoguardrails.llm.providers import register_llm_provider


class MyCustomTextLLM(BaseLLM):
    """Custom text completion LLM."""

    @property
    def _llm_type(self) -> str:
        return "custom_text_llm"

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
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Asynchronous text completion (recommended)."""
        # Your async implementation here
        return "Generated text response"

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        """Optional: Streaming text completion."""
        # Yield chunks of text
        yield GenerationChunk(text="chunk1")
        yield GenerationChunk(text="chunk2")


register_llm_provider("custom_text_llm", MyCustomTextLLM)
```

### Custom Chat Model (BaseChatModel)

To register a custom chat model, create a class that inherits from `BaseChatModel` and register it using `register_chat_provider`.

**Required methods:**
- `_generate` - Synchronous chat completion
- `_llm_type` - Returns the LLM type identifier

**Optional methods:**
- `_agenerate` - Asynchronous chat completion (recommended)
- `_stream` - Streaming chat completion
- `_astream` - Async streaming chat completion

```python
from typing import Any, Iterator, List, Optional

from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

from nemoguardrails.llm.providers import register_chat_provider


class MyCustomChatModel(BaseChatModel):
    """Custom chat model."""

    @property
    def _llm_type(self) -> str:
        return "custom_chat_model"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Synchronous chat completion."""
        # Convert messages to your model's format and generate response
        response_text = "Generated chat response"

        message = AIMessage(content=response_text)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Asynchronous chat completion (recommended)."""
        # Your async implementation
        response_text = "Generated chat response"

        message = AIMessage(content=response_text)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Optional: Streaming chat completion."""
        # Yield chunks
        chunk = ChatGenerationChunk(message=AIMessageChunk(content="chunk1"))
        yield chunk


register_chat_provider("custom_chat_model", MyCustomChatModel)
```

### Using Custom LLM Providers

After registering your custom provider, you can use it in your configuration:

```yaml
models:
  - type: main
    engine: custom_text_llm  # or custom_chat_model
```

### Important Notes

1. **Import from langchain-core:** Always import base classes from `langchain_core.language_models`:
   ```python
   from langchain_core.language_models import BaseLLM, BaseChatModel
   ```

2. **Implement async methods:** For better performance, always implement `_acall` (for BaseLLM) or `_agenerate` (for BaseChatModel).

3. **Choose the right base class:**
   - Use `BaseLLM` for text completion models (prompt → text)
   - Use `BaseChatModel` for chat models (messages → message)

4. **Registration functions:**
   - Use `register_llm_provider()` for `BaseLLM` subclasses
   - Use `register_chat_provider()` for `BaseChatModel` subclasses

## Custom Embedding Provider Registration

You can also register a custom embedding provider by using the `LLMRails.register_embedding_provider` function.

To register a custom embedding provider, create a class that inherits from `EmbeddingModel` and register it in your `config.py`.

```python
from typing import List
from nemoguardrails.embeddings.providers.base import EmbeddingModel
from nemoguardrails import LLMRails


class CustomEmbeddingModel(EmbeddingModel):
    """An implementation of a custom embedding provider."""
    engine_name = "CustomEmbeddingModel"

    def __init__(self, embedding_model: str):
        # Initialize the model
        ...

    async def encode_async(self, documents: List[str]) -> List[List[float]]:
        """Encode the provided documents into embeddings.

        Args:
            documents (List[str]): The list of documents for which embeddings should be created.

        Returns:
            List[List[float]]: The list of embeddings corresponding to the input documents.
        """
        ...

    def encode(self, documents: List[str]) -> List[List[float]]:
        """Encode the provided documents into embeddings.

        Args:
            documents (List[str]): The list of documents for which embeddings should be created.

        Returns:
            List[List[float]]: The list of embeddings corresponding to the input documents.
        """
        ...


def init(app: LLMRails):
    """Initialization function in your config.py."""
    app.register_embedding_provider(CustomEmbeddingModel, "CustomEmbeddingModel")
```

You can then use the custom embedding provider in your configuration:

```yaml
models:
  # ...
  - type: embeddings
    engine: SomeCustomName
    model: SomeModelName      # supported by the provider.
```
