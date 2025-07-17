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

To register a custom LLM provider, you need to create a class that inherits from `BaseLanguageModel` and register it using `register_llm_provider`.

It is important to implement the following methods:

**Required**:

- `_call`
- `_llm_type`

**Optional**:

- `_acall`
- `_astream`
- `_stream`
- `_identifying_params`

In other words, to create your custom LLM provider, you need to implement the following interface methods: `_call`, `_llm_type`, and optionally `_acall`, `_astream`, `_stream`, and `_identifying_params`. Here's how you can do it:

```python
from typing import Any, Iterator, List, Optional

from langchain.base_language import BaseLanguageModel
from langchain_core.callbacks.manager import (
    CallbackManagerForLLMRun,
    AsyncCallbackManagerForLLMRun,
)
from langchain_core.outputs import GenerationChunk

from nemoguardrails.llm.providers import register_llm_provider


class MyCustomLLM(BaseLanguageModel):

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs,
    ) -> str:
        pass

    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs,
    ) -> str:
        pass

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        pass

    # rest of the implementation
    ...

register_llm_provider("custom_llm", MyCustomLLM)
```

You can then use the custom LLM provider in your configuration:

```yaml
models:
  - type: main
    engine: custom_llm
```

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
