---
title: Custom Embedding Providers
description: Register custom embedding providers for vector similarity search in NeMo Guardrails.
---

# Custom Embedding Providers

Custom embedding providers enable you to use your own embedding models for semantic similarity search in the knowledge base and intent detection.

## Creating a Custom Embedding Provider

Create a class that inherits from `EmbeddingModel`:

```python
from typing import List
from nemoguardrails.embeddings.providers.base import EmbeddingModel


class CustomEmbedding(EmbeddingModel):
    """Custom embedding provider."""

    engine_name = "custom_embedding"

    def __init__(self, embedding_model: str):
        """Initialize the embedding model.

        Args:
            embedding_model: The model name from config.yml
        """
        self.model_name = embedding_model
        # Initialize your model here
        self.model = load_model(embedding_model)

    def encode(self, documents: List[str]) -> List[List[float]]:
        """Encode documents into embeddings (synchronous).

        Args:
            documents: List of text documents to encode

        Returns:
            List of embedding vectors
        """
        return [self.model.encode(doc) for doc in documents]

    async def encode_async(self, documents: List[str]) -> List[List[float]]:
        """Encode documents into embeddings (asynchronous).

        Args:
            documents: List of text documents to encode

        Returns:
            List of embedding vectors
        """
        # For simple models, can just call sync version
        return self.encode(documents)
```

## Registering the Provider

Register the provider in your `config.py`:

```python
from nemoguardrails import LLMRails


def init(app: LLMRails):
    from .embeddings import CustomEmbedding

    app.register_embedding_provider(CustomEmbedding, "custom_embedding")
```

## Using the Provider

Configure in `config.yml`:

```yaml
models:
  - type: embeddings
    engine: custom_embedding
    model: my-model-name
```

## Example: Sentence Transformers

```python
from typing import List
from sentence_transformers import SentenceTransformer
from nemoguardrails.embeddings.providers.base import EmbeddingModel


class SentenceTransformerEmbedding(EmbeddingModel):
    """Embedding provider using sentence-transformers."""

    engine_name = "sentence_transformers"

    def __init__(self, embedding_model: str):
        self.model = SentenceTransformer(embedding_model)

    def encode(self, documents: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(documents)
        return embeddings.tolist()

    async def encode_async(self, documents: List[str]) -> List[List[float]]:
        return self.encode(documents)
```

**config.py:**

```python
from nemoguardrails import LLMRails

def init(app: LLMRails):
    app.register_embedding_provider(
        SentenceTransformerEmbedding,
        "sentence_transformers"
    )
```

**config.yml:**

```yaml
models:
  - type: embeddings
    engine: sentence_transformers
    model: all-MiniLM-L6-v2
```

## Example: OpenAI-Compatible API

```python
from typing import List
import httpx
from nemoguardrails.embeddings.providers.base import EmbeddingModel


class OpenAICompatibleEmbedding(EmbeddingModel):
    """Embedding provider for OpenAI-compatible APIs."""

    engine_name = "openai_compatible"

    def __init__(self, embedding_model: str):
        self.model = embedding_model
        self.api_url = "http://localhost:8080/v1/embeddings"

    def encode(self, documents: List[str]) -> List[List[float]]:
        response = httpx.post(
            self.api_url,
            json={"input": documents, "model": self.model}
        )
        data = response.json()
        return [item["embedding"] for item in data["data"]]

    async def encode_async(self, documents: List[str]) -> List[List[float]]:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.api_url,
                json={"input": documents, "model": self.model}
            )
            data = response.json()
            return [item["embedding"] for item in data["data"]]
```

## Required Methods

| Method | Description |
|--------|-------------|
| `__init__(embedding_model: str)` | Initialize with model name from config |
| `encode(documents: List[str])` | Synchronous encoding |
| `encode_async(documents: List[str])` | Asynchronous encoding |

## Class Attributes

| Attribute | Description |
|-----------|-------------|
| `engine_name` | Identifier used in `config.yml` |
