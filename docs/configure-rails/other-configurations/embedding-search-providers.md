---
title:
  page: "Embedding Search Providers"
  nav: "Embedding Search Providers"
description: "Configure embedding search providers for vector similarity search using FastEmbed, OpenAI, or custom implementations."
keywords: ["nemo guardrails embeddings", "vector search", "FastEmbed", "OpenAI embeddings", "custom embedding provider"]
topics: ["generative_ai", "developer_tools"]
tags: ["llms", "ai_inference", "embeddings"]
content:
  type: reference
  difficulty: technical_intermediate
  audience: ["engineer"]
---

# Embedding Search Providers

NeMo Guardrails utilizes embedding search, also known as vector databases, for implementing the [guardrails process](../../reference/colang-architecture-guide.md#the-guardrails-process) and for the [knowledge base](knowledge-base.md) functionality.

To enhance the efficiency of the embedding search process, NeMo Guardrails can employ a caching mechanism for embeddings. This mechanism stores computed embeddings, thereby reducing the need for repeated computations and accelerating the search process. By default, the caching mechanism is disabled.

The default embedding search uses FastEmbed for computing the embeddings (the `all-MiniLM-L6-v2` model) and Annoy for performing the search. The default configuration is as follows:

```yaml
core:
  embedding_search_provider:
    name: default
    parameters:
      embedding_engine: FastEmbed
      embedding_model: all-MiniLM-L6-v2
      use_batching: False
      max_batch_size: 10
      max_batch_hold: 0.01
      search_threshold: None
    cache:
      enabled: False
      key_generator: sha256
      store: filesystem
      store_config: {}

knowledge_base:
  embedding_search_provider:
    name: default
    parameters:
      embedding_engine: FastEmbed
      embedding_model: all-MiniLM-L6-v2
      use_batching: False
      max_batch_size: 10
      max_batch_hold: 0.01
      search_threshold: None
    cache:
      enabled: False
      key_generator: sha256
      store: filesystem
      store_config: {}
```

The default embedding search provider can also work with OpenAI embeddings:

```yaml
core:
  embedding_search_provider:
    name: default
    parameters:
      embedding_engine: openai
      embedding_model: text-embedding-3-small
    cache:
      enabled: False
      key_generator: sha256
      store: filesystem
      store_config: {}

knowledge_base:
  embedding_search_provider:
    name: default
    parameters:
      embedding_engine: openai
      embedding_model: text-embedding-3-small
    cache:
      enabled: False
      key_generator: sha256
      store: filesystem
      store_config: {}
```

The default embedding search provider can also use embedding models served through AWS Bedrock. Install the optional dependency with `pip install nemoguardrails[bedrock]` and configure as follows:

```yaml
core:
  embedding_search_provider:
    name: default
    parameters:
      embedding_engine: bedrock
      embedding_model: amazon.titan-embed-text-v2:0
      # Optional. If omitted, boto3 resolves the region from AWS_REGION,
      # AWS_DEFAULT_REGION, or ~/.aws/config.
      region_name: us-east-1
      # Optional AWS profile from ~/.aws/credentials.
      # profile_name: my-profile
      # Optional, only for amazon.titan-embed-text-v2:* (256, 512, or 1024).
      # dimensions: 1024
      # normalize: true
      # Optional, only for cohere.* models
      # (search_document, search_query, classification, clustering).
      # input_type: search_document
    cache:
      enabled: False
      key_generator: sha256
      store: filesystem
      store_config: {}

knowledge_base:
  embedding_search_provider:
    name: default
    parameters:
      embedding_engine: bedrock
      embedding_model: amazon.titan-embed-text-v2:0
      region_name: us-east-1
```

Supported models include `amazon.titan-embed-text-v1`, `amazon.titan-embed-text-v2:0`, and the Cohere Embed v3 family on Bedrock (`cohere.embed-english-v3`, `cohere.embed-multilingual-v3`, `cohere.embed-english-light-v3`, `cohere.embed-multilingual-light-v3`). Models from other vendors raise `ValueError` at configuration time. The target model must be enabled in your AWS account and region. Authentication uses the standard `boto3` credential chain (environment variables, `~/.aws/credentials`, IAM role); `region_name` is optional when the region is already set via that chain.

The default implementation is also designed to support asynchronous execution of the embedding computation process, thereby enhancing the efficiency of the search functionality.

The `cache` configuration is optional. If enabled, it uses the specified `key_generator` and `store` to cache the embeddings. The `store_config` can be used to provide additional configuration options required for the store.
The default `cache` configuration uses the `sha256` key generator and the `filesystem` store. The cache is disabled by default.

## Batch Implementation

The default embedding provider includes a batch processing feature designed to optimize the embedding generation process. This feature is designed to initiate the embedding generation process after a predefined latency of 10 milliseconds.

## Custom Embedding Search Providers

You can implement your own custom embedding search provider by subclassing `EmbeddingsIndex`. For quick reference, the complete interface is included below:

```python
class EmbeddingsIndex:
    """The embeddings index is responsible for computing and searching a set of embeddings."""

    @property
    def embedding_size(self):
        raise NotImplementedError

    @property
    def cache_config(self):
      raise NotImplementedError

    async def _get_embeddings(self, texts: List[str]):
        raise NotImplementedError

    async def add_item(self, item: IndexItem):
        """Adds a new item to the index."""
        raise NotImplementedError()

    async def add_items(self, items: List[IndexItem]):
        """Adds multiple items to the index."""
        raise NotImplementedError()

    async def build(self):
        """Build the index, after the items are added.

        This is optional, might not be needed for all implementations."""
        pass

    async def search(self, text: str, max_results: int, threshold: Optional[float]) -> List[IndexItem]:
        """Searches the index for the closest matches to the provided text."""
        raise NotImplementedError()

@dataclass
class IndexItem:
    text: str
    meta: Dict = field(default_factory=dict)
```

In order to use your custom embedding search provider, you have to register it in your `config.py`:

```python
def init(app: LLMRails):
    app.register_embedding_search_provider("simple", SimpleEmbeddingSearchProvider)
```

For a complete example, check out [this test configuration](https://github.com/NVIDIA-NeMo/Guardrails/tree/develop/tests/test_configs/with_custom_embedding_search_provider).
