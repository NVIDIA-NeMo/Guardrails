---
title: Knowledge Base
description: Configure the knowledge base folder for RAG-based responses using markdown documents.
---

# Knowledge Base

The NeMo Guardrails library supports using a set of documents as context for generating bot responses through Retrieval-Augmented Generation (RAG). This guide explains how to configure and use the knowledge base folder.

## Overview

By default, an `LLMRails` instance supports using documents as context for generating responses. To include documents as part of your knowledge base, place them in the `kb` folder inside your configuration folder:

```text
.
├── config
│   ├── config.yml
│   ├── kb
│   │   ├── file_1.md
│   │   ├── file_2.md
│   │   └── ...
│   └── rails
│       └── ...
```

```{note}
Currently, only the Markdown format is supported.
```

## Document Structure

Documents in the knowledge base `kb` folder are automatically processed and indexed for retrieval. The system:

1. Splits documents into topic chunks based on markdown headers.
2. Uses the configured embedding model to create vector representations of each chunk.
3. Stores the embeddings for efficient similarity search.

### Example Document

```markdown
# Employee Handbook

## Time Off Policy

Employees are eligible for the following time off:
* Vacation: 20 days per year, accrued monthly.
* Sick leave: 15 days per year, accrued monthly.
* Personal days: 5 days per year, accrued monthly.

## Holiday Schedule

Paid holidays include:
* New Year's Day
* Memorial Day
* Independence Day
* Thanksgiving Day
* Christmas Day
```

## Retrieval Process

When a user query is received, the system:

1. Computes embeddings for the user query using the configured embedding model.
2. Performs similarity search against the indexed document chunks.
3. Retrieves the most relevant chunks based on similarity scores.
4. Makes the retrieved chunks available as `$relevant_chunks` in the context.
5. Uses these chunks as additional context when generating the bot response.

## Configuration

The knowledge base functionality is automatically enabled when documents are present in the `kb` folder. You can customize the behavior using the `knowledge_base` section in your `config.yml`:

```yaml
knowledge_base:
  folder: "kb"  # Default folder name
  embedding_search_provider:
    name: "default"
    parameters: {}
```

### Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `folder` | The folder from which documents should be loaded | `"kb"` |
| `embedding_search_provider.name` | The name of the embedding search provider | `"default"` |
| `embedding_search_provider.parameters` | Provider-specific parameters | `{}` |

### Embedding Model Configuration

The knowledge base uses the embedding model configured in the `models` section of your `config.yml`:

```yaml
models:
  - type: main
    engine: openai
    model: gpt-4

  - type: embeddings
    engine: openai
    model: text-embedding-ada-002
```

For more details on embedding model configuration, refer to [Model Configuration](../yaml-schema/model-configuration.md).

## Alternative Knowledge Base Methods

There are three ways to configure a knowledge base:

### 1. Using the kb Folder (Default)

Place markdown files in the `kb` folder as described above. This is the simplest approach for static document collections.

### 2. Using Custom retrieve_relevant_chunks Action

Implement a custom action to retrieve chunks from external sources:

```python
from nemoguardrails.actions import action

@action()
async def retrieve_relevant_chunks(context: dict, llm: BaseLLM):
    """Custom retrieval from external knowledge base."""
    user_message = context.get("last_user_message")

    # Implement custom retrieval logic
    # For example, query an external vector database
    chunks = await query_external_kb(user_message)

    return chunks
```

### 3. Using Custom EmbeddingSearchProvider

For advanced use cases, implement a custom embedding search provider:

```python
from nemoguardrails.embeddings.index import EmbeddingsIndex

class CustomEmbeddingSearchProvider(EmbeddingsIndex):
    """Custom embedding search provider."""

    async def add_item(self, item: IndexItem):
        # Custom indexing logic
        pass

    async def search(self, text: str, max_results: int) -> List[IndexItem]:
        # Custom search logic
        pass
```

For more details, refer to [Embedding Search Providers](../../user-guides/advanced/embedding-search-providers.md).

## Passing Context Directly

You can also pass relevant context directly when making a `generate` call:

```python
response = rails.generate(messages=[
    {
        "role": "context",
        "content": {
            "relevant_chunks": """
                Employees are eligible for the following time off:
                * Vacation: 20 days per year, accrued monthly.
                * Sick leave: 15 days per year, accrued monthly.
            """
        }
    },
    {
        "role": "user",
        "content": "How many vacation days do I have per year?"
    }
])
```

## Using Knowledge Base in Colang Flows

You can reference the retrieved chunks in your Colang flows:

````{tab-set}
```{tab-item} Colang 2.0
~~~colang
import core
import llm

flow main
  activate llm continuation

  user asked question
  $chunks = ..."Summarize the relevant information from the knowledge base."
  bot say $chunks

flow user asked question
  user said "what" or user said "how" or user said "tell me"
~~~
```

```{tab-item} Colang 1.0
~~~colang
define flow answer question
  user ask question
  # Use the retrieved knowledge base chunks to answer
  bot respond with knowledge
~~~
```
````

## Best Practices

1. **Organize documents logically**: Use clear markdown headers to structure your documents. The system chunks documents based on headers.

2. **Keep chunks focused**: Each section should cover a single topic for better retrieval accuracy.

3. **Use descriptive headers**: Headers help the system understand the content of each chunk.

4. **Test retrieval quality**: Verify that the system retrieves relevant chunks for common user queries.

5. **Monitor embedding model**: Ensure your embedding model is appropriate for your document content and user queries.

## Complete Example

Here's a complete example configuration with a knowledge base:

**Directory structure:**

```text
.
├── config
│   ├── config.yml
│   ├── kb
│   │   └── company_policy.md
│   └── rails
│       └── main.co
```

**config.yml:**

```yaml
models:
  - type: main
    engine: openai
    model: gpt-4

  - type: embeddings
    engine: openai
    model: text-embedding-ada-002

instructions:
  - type: general
    content: |
      You are a helpful HR assistant. Answer questions based on the
      company policy documents provided.

knowledge_base:
  folder: "kb"
```

**kb/company_policy.md:**

```markdown
# Company Policy

## Vacation Policy

All full-time employees receive 20 days of paid vacation per year.
Vacation days accrue monthly at a rate of 1.67 days per month.

## Sick Leave

Employees receive 15 days of paid sick leave per year.
Unused sick days do not carry over to the next year.
```

## Related Resources

- [RAG Getting Started Guide](../../getting-started/7-rag/README.md)
- [Embedding Search Providers](../../user-guides/advanced/embedding-search-providers.md)
- [Model Configuration](../yaml-schema/model-configuration.md)
