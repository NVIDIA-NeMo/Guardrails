# Knowledge Base

By default, an `LLMRails` instance supports using a set of documents as context for generating the bot responses. To include documents as part of your knowledge base, you must place them in the `kb` folder inside your config folder:

```
.
├── config
│   └── kb
│       ├── file_1.md
│       ├── file_2.md
│       └── ...
```

Currently, only the Markdown format is supported.

## Document Structure

Documents in the knowledge base `kb` folder are automatically processed and indexed for retrieval. The system uses the configured embedding model to create vector representations of the document chunks, which are then stored for efficient similarity search.

## Retrieval Process

When a user query is received, the system:

1. Computes embeddings for the user query using the configured embedding model.
2. Performs similarity search against the indexed document chunks.
3. Retrieves the most relevant chunks based on similarity scores.
4. Makes the retrieved chunks available as `$relevant_chunks` in the context.
5. Uses these chunks as additional context when generating the bot response.

## Configuration

The knowledge base functionality is automatically enabled when documents are present in the `kb` folder. The system uses the same embedding model configuration specified in your `config.yml` under the `models` section. For embedding model configuration examples, refer to [](llm-configuration).

<!--
## Retrieval Rails

You can configure retrieval rails to process the retrieved chunks before they are used for response generation. Retrieval rails are triggered after the `retrieve_relevant_chunks` action has finished and the `$relevant_chunks` variable is populated.

```yaml
rails:
  retrieval:
    flows:
      - check retrieval sensitive data
      - filter irrelevant chunks
      - validate chunk quality
```

## Custom Retrieval Actions

You can implement custom retrieval logic by creating actions that modify the `$relevant_chunks` variable. These actions can:

- Filter chunks based on custom criteria
- Re-rank chunks using different algorithms
- Add metadata to chunks
- Combine chunks from multiple sources

Example custom retrieval action:

```python
def custom_retrieval_filter(context):
    """Filter and re-rank retrieved chunks based on custom logic."""
    chunks = context.get("relevant_chunks", [])

    # Apply custom filtering logic
    filtered_chunks = [chunk for chunk in chunks if custom_filter_criteria(chunk)]

    # Re-rank based on custom scoring
    ranked_chunks = sorted(filtered_chunks, key=custom_scoring_function, reverse=True)

    # Update the context with filtered chunks
    context["relevant_chunks"] = ranked_chunks[:5]  # Keep top 5 chunks
```

## Integration with Dialog Flows

The knowledge base integrates with dialog flows. You can reference the retrieved chunks in your Colang flows as follows:

```colang
define flow answer question
  when user asks question
    retrieve relevant chunks
    bot respond with knowledge
      "Based on the available information: {{ $relevant_chunks }}"
```

## Performance Considerations

- **Chunk Size**: Documents are automatically chunked for optimal retrieval. You can adjust chunk size in advanced configurations.
- **Indexing**: Document indexing happens automatically when the configuration is loaded.
- **Caching**: Embeddings are cached to improve performance for repeated queries.
- **Search Parameters**: You can configure similarity thresholds and maximum number of retrieved chunks.

## Advanced Configuration

For advanced use cases, you can:

- Configure custom embedding search providers
- Implement hybrid search (combining vector and keyword search)
- Set up document preprocessing pipelines
- Configure chunk overlap and size parameters

For more details on advanced embedding search configurations, see the [Embedding Search Providers](../advanced/embedding-search-providers.md) guide.
-->
