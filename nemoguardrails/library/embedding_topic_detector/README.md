# Embedding Topic Detector

Embedding-based topic detection for NeMo Guardrails. Blocks off-topic queries using semantic similarity.

## Quick Start

```yaml
rails:
  config:
    embedding_topic_detector:
      embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
      embedding_engine: "SentenceTransformers"
      threshold: 0.5
      top_k: 3
      examples:
        coffee:
          - "how to brew the perfect cup of coffee"
          - "best coffee beans for espresso"

  input:
    flows:
      - embedding topic check
  output:
    flows:
      - embedding topic check output
```

## How It Works

1. Pre-computes embeddings for your example queries (once at startup)
2. Embeds incoming user query
3. Compares against examples using cosine similarity
4. Returns `on_topic: true/false` based on threshold

**On-topic:** "How do I make espresso?" -> similarity 0.85
**Off-topic:** "Who won the Super Bowl?" -> similarity 0.04

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `embedding_model` | Required | Model name (e.g., `all-MiniLM-L6-v2`) |
| `embedding_engine` | Required | Engine (e.g., `SentenceTransformers`) |
| `threshold` | `0.75` | Min similarity to be on-topic (0-1) |
| `top_k` | `3` | Average top-K most similar examples |
| `examples` | Required | Dict of `{category: [example queries]}` |
