---
title: Output Rail Streaming
description: Configure how output rails process streamed tokens in chunked mode.
---

# Output Rail Streaming

Configure how output rails are applied to streamed tokens under `rails.output.streaming`.

## Configuration

```yaml
rails:
  output:
    flows:
      - self check output
    streaming:
      enabled: True
      chunk_size: 200
      context_size: 50
      stream_first: True
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | bool | `False` | Must be `True` to use `stream_async()` with output rails |
| `chunk_size` | int | `200` | Number of tokens per chunk that output rails process |
| `context_size` | int | `50` | Tokens carried over between chunks for continuity |
| `stream_first` | bool | `True` | If `True`, tokens stream immediately before output rails are applied |

---

## Parameter Details

### enabled

When output rails are configured and you want to use `stream_async()`, this must be set to `True`.

If not enabled, you receive an error:

```text
stream_async() cannot be used when output rails are configured but
rails.output.streaming.enabled is False. Either set
rails.output.streaming.enabled to True in your configuration, or use
generate_async() instead of stream_async().
```

### chunk_size

The number of tokens buffered before output rails are applied.

- **Larger values**: Fewer rail executions, but higher latency to first output
- **Smaller values**: More rail executions, but faster time-to-first-token

**Default:** `200` tokens

### context_size

The number of tokens from the previous chunk carried over to provide context for the next chunk.

This helps output rails make consistent decisions across chunk boundaries. For example, if a sentence spans two chunks, the context ensures the rail can evaluate the complete sentence.

**Default:** `50` tokens

### stream_first

Controls when tokens are streamed relative to output rail processing:

- `True` (default): Tokens are streamed to the client immediately, then output rails are applied. Provides faster time-to-first-token but rails run after streaming.
- `False`: Output rails are applied to each chunk before streaming. Safer but adds latency.

---

## Requirements

Output rail streaming requires [global streaming](global-streaming.md) to also be enabled:

```yaml
# Both are required
streaming: True

rails:
  output:
    flows:
      - self check output
    streaming:
      enabled: True
```

---

## Usage Examples

### Basic Output Rail Streaming

```yaml
streaming: True

rails:
  output:
    flows:
      - self check output
    streaming:
      enabled: True
      chunk_size: 200
      context_size: 50
```

### Parallel Output Rails With Streaming

For parallel execution of multiple output rails during streaming:

```yaml
streaming: True

rails:
  output:
    parallel: True
    flows:
      - content_safety_check
      - pii_detection
      - hallucination_check
    streaming:
      enabled: True
      chunk_size: 200
      context_size: 50
      stream_first: True
```

### Low-Latency Configuration

For faster time-to-first-token with smaller chunks:

```yaml
streaming: True

rails:
  output:
    flows:
      - self check output
    streaming:
      enabled: True
      chunk_size: 50
      context_size: 20
      stream_first: True
```

### Safety-First Configuration

For maximum safety with rails applied before streaming:

```yaml
streaming: True

rails:
  output:
    flows:
      - content_safety_check
    streaming:
      enabled: True
      chunk_size: 300
      context_size: 75
      stream_first: False
```

---

## How It Works

1. **Token Buffering**: Tokens from the LLM are buffered until `chunk_size` is reached
2. **Context Overlap**: The last `context_size` tokens from the previous chunk are prepended
3. **Rail Execution**: Output rails are applied to the chunk
4. **Streaming**: If `stream_first: True`, tokens stream before rail execution completes

```text
Chunk 1: [token1, token2, ..., token200]
         └─────────────────────────────┘
                    ↓
              Output Rails
                    ↓
              Stream to Client

Chunk 2: [token151, ..., token200, token201, ..., token400]
         └─── context_size ───┘   └─── new tokens ───────┘
                    ↓
              Output Rails
                    ↓
              Stream to Client
```

---

## Python API

```python
from nemoguardrails import LLMRails, RailsConfig

config = RailsConfig.from_path("./config")
rails = LLMRails(config)

messages = [{"role": "user", "content": "Tell me a story"}]

# stream_async() automatically uses output rail streaming when configured
async for chunk in rails.stream_async(messages=messages):
    print(chunk, end="", flush=True)
```

---

## Related Topics

- [Global Streaming](global-streaming.md) - Enable LLM streaming
- [Guardrails Configuration](../guardrails-configuration/index.md) - Configure output rail flows
