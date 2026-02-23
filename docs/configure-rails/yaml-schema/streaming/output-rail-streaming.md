---
title:
  page: Output Rail Streaming Configuration
  nav: Output Rail Streaming
description: Configure how output rails process streamed tokens in chunked mode.
topics:
- Configuration
- Streaming
- Output Rails
tags:
- Streaming
- Output Rails
- chunk_size
- context_size
- stream_first
content:
  type: reference
  difficulty: technical_intermediate
  audience:
  - engineer
  - AI Engineer
---

# Output Rail Streaming

Configure how output rails process streamed tokens under `rails.output.streaming`.

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
| `stream_first` | bool | `True` | If `True`, the client receives tokens before output rails run on the chunk |

---

### Tips for Setting Parameters

#### enabled

When you configure output rails and want to use `stream_async()`, set this to `True`.

If not enabled, you receive an error:

```text
stream_async() cannot be used when output rails are configured but
rails.output.streaming.enabled is False. Either set
rails.output.streaming.enabled to True in your configuration, or use
generate_async() instead of stream_async().
```

#### chunk_size

The number of tokens buffered before output rails run.

- **Larger values**: Fewer rail executions, but higher latency to first output
- **Smaller values**: More rail executions, but faster time-to-first-token

**Default:** `200` tokens

#### context_size

The number of tokens from the previous chunk carried over to provide context for the next chunk.

This helps output rails make consistent decisions across chunk boundaries. For example, if a sentence spans two chunks, the context ensures the rail can evaluate the complete sentence.

**Default:** `50` tokens

#### stream_first

Controls when tokens are streamed relative to output rail processing:

- `True` (default): The client receives each chunk of tokens before output rails process that chunk. This provides faster time-to-first-token, but if a rail blocks the content, the user has already received the tokens. The stream terminates with a JSON error on violation.
- `False`: Output rails process each chunk before the client receives tokens. The user never sees blocked content, but time-to-first-token increases by the rail execution time per chunk.

---

## Requirements

Output rail streaming requires using the `stream_async()` method:

```yaml
rails:
  output:
    flows:
      - self check output
    streaming:
      enabled: True
```

```{note}
The top-level `streaming: True` field is deprecated and no longer required. Use `stream_async()` directly instead.
```

---

## Usage Examples

### Basic Output Rail Streaming

```yaml
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

```{warning}
With `stream_first: True`, the client receives tokens before output rails run. If a rail blocks the content, the user has already received the tokens up to that point. The stream terminates with a JSON error object when it detects a violation.
```

### Safety-First Configuration

For maximum safety with rails applied before streaming:

```yaml
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

1. **Token Buffering**: The system buffers tokens from the LLM until `chunk_size` tokens accumulate.
2. **Streaming or Rail Execution** (depends on `stream_first`):
   - `stream_first: True` (default): The client receives the new tokens immediately, then output rails run on the chunk (including context). If the rails block the content, the stream terminates with a JSON error, while the client receives the tokens up to that point.
   - `stream_first: False`: Output rails run on the chunk first. The client receives the new tokens only if rails pass. If the rails block the content, the client never receives the tokens.
3. **Context Overlap**: The system retains the last `context_size` tokens from the current chunk and prepends them to the next chunk's processing context. This gives rails visibility across chunk boundaries.
4. **Blocking**: If any rail blocks the content, the stream yields a JSON error object (`{"error": {...}}`) and terminates immediately.

### stream_first: True (default)

```text
Buffer fills to chunk_size
         ↓
  Yield new tokens to client (user sees them immediately)
         ↓
  Run output rails on [context + new tokens]
         ↓
  Pass → continue to next chunk
  Block → yield JSON error, terminate stream
```

### stream_first: False

```text
Buffer fills to chunk_size
         ↓
  Run output rails on [context + new tokens]
         ↓
  Pass → yield new tokens to client
  Block → yield JSON error, terminate stream (user never sees blocked content)
```

### Buffer Overlap

The client receives only new tokens. Output rails use the `context_size` tokens solely for processing context:

```text
Chunk 1: rails process [token1 ... token200]
         user receives  [token1 ... token200]

Chunk 2: rails process [token151 ... token200, token201 ... token400]
                        └── context_size ──┘  └── new tokens ───────┘
         user receives  [token201 ... token400]
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
