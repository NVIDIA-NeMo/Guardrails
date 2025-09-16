# Content Safety Configuration

This example demonstrates how to configure content safety rails with NeMo Guardrails, including optional cache persistence.

## Features

- **Input Safety Checks**: Validates user inputs before processing
- **Output Safety Checks**: Ensures bot responses are appropriate
- **Caching**: Reduces redundant API calls with LFU cache
- **Persistence**: Optional cache persistence for resilience across restarts
- **Thread Safety**: Fully thread-safe for use in multi-threaded web servers

## Configuration Overview

The configuration includes:

1. **Main Model**: The primary LLM for conversations (Llama 3.3 70B)
2. **Content Safety Model**: Dedicated model for safety checks (NemoGuard 8B)
3. **Rails**: Input and output safety check flows
4. **Cache Configuration**: Memory cache with optional persistence

## How It Works

1. **User Input**: When a user sends a message, it's checked by the content safety model
2. **Cache Check**: The system first checks if this content was already evaluated (cache hit)
3. **Safety Evaluation**: If not cached, the content safety model evaluates the input
4. **Result Caching**: The safety check result is cached for future use
5. **Response Generation**: If safe, the main model generates a response
6. **Output Check**: The response is also checked for safety before returning to the user

## Cache Persistence

The cache configuration includes:

- **Automatic Saves**: Every 5 minutes (configurable)
- **Shutdown Saves**: Caches are automatically persisted when the application closes
- **Crash Recovery**: Cache reloads from disk on restart
- **Per-Model Storage**: Each model gets its own cache file

To disable persistence, you can either:

1. Set `enabled: false` in the persistence section
2. Remove the `persistence` section entirely
3. Set `interval` to `null` or remove it

Note: Persistence requires both `enabled: true` and a valid `interval` value to be active.

## Thread Safety

The content safety implementation is fully thread-safe:

- **Concurrent Requests**: Safely handles multiple simultaneous safety checks
- **No Data Corruption**: Thread-safe cache operations prevent data corruption
- **Efficient Locking**: Uses RLock for minimal performance impact
- **Atomic Operations**: Prevents duplicate LLM calls for the same content

This makes it suitable for:

- Multi-threaded web servers (FastAPI, Flask, Django)
- Concurrent request processing
- High-traffic applications

### Proper Shutdown

For best results, use one of these patterns:

```python
# Context manager (recommended)
with LLMRails(config) as rails:
    # Your code here
    pass
# Caches automatically persisted on exit

# Or manual cleanup
rails = LLMRails(config)
# Your code here
rails.close()  # Persist caches
```

## Running the Example

```bash
# From the NeMo-Guardrails root directory
nemoguardrails server --config examples/configs/content_safety/
```

## Customization

### Adjust Cache Settings

```yaml
cache:
  enabled: true  # Enable/disable caching
  capacity_per_model: 5000  # Maximum entries per model
  persistence:
    interval: 300.0  # Seconds between saves
    path: ./my_cache.json  # Custom path
```

### Memory-Only Cache

For memory-only caching without persistence:

```yaml
cache:
  enabled: true
  capacity_per_model: 5000
  store: memory
  # No persistence section
```

## Benefits

1. **Performance**: Avoid redundant content safety API calls
2. **Cost Savings**: Reduce API usage for repeated content
3. **Reliability**: Cache survives process restarts
4. **Flexibility**: Easy to enable/disable features as needed
