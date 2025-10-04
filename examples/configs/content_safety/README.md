# Content Safety Configuration

This example demonstrates how to configure content safety rails with NeMo Guardrails, from basic setup to advanced per-model configurations.

## Features

- **Input Safety Checks**: Validates user inputs before processing
- **Output Safety Checks**: Ensures bot responses are appropriate
- **Thread Safety**: Fully thread-safe for use in multi-threaded web servers
- **Per-Model Caching**: Optional caching with configurable settings per model
- **Multiple Models**: Support for different content safety models with different configurations

## Configuration Overview

### Basic Configuration

The simplest configuration uses a single content safety model:

```yaml
models:
  - type: main
    engine: nim
    model: meta/llama-3.3-70b-instruct

  - type: content_safety
    engine: nim
    model: nvidia/llama-3.1-nemoguard-8b-content-safety

rails:
  input:
    flows:
      - content safety check input $model=content_safety
  output:
    flows:
      - content safety check output $model=content_safety
```

### Advanced Configuration with Per-Model Caching

For production environments, you can configure caching per model:

```yaml
models:
  - type: content_safety
    engine: nim
    model: nvidia/llama-3.1-nemoguard-8b-content-safety
    cache:
      enabled: true
      capacity_per_model: 50000  # Larger cache for primary model
      stats:
        enabled: true
        log_interval: 60.0  # Log stats every 60 seconds

  - type: llama_guard
    engine: vllm_openai
    model: meta-llama/Llama-Guard-7b
    cache:
      enabled: true
      capacity_per_model: 25000  # Medium cache
      stats:
        enabled: false  # No stats for this model
```

## How It Works

1. **User Input**: When a user sends a message, it's checked by the content safety model(s)
2. **Safety Evaluation**: The content safety model evaluates the input
3. **Caching** (optional): Results are cached to avoid duplicate API calls
4. **Response Generation**: If safe, the main model generates a response
5. **Output Check**: The response is also checked for safety before returning to the user

## Cache Configuration Options

### Default Behavior (No Caching)

By default, caching is **disabled**. Models without cache configuration will have no caching:

```yaml
models:
  - type: shieldgemma
    engine: google
    model: google/shieldgemma-2b
    # No cache config = no caching (default)
```

### Enabling Cache

Add cache configuration to any model definition:

```yaml
cache:
  enabled: true                # Enable caching
  capacity_per_model: 10000   # Cache capacity (number of entries)
  store: "memory"             # Cache storage type (currently only memory)
  stats:
    enabled: true             # Enable statistics tracking
    log_interval: 300.0       # Log stats every 5 minutes (optional)
```

## Architecture

Each content safety model gets its own dedicated cache instance, providing:

- **Isolated cache management** per model
- **Different cache capacities** for different models
- **Model-specific performance tuning**
- **Thread-safe concurrent access**

## Thread Safety

The content safety implementation is fully thread-safe:

- **Concurrent Requests**: Safely handles multiple simultaneous safety checks
- **Efficient Locking**: Uses RLock for minimal performance impact
- **Atomic Operations**: Prevents duplicate LLM calls for the same content

This makes it suitable for:

- Multi-threaded web servers (FastAPI, Flask, Django)
- Concurrent request processing
- High-traffic applications

## Running the Example

```bash
# From the NeMo-Guardrails root directory
nemoguardrails server --config examples/configs/content_safety/
```

## Benefits

1. **Performance**: Avoid redundant content safety API calls
2. **Cost Savings**: Reduce API usage for repeated content
3. **Flexibility**: Enable caching only for models that benefit from it
4. **Clean Architecture**: Each model has its own dedicated cache
5. **Scalability**: Easy to add new models with different caching strategies

## Tips

- Start with no caching to establish baseline performance
- Enable caching for frequently-used models first
- Use stats logging to monitor cache effectiveness
- Adjust cache capacity based on your usage patterns
- Consider different cache sizes for different models based on their usage
