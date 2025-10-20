# NeMoGuard Safety Rails with Caching

This example demonstrates how to configure NeMo Guardrails with caching support for multiple NVIDIA NeMoGuard NIMs, including content safety, topic control, and jailbreak detection.

## Features

- **Content Safety Checks**: Validates content against 23 safety categories (input and output)
- **Topic Control**: Ensures conversations stay within allowed topics (input)
- **Jailbreak Detection**: Detects and prevents jailbreak attempts (input)
- **Per-Model Caching**: Each safety model has its own dedicated cache instance
- **Thread Safety**: Fully thread-safe for use in multi-threaded web servers
- **Cache Statistics**: Optional performance monitoring for each model

## Folder Structure

- `config.yml` - Main configuration file with model definitions, rails configuration, and cache settings
- `prompts.yml` - Prompt templates for content safety and topic control checks

## Configuration Overview

### Basic Configuration with Caching

```yaml
models:
  - type: main
    engine: nim
    model: meta/llama-3.3-70b-instruct

  - type: content_safety
    engine: nim
    model: nvidia/llama-3.1-nemoguard-8b-content-safety
    cache:
      enabled: true
      maxsize: 10000
      stats:
        enabled: true

  - type: topic_control
    engine: nim
    model: nvidia/llama-3.1-nemoguard-8b-topic-control
    cache:
      enabled: true
      maxsize: 10000
      stats:
        enabled: true

  - type: jailbreak_detection
    engine: nim
    model: jailbreak_detect
    cache:
      enabled: true
      maxsize: 10000
      stats:
        enabled: true

rails:
  input:
    flows:
      - jailbreak detection model
      - content safety check input $model=content_safety
      - topic safety check input $model=topic_control

  output:
    flows:
      - content safety check output $model=content_safety

  config:
    jailbreak_detection:
      nim_base_url: "https://ai.api.nvidia.com"
      nim_server_endpoint: "/v1/security/nvidia/nemoguard-jailbreak-detect"
      api_key_env_var: NVIDIA_API_KEY
```

## NeMoGuard NIMs Used

### 1. Content Safety (`nvidia/llama-3.1-nemoguard-8b-content-safety`)

Checks for unsafe content across 23 safety categories including violence, hate speech, sexual content, and more.

**Cache Configuration:**

```yaml
- type: content_safety
  engine: nim
  model: nvidia/llama-3.1-nemoguard-8b-content-safety
  cache:
    enabled: true
    maxsize: 10000
    stats:
      enabled: true
```

### 2. Topic Control (`nvidia/llama-3.1-nemoguard-8b-topic-control`)

Ensures conversations stay within allowed topics and prevents topic drift.

**Cache Configuration:**

```yaml
- type: topic_control
  engine: nim
  model: nvidia/llama-3.1-nemoguard-8b-topic-control
  cache:
    enabled: true
    maxsize: 10000
    stats:
      enabled: true
```

### 3. Jailbreak Detection (`jailbreak_detect`)

Detects and prevents jailbreak attempts that try to bypass safety measures.

**IMPORTANT**: For jailbreak detection caching to work, the `type` and `model` **MUST** be set to these exact values:

- `type: jailbreak_detection`
- `model: jailbreak_detect`

**Cache Configuration:**

```yaml
- type: jailbreak_detection
  engine: nim
  model: jailbreak_detect
  cache:
    enabled: true
    maxsize: 10000
    stats:
      enabled: true
```

The actual NIM endpoint is configured separately in the `rails.config` section:

```yaml
rails:
  config:
    jailbreak_detection:
      nim_base_url: "https://ai.api.nvidia.com"
      nim_server_endpoint: "/v1/security/nvidia/nemoguard-jailbreak-detect"
      api_key_env_var: NVIDIA_API_KEY
```

## How It Works

1. **User Input**: When a user sends a message, it goes through multiple safety checks:
   - Jailbreak detection evaluates for manipulation attempts
   - Content safety checks for unsafe content
   - Topic control validates topic adherence

2. **Caching**: Each model has its own cache:
   - First check: API call to NeMoGuard NIM, result cached
   - Subsequent identical inputs: Cache hit, no API call needed

3. **Response Generation**: If all input checks pass, the main model generates a response

4. **Output Check**: The response is checked by content safety before returning to user

## Cache Configuration Options

### Default Behavior (No Caching)

By default, caching is **disabled**. Models without cache configuration will have no caching.

### Enabling Cache

Add cache configuration to any model definition:

```yaml
cache:
  enabled: true      # Enable caching
  maxsize: 10000     # Cache capacity (number of entries)
  stats:
    enabled: true    # Enable statistics tracking
    log_interval: 300.0  # Log stats every 5 minutes (optional)
```

### Cache Configuration Parameters

- **enabled**: `true` to enable caching, `false` to disable
- **maxsize**: Maximum number of entries in the cache (LRU eviction when full)
- **stats.enabled**: Track cache hit/miss rates and performance metrics
- **stats.log_interval**: How often to log statistics (in seconds, optional)

## Architecture

Each NeMoGuard model gets its own dedicated cache instance, providing:

- **Isolated cache management** per model
- **Different cache capacities** for different models
- **Model-specific performance tuning**
- **Thread-safe concurrent access**

This architecture allows you to:

- Set larger caches for frequently-used models
- Disable caching for specific models
- Monitor performance per model

## Thread Safety

The implementation is fully thread-safe:

- **Concurrent Requests**: Safely handles multiple simultaneous safety checks
- **Efficient Locking**: Uses RLock for minimal performance impact
- **Atomic Operations**: Prevents duplicate LLM calls for the same content

Suitable for:

- Multi-threaded web servers (FastAPI, Flask, Django)
- Concurrent request processing
- High-traffic applications

## Running the Example

```bash
export NVIDIA_API_KEY=your_api_key_here

nemoguardrails server --config examples/configs/nemoguards_cache/
```

## Benefits

1. **Performance**: Avoid redundant NeMoGuard API calls for repeated inputs
2. **Cost Savings**: Reduce API usage significantly
3. **Flexibility**: Enable caching per model based on usage patterns
4. **Clean Architecture**: Each model has its own dedicated cache
5. **Scalability**: Easy to add new models with different caching strategies
6. **Observability**: Cache statistics help monitor effectiveness

## Tips

- Start with moderate cache sizes (5,000-10,000 entries) and adjust based on usage
- Enable stats logging to monitor cache effectiveness
- Jailbreak detection typically has high cache hit rates
- Content safety caching is most effective for chatbots with common queries
- Topic control benefits from caching when topics are well-defined
- Adjust cache sizes independently for each model based on their usage patterns

## Documentation

For more details about NeMoGuard NIMs and deployment options, see:

- [NeMo Guardrails Documentation](https://docs.nvidia.com/nemo/guardrails/index.html)
- [Llama 3.1 NemoGuard 8B ContentSafety NIM](https://docs.nvidia.com/nim/llama-3-1-nemoguard-8b-contentsafety/latest/)
- [Llama 3.1 NemoGuard 8B TopicControl NIM](https://docs.nvidia.com/nim/llama-3-1-nemoguard-8b-topiccontrol/latest/)
- [NemoGuard JailbreakDetect NIM](https://docs.nvidia.com/nim/nemoguard-jailbreakdetect/latest/)
- [NeMoGuard Models on NVIDIA API Catalog](https://build.nvidia.com/search?q=nemoguard)
