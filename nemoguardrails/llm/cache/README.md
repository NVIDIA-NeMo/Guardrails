# Content Safety LLM Call Caching

## Overview

The content safety checks in `actions.py` now use an LFU (Least Frequently Used) cache to improve performance by avoiding redundant LLM calls for identical safety checks.

## Implementation Details

### Cache Configuration

- Per-model caches: Each model gets its own LFU cache instance
- Default capacity: 50,000 entries per model
- Eviction policy: LFU with LRU tiebreaker
- Statistics tracking: Disabled by default (configurable)
- Tracks timestamps: `created_at` and `accessed_at` for each entry
- Cache creation: Automatic when a model is initialized with cache enabled
- Supported model types: Any non-`main` and non-`embeddings` model type (typically content safety models)

### Cached Functions

`content_safety_check_input()` - Caches safety checks for user inputs

### Cache Key Components

The cache key is generated from:

- The rendered prompt (normalized for whitespace)

Since temperature is fixed (1e-20) and stop/max_tokens are derived from the model configuration, they don't need to be part of the cache key.

### How It Works

1. **Before LLM Call**:
   - Generate cache key from request parameters
   - Check if result exists in cache
   - If found, return cached result (cache hit)

2. **After LLM Call**:
   - If not in cache, make the actual LLM call
   - Store the result in cache for future use

### Cache Management

The caching system automatically creates and manages separate caches for each model. Key features:

- **Automatic Creation**: Caches are created when the model is initialized with cache configuration
- **Isolated Storage**: Each model maintains its own cache, preventing cross-model interference
- **Default Settings**: Each cache has 50,000 entry capacity (configurable)
- **Per-Model Configuration**: Cache is configured per model in the YAML configuration

### Statistics and Monitoring

The cache supports detailed statistics tracking and periodic logging for monitoring cache performance:

```yaml
models:
  - type: content_safety
    engine: nim
    model: nvidia/llama-3.1-nemoguard-8b-content-safety
    cache:
      enabled: true
      capacity_per_model: 10000
      store: memory  # Currently only 'memory' is supported
      stats:
        enabled: true      # Enable stats tracking
        log_interval: 60.0 # Log stats every minute
```

**Statistics Features:**

1. **Tracking Only**: Set `stats.enabled: true` with no `log_interval` to track stats without logging
2. **Automatic Logging**: Set both `stats.enabled: true` and `log_interval` for periodic logging

**Statistics Tracked:**

- **Hits**: Number of cache hits (successful lookups)
- **Misses**: Number of cache misses (failed lookups)
- **Hit Rate**: Percentage of requests served from cache
- **Evictions**: Number of items removed due to capacity
- **Puts**: Number of new items added to cache
- **Updates**: Number of existing items updated
- **Current Size**: Number of items currently in cache

**Log Format:**

```
LFU Cache Statistics - Size: 2456/10000 | Hits: 15234 | Misses: 2456 | Hit Rate: 86.11% | Evictions: 0 | Puts: 2456 | Updates: 0
```

**Usage Examples:**

The cache is managed internally by the NeMo Guardrails framework. When you configure a model with caching enabled, the framework automatically:

1. Creates an LFU cache instance for that model
2. Passes the cache to content safety actions via kwargs
3. Tracks statistics if configured
4. Logs statistics at the specified interval

**Configuration Options:**

- `stats.enabled`: Enable/disable statistics tracking (default: false)
- `stats.log_interval`: Seconds between automatic stats logs (None = no logging)

**Notes:**

- Stats logging requires stats tracking to be enabled
- Logs appear at INFO level in the `nemoguardrails.cache.lfu` logger
- Stats are reset when cache is cleared or when `reset_stats()` is called
- Each model maintains independent statistics

### Example Configuration

```yaml
# config.yml
models:
  - type: main
    engine: openai
    model: gpt-4

  - type: content_safety
    engine: nim
    model: nvidia/llama-3.1-nemoguard-8b-content-safety
    cache:
      enabled: true
      capacity_per_model: 50000
      store: memory
      stats:
        enabled: true
        log_interval: 300.0  # Log stats every 5 minutes

rails:
  input:
    flows:
      - content safety check input model="content_safety"
```

### Example Usage

```python
from nemoguardrails import RailsConfig, LLMRails

# The cache is automatically configured based on your YAML config
config = RailsConfig.from_path("./config.yml")
rails = LLMRails(config)

# Content safety checks will be cached automatically
response = await rails.generate_async(
    messages=[{"role": "user", "content": "Hello, how are you?"}]
)
```

### Thread Safety

The content safety caching system is **thread-safe** for single-node deployments:

1. **LFUCache Implementation**:
   - Uses `threading.RLock` for all operations
   - All public methods (`get`, `put`, `size`, `clear`, etc.) are protected by locks
   - Supports atomic `get_or_compute()` operations that prevent duplicate computations

2. **LLMRails Model Initialization**:
   - Thread-safe cache creation during model initialization
   - Ensures only one cache instance per model across all threads

3. **Key Features**:
   - **No Data Corruption**: Concurrent operations maintain data integrity
   - **No Race Conditions**: Proper locking prevents race conditions
   - **Atomic Operations**: `get_or_compute()` ensures expensive computations happen only once
   - **Minimal Lock Contention**: Efficient locking patterns minimize performance impact

4. **Usage in Web Servers**:
   - Safe for use in multi-threaded web servers (FastAPI, Flask, etc.)
   - Handles concurrent requests without issues
   - Each thread sees consistent cache state

**Note**: This implementation is designed for single-node deployments. For distributed systems, consider using external caching solutions like Redis.

### Benefits

1. **Performance**: Eliminates redundant LLM calls for identical inputs
2. **Cost Savings**: Reduces API calls to LLM services
3. **Consistency**: Ensures identical inputs always produce identical outputs
4. **Smart Eviction**: LFU policy keeps frequently checked content in cache
5. **Model Isolation**: Each model has its own cache, preventing interference between different safety models
6. **Statistics Tracking**: Monitor cache performance with hit rates, evictions, and more per model
7. **Timestamp Tracking**: Track when entries were created and last accessed
8. **Efficiency**: LFU eviction algorithm ensures the most useful entries remain in cache
9. **Thread Safety**: Safe for concurrent access in multi-threaded environments

### Example Usage Pattern

```python
# First call - takes ~500ms (LLM API call)
result = await content_safety_check_input(
    llms=llms,
    llm_task_manager=task_manager,
    model_name="content_safety",
    context={"user_message": "Hello world"}
)

# Subsequent identical calls - takes ~1ms (cache hit)
result = await content_safety_check_input(
    llms=llms,
    llm_task_manager=task_manager,
    model_name="content_safety",
    context={"user_message": "Hello world"}
)
```

### Logging

The implementation includes debug logging:

- Cache creation: `"Created cache for model '{model_name}' with capacity {capacity}"`
- Cache hits: `"Content safety cache hit for model '{model_name}'"`
- Cache stores: `"Content safety result cached for model '{model_name}'"`

Enable debug logging to monitor cache behavior:

```python
import logging
logging.getLogger("nemoguardrails.library.content_safety.actions").setLevel(logging.DEBUG)
```
