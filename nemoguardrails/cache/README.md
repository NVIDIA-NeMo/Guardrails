# Content Safety LLM Call Caching

## Overview

The content safety checks in `actions.py` now use an LFU (Least Frequently Used) cache to improve performance by avoiding redundant LLM calls for identical safety checks. The cache supports optional persistence to disk for resilience across restarts.

## Implementation Details

### Cache Configuration

- Per-model caches: Each model gets its own LFU cache instance
- Default capacity: 50,000 entries per model
- Eviction policy: LFU with LRU tiebreaker
- Statistics tracking: Enabled by default
- Tracks timestamps: `created_at` and `accessed_at` for each entry
- Cache creation: Automatic when a model is first used
- Persistence: Optional periodic save to disk with configurable interval

### Cached Functions

1. `content_safety_check_input()` - Caches safety checks for user inputs

Note: `content_safety_check_output()` does not use caching to ensure fresh evaluation of bot responses.

### Cache Key Components

The cache key is a SHA256 hash of:

- The rendered prompt only (can be a string or list of strings)

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

- **Automatic Creation**: Caches are created on first use for each model
- **Isolated Storage**: Each model maintains its own cache, preventing cross-model interference
- **Default Settings**: Each cache has 50,000 entry capacity with stats tracking enabled

```python
# Internal cache access (for debugging/monitoring):
from nemoguardrails.library.content_safety.actions import _MODEL_CACHES

# View which models have caches
models_with_caches = list(_MODEL_CACHES.keys())

# Get stats for a specific model's cache
if "llama_guard" in _MODEL_CACHES:
    stats = _MODEL_CACHES["llama_guard"].get_stats()
```

### Persistence Configuration

The cache supports optional persistence to disk for resilience across restarts:

```yaml
rails:
  config:
    content_safety:
      cache:
        enabled: true
        capacity_per_model: 5000
        persistence:
          interval: 300.0  # Persist every 5 minutes
          path: ./cache_{model_name}.json  # {model_name} is replaced
```

**Configuration Options:**

- `persistence.interval`: Seconds between automatic saves (None = no persistence)
- `persistence.path`: Where to save cache data (can include `{model_name}` placeholder)

**How Persistence Works:**

1. **Automatic Saves**: Cache checks trigger persistence if interval has passed
2. **On Shutdown**: Caches are automatically persisted when LLMRails is closed or garbage collected
3. **On Restart**: Cache loads from disk if persistence file exists
4. **Preserves State**: Frequencies and access patterns are maintained
5. **Per-Model Files**: Each model gets its own persistence file

**Manual Persistence:**

```python
# Force immediate persistence of all caches
content_safety_manager.persist_all_caches()
```

This is useful for graceful shutdown scenarios.

**Notes on Persistence:**

- Persistence only works with "memory" store type
- Cache files are JSON format for easy inspection and debugging
- Set `persistence.interval` to None to disable persistence
- The cache automatically persists on each check if the interval has passed

### Statistics and Monitoring

The cache supports detailed statistics tracking and periodic logging for monitoring cache performance:

```yaml
rails:
  config:
    content_safety:
      cache:
        enabled: true
        capacity_per_model: 10000
        stats:
          enabled: true      # Enable stats tracking
          log_interval: 60.0 # Log stats every minute
```

**Statistics Features:**

1. **Tracking Only**: Set `stats.enabled: true` with no `log_interval` to track stats without logging
2. **Automatic Logging**: Set both `stats.enabled: true` and `log_interval` for periodic logging
3. **Manual Logging**: Force immediate stats logging with `cache.log_stats_now()`

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

```python
# Programmatically access stats
if "safety_model" in _MODEL_CACHES:
    cache = _MODEL_CACHES["safety_model"]
    stats = cache.get_stats()
    print(f"Cache hit rate: {stats['hit_rate']:.2%}")

    # Force immediate stats logging
    if cache.supports_stats_logging():
        cache.log_stats_now()
```

**Configuration Options:**

- `stats.enabled`: Enable/disable statistics tracking (default: false)
- `stats.log_interval`: Seconds between automatic stats logs (None = no logging)

**Notes:**

- Stats logging requires stats tracking to be enabled
- Logs appear at INFO level in the `nemoguardrails.cache.lfu` logger
- Stats are reset when cache is cleared or when `reset_stats()` is called
- Each model maintains independent statistics

### Example Configuration Usage

```python
from nemoguardrails import RailsConfig, LLMRails

# Method 1: Using context manager (recommended - ensures cleanup)
config = RailsConfig.from_path("./config.yml")
with LLMRails(config) as rails:
    # Content safety checks will be cached and persisted automatically
    response = await rails.generate_async(
        messages=[{"role": "user", "content": "Hello, how are you?"}]
    )
# Caches are automatically persisted on exit

# Method 2: Manual cleanup
rails = LLMRails(config)
response = await rails.generate_async(
    messages=[{"role": "user", "content": "Hello, how are you?"}]
)
rails.close()  # Manually persist caches

# Note: If neither method is used, caches will still be persisted
# when the object is garbage collected (__del__)
```

### Thread Safety

The content safety caching system is **thread-safe** for single-node deployments:

1. **LFUCache Implementation**:
   - Uses `threading.RLock` for all operations
   - All public methods (`get`, `put`, `size`, `clear`, etc.) are protected by locks
   - Supports atomic `get_or_compute()` operations that prevent duplicate computations

2. **ContentSafetyManager**:
   - Thread-safe cache creation using double-checked locking pattern
   - Ensures only one cache instance per model across all threads
   - Thread-safe persistence operations

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
8. **Resilience**: Cache survives process restarts without losing data when persistence is enabled
9. **Efficiency**: LFU eviction algorithm ensures the most useful entries remain in cache
10. **Thread Safety**: Safe for concurrent access in multi-threaded environments

### Example Usage Pattern

```python
# First call - takes ~500ms (LLM API call)
result = await content_safety_check_input(
    llms=llms,
    llm_task_manager=task_manager,
    model_name="safety_model",
    context={"user_message": "Hello world"}
)

# Subsequent identical calls - takes ~1ms (cache hit)
result = await content_safety_check_input(
    llms=llms,
    llm_task_manager=task_manager,
    model_name="safety_model",
    context={"user_message": "Hello world"}
)
```

### Logging

The implementation includes debug logging:

- Cache creation: `"Created cache for model '{model_name}' with capacity {capacity}"`
- Cache hits: `"Content safety cache hit for model '{model_name}', key: {key[:8]}..."`
- Cache stores: `"Content safety result cached for model '{model_name}', key: {key[:8]}..."`

Enable debug logging to monitor cache behavior:

```python
import logging
logging.getLogger("nemoguardrails.library.content_safety.actions").setLevel(logging.DEBUG)
```
