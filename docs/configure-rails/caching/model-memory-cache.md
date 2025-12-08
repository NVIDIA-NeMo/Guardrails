(model-memory-cache)=

# Memory Model Cache

Guardrails supports an in-memory cache that avoids making LLM calls for repeated prompts. The cache stores user prompts and their corresponding LLM responses. Prior to making an LLM call, Guardrails checks if the prompt already exists in the cache. If found, the stored response is returned instead of calling the LLM, improving latency.

In-memory caches are supported for all Nemoguard models: [Content-Safety](https://build.nvidia.com/nvidia/llama-3_1-nemoguard-8b-content-safety), [Topic-Control](https://build.nvidia.com/nvidia/llama-3_1-nemoguard-8b-topic-control), and [Jailbreak Detection](https://build.nvidia.com/nvidia/nemoguard-jailbreak-detect). Each model can be configured independently.

The cache uses exact matching (after removing whitespace) on LLM prompts with a Least-Frequently-Used (LFU) algorithm for cache evictions.

For observability, cache hits and misses are visible in OpenTelemetry (OTEL) telemetry and stored in logs on a configurable cadence.

To get started with caching, refer to the example configurations below. The rest of this page provides a deep dive into how the cache works, telemetry, and considerations when enabling caching in a horizontally scalable service.

---

## Example Configuration

The following example configurations show how to add caching to a Content-Safety Guardrails application.
The examples use a [Llama 3.3 70B-Instruct](https://build.nvidia.com/meta/llama-3_3-70b-instruct) as the main LLM to generate responses. Inputs are checked by the [Content-Safety](https://build.nvidia.com/nvidia/llama-3_1-nemoguard-8b-content-safety), [Topic-Control](https://build.nvidia.com/nvidia/llama-3_1-nemoguard-8b-topic-control), and [Jailbreak Detection](https://build.nvidia.com/nvidia/nemoguard-jailbreak-detect) models. The LLM response is also checked by the Content-Safety model.
The input rails check the user prompt before sending it to the main LLM to generate a response. The output rail checks both the user input and main LLM response to ensure the response is safe.

### Without Caching

The following `config.yml` file shows the initial configuration without caching.

```yaml
models:
  - type: main
    engine: nim
    model: meta/llama-3.3-70b-instruct

  - type: content_safety
    engine: nim
    model: nvidia/llama-3.1-nemoguard-8b-content-safety

  - type: topic_control
    engine: nim
    model: nvidia/llama-3.1-nemoguard-8b-topic-control

  - type: jailbreak_detection
    engine: nim
    model: jailbreak_detect

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

### With Caching

The following configuration file shows the same configuration with caching enabled on the Content-Safety, Topic-Control, and Jailbreak Detection Nemoguard NIM microservices.
All three caches have a size of 10,000 records and log their statistics every 60 seconds.

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
        log_interval: 60

  - type: topic_control
    engine: nim
    model: nvidia/llama-3.1-nemoguard-8b-topic-control
    cache:
      enabled: true
      maxsize: 10000
      stats:
        enabled: true
        log_interval: 60

  - type: jailbreak_detection
    engine: nim
    model: jailbreak_detect
    cache:
      enabled: true
      maxsize: 10000
      stats:
        enabled: true
        log_interval: 60

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

---

## How the Cache Works

When the cache is enabled, Guardrails checks whether a prompt was already sent to the LLM before making each call. This uses an exact-match lookup after removing whitespace.

If there is a cache hit (that is, the same prompt was sent to the same LLM earlier and the response was stored in the cache), the response is returned without calling the LLM.

If there is a cache miss (that is, there is no stored LLM response for this prompt in the cache), the LLM is called as usual. When the response is received, it is stored in the cache.

For security reasons, user prompts are not stored directly. After removing whitespace, the user prompt is hashed using SHA256 and then used as a cache key.

If a new cache record needs to be added and the cache already has `maxsize` entries, the Least-Frequently Used (LFU) algorithm is used to decide which cache record to evict.
The LFU algorithm ensures that the most frequently accessed cache entries remain in the cache, improving the probability of a cache hit.

---

## Telemetry and Logging

Guardrails supports OTEL telemetry to trace client requests through Guardrails and any calls to LLMs or APIs. The cache operation is reflected in these traces:

- **Cache hits** have a far shorter duration with no LLM call
- **Cache misses** include an LLM call

This OTEL telemetry is suited for operational dashboards.

The cache statistics are also logged on a configurable cadence if `cache.stats.enabled` is set to `true`. Every `log_interval` seconds, the cache statistics are logged with the format shown below.

The most important metric is the *Hit Rate*, which represents the proportion of LLM calls returned from the cache. If this value remains low, the exact-match approach might not be a good fit for your use case.

These statistics accumulate while Guardrails is running.

```text
"LFU Cache Statistics - "
"Size: 23/10000 | "
"Hits: 20 | "
"Misses: 3 | "
"Hit Rate: 87% | "
"Evictions: 0 | "
"Puts: 21 | "
"Updates: 4"
```

The following list describes the metrics included in the cache statistics:

- **Size**: The number of LLM calls stored in the cache.
- **Hits**: The number of cache hits.
- **Misses**: The number of cache misses.
- **Hit Rate**: The proportion of calls returned from the cache. This is a float between 1.0 (all calls returned from the cache) and 0.0 (all calls sent to the LLM).
- **Evictions**: The number of cache evictions.
- **Puts**: The number of new cache records stored.
- **Updates**: The number of existing cache records updated.

---

## Horizontal Scaling and Caching

This cache is implemented in-memory on each Guardrails node. When operating as a horizontally-scaled backend service, multiple Guardrails nodes run behind an API Gateway and load balancer to distribute traffic and meet availability and performance targets.

The current cache implementation maintains a separate cache on each node without sharing cache entries between nodes. For a cache hit to occur, the following conditions must be met:

1. The request must have been previously sent and stored in a cache.
2. The load balancer must direct the subsequent request to the same node.

In practice, the load balancer spreads traffic across all Guardrails nodes, distributing frequently-requested user prompts across multiple nodes. This reduces cache hit rates in horizontally-scaled deployments compared to single-node deployments.
