(model-memory-cache)=

# Memory Model Cache

Guardrails supports an in-memory cache which avoids making LLM calls for repeated prompts. It stores user-prompts and the corresponding LLM response. Prior to making an LLM call, Guardrails first checks if the prompt matches one already in the cache. If the prompt is found in the cache, the stored response is returned from the cache, rather than prompting the LLM. This improves latency.
In-memory caches are supported for all Nemoguard models ([Content-Safety](https://build.nvidia.com/nvidia/llama-3_1-nemoguard-8b-content-safety), [Topic-Control](https://build.nvidia.com/nvidia/llama-3_1-nemoguard-8b-topic-control), and [Jailbreak Detection](https://build.nvidia.com/nvidia/nemoguard-jailbreak-detect)). Each model can be configured independently.
The cache uses exact-matching (after removing whitespace) on LLM prompts with a Least-Frequently-Used (LFU) algorithm for cache evictions.
For observability, cache hits and misses are visible in OTEL telemetry, and stored in logs on a configurable cadence.
To get started with caching, an example configuration is shown below. The rest of the page has a deep-dive into how the cache works, telemetry, and considerations when enabling caching in a horizontally-scalable service.

## Example Configuration

Let's walk through an example of adding caching to a Content-Safety Guardrails application. The initial `config.yml` without caching is shown below.
We are using a [Llama 3.3 70B-Instruct](https://build.nvidia.com/meta/llama-3_3-70b-instruct) main LLM to generate responses. Inputs are checked by [Content-Safety](https://build.nvidia.com/nvidia/llama-3_1-nemoguard-8b-content-safety), [Topic-Control](https://build.nvidia.com/nvidia/llama-3_1-nemoguard-8b-topic-control) and [Jailbreak detection](https://build.nvidia.com/nvidia/nemoguard-jailbreak-detect) models. The LLM response is also checked by the [Content-Safety](https://build.nvidia.com/nvidia/llama-3_1-nemoguard-8b-content-safety) model.
The input rails check the user prompt before sending it to the Main LLM to generate a response. The output rail checks both the user input and Main LLM response to make sure the response is safe.

```yaml
# Content-Safety config.yml (without caching)
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

The yaml file below shows the same configuration, with caching enabled on the Content-Safety, Topic-Control, and Jailbreak detection Nemoguard NIMs. All three caches have a size of 10,000 records. The caches log their statistics every 60 seconds.

```yaml
# Content-Safety config.yml (with caching)
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

## How does the Cache work?

When the cache is enabled, prior to each LLM call we first check to see if we already sent the same prompt to the LLM, and return the response if so. This uses an exact-match lookup, after removing whitespace.
If there's a cache hit (i.e. the same prompt was sent to the same LLM earlier and the response was stored in the cache), then the response can be returned without calling the LLM.
If there's a cache miss (i.e. we don't have a stored LLM response for this prompt in the cache), then the LLM is called as usual. When the response is received, this is stored in the cache.

For security reasons, user prompts are not stored directly. After removing whitespace, the user-prompt is hashed using SHA256 and then used as a cache key.

If a new cache record needs to be added and the cache already has `maxsize` entries, the Least-Frequently Used (LFU) algorithm is used to decide which cache record to evict.
The LFU algorithm ensures that the most frequently accessed cache entries remain in the cache, improving the probability of a cache hit.

## Telemetry and logging

Guardrails supports OTEL telemetry to trace client requests through Guardrails and any calls to LLMs or APIs. The cache operation is reflected in these traces, with cache hits having a far shorter duration and no LLM call and cache misses having an LLM call. This OTEL telemetry is a good fit for operational dashboards.
The cache statistics are also logged on a configurable cadence if `cache.stats.enabled` is set to `true`. Every `log_interval` seconds, the cache statistics are logged with the format below.
The most important metric below is the "Hit Rate", which is the proportion of LLM calls returned from the cache. If this value remains low, the exact-match may not be a good fit for your usecase.
These statistics accumulate for the time Guardrails is running.


```
"LFU Cache Statistics - "
"Size: 0.23453 | "
"Hits: 20 | "
"Misses: 3 | "
"Hit Rate: 87% | "
"Evictions: 0 | "
"Puts: 20 | "
"Updates: 0"
```

These metrics are detailed below:

* Size: The number of LLM calls stored in the cache.
* Hits: The number of cache hits.
* Misses: The number of cache misses.
* Hit Rate: The proportion of calls returned from the cache. This is a float between 1.0 (all calls returned from cache) and 0.0 (all calls sent to LLM)
* Evictions: Number of cache evictions.
* Puts: Number of new cache records stored.
* Updates: Number of existing cache records updated.


## Horizontal scaling and caching

This cache is implemented in-memory on each Guardrails node. When operating as a horizontally-scaled backend-service, there are many Guardrails nodes running behind an API Gateway and load-balancer to distribute traffic and meet availability and performance targets.
The current cache implementation has a separate cache on each node, with no sharing of cache entries between nodes.
Because the load balancer spreads traffic over all Guardrails nodes, requests have to both be stored in cache, with the load balancer directing the same request to the same node.
In practice, frequently-requested user prompts will likely be spread over Guardrails nodes by the load balancer, so the performance impact may ne less significant.
