(model-memory-cache)=

# In-Memory Model Cache

Guardrails supports an in-memory cache which avoids making LLM calls for repeated prompts. It stores user-prompts and the corresponding LLM response. Prior to making an LLM call, Guardrails first checks if the prompt matches one already in the cache. If the prompt is found in the cache, the stored response is returned from the cache, rather than prompting the LLM. This improves latency.
In-memory caches are supported for the Main LLM, and all Nemoguard models ([Content-Safety](https://build.nvidia.com/nvidia/llama-3_1-nemoguard-8b-content-safety), [Topic-Control](https://build.nvidia.com/nvidia/llama-3_1-nemoguard-8b-topic-control), and [Jailbreak Detection](https://build.nvidia.com/nvidia/nemoguard-jailbreak-detect)). Each model can be configured independently.
The cache uses exact-matching (after removing whitespace) on LLM prompts with a Least-Frequently-Used (LFU) algorithm for cache evictions.
For observability, cache hits and misses are visible in OTEL telemetry, and stored in logs on a configurable cadence.
To get started with caching, an example configuration is shown below. The rest of the page has a deep-dive into how the cache works, telemetry, and considerations when enabling caching in a horizontally-scalable service.

## Example Configuration

Let's walk through an example of adding caching to a Content-Safety Guardrails application. The initial `config.yml` without caching is shown below.
We are using a [Llama 3.3 70B-Instruct](https://build.nvidia.com/meta/llama-3_3-70b-instruct) main LLM to generate responses, and checking user-input and LLM-response using the [Llama 3.1 Nemoguard 8B Content Safety](https://build.nvidia.com/nvidia/llama-3_1-nemoguard-8b-content-safety) model.
The input rail checks the safety of the user prompt before sending it to the main LLM. The output rail checks both the user input and Main LLM response to make sure the response is safe.

```yaml
# Content-Safety config.yml (without caching)
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

The yaml file below shows the same configuration, with caching enabled on the Main and Content-Safety Nemoguard models.
The Main LLM and Nemoguard Content-Safety caches have maximum sizes of 1,000 and 10,000 respectively.
Both caches are configured to log cache statistics. The Main LLM cache statistics are logged every 60 seconds (or 1 minute), while the Content-Safety cache statistics are logged every 360 seconds (or 5 minutes).

```yaml
# Content-Safety config.yml (with caching)
models:
  - type: main
    engine: nim
    model: meta/llama-3.3-70b-instruct
    cache:
      enabled: true
      maxsize: 1000
      stats:
        enabled: true
        log_interval: 60

  - type: content_safety
    engine: nim
    model: nvidia/llama-3.1-nemoguard-8b-content-safety
    cache:
      enabled: true
      maxsize: 10000
      stats:
        enabled: true
        log_interval: 360

rails:
  input:
    flows:
      - content safety check input $model=content_safety
  output:
    flows:
      - content safety check output $model=content_safety
```

## How does the Cache work?

When the cache is enabled, prior to each LLM call we first check to see if we sent the same prompt to the same LLM. This uses an exact-match lookup, after removing whitespace.
If there's a cache hit (i.e. the same prompt was sent to the same LLM earlier and the response was stored in the cache), then the response can be returned without calling the LLM.
If there's a cache miss (i.e. we don't have a stored LLM response for this prompt in the cache), then the LLM is called as usual. When the response is received, this is stored in the cache.

For security reasons, user prompts are not stored directly. After removing whitespace, the user-prompt is hashed using SHA256 and then used as a cache key.

If a new cache record needs to be added and the cache already has `maxsize` entries, the Least-Frequently Used (LFU) algorithm is used to decide which cache record to evict.
The LFU algorithm ensures that the most frequently accessed cache entries remain in the cache, improving the probability of a cache hit.

## Telemetry and logging

Guardrails supports OTEL telemetry to trace client requests through Guardrails and any calls to LLMs or APIs. The cache operation is reflected in these traces, with cache hits having a far shorter duration and no LLM call and cache misses having an LLM call. This OTEL telemetry is a good fit for operational dashboards.
The cache statistics are also logged on a configurable cadence if `cache.stats.enabled` is set to `true`. Every `log_interval` seconds, the cache statistics are logged with the format below.
The most important metric below is the "Hit Rate", which is the proportion of LLM calls returned from the cache. If this value remains low, the exact-match may not be a good fit for your usecase.
**TODO! Do these reset on every measurement period, or increment forever (rollover concerns?)**


```
# TODO! Replace with measured values
"LFU Cache Statistics - "
"Size: {stats['current_size']}/{stats['maxsize']} | "
"Hits: {stats['hits']} | "
"Misses: {stats['misses']} | "
"Hit Rate: {stats['hit_rate']:.2%} | "
"Evictions: {stats['evictions']} | "
"Puts: {stats['puts']} | "
"Updates: {stats['updates']}"
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
