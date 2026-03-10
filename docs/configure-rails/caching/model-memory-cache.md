---
title:
  page: "Memory Model Cache"
  nav: "Memory Model Cache"
description: "Configure in-memory caching to avoid repeated LLM calls for identical prompts using LFU eviction."
keywords: ["nemo guardrails memory cache", "LLM caching", "LFU cache", "prompt caching", "NemoGuard cache"]
topics: ["generative_ai", "developer_tools"]
tags: ["llms", "ai_inference", "performance", "caching"]
content:
  type: how_to
  difficulty: technical_intermediate
  audience: ["engineer"]
---

(model-memory-cache)=

# Memory Model Cache

The NVIDIA NeMo Guardrails library supports an in-memory cache that avoids making LLM calls for repeated prompts. The cache stores user prompts and their corresponding LLM responses. Before making an LLM call, the library checks if the prompt already exists in the cache. If found, the stored response is returned instead of calling the LLM, which improves latency.

In-memory caches are supported for all NemoGuard models: [Content-Safety](https://build.nvidia.com/nvidia/llama-3_1-nemoguard-8b-content-safety), [Topic-Control](https://build.nvidia.com/nvidia/llama-3_1-nemoguard-8b-topic-control), and [Jailbreak Detection](https://build.nvidia.com/nvidia/nemoguard-jailbreak-detect). You can configure each model independently.

The cache uses exact matching (after normalizing whitespace) on LLM prompts with a Least-Frequently-Used (LFU) algorithm for cache evictions. Whitespace normalization collapses consecutive whitespace characters into a single space and trims leading/trailing whitespace.

For observability, cache hits and misses are visible in OpenTelemetry (OTEL) telemetry and stored in logs at a configurable interval.

To get started with caching, refer to the example configurations below. The rest of this page provides details about how the cache works, telemetry, and considerations for enabling caching in a horizontally scalable service.

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

The following configuration file shows the same configuration with caching enabled on the Content Safety, Topic Control, and Jailbreak Detection NemoGuard NIM microservices.
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

When the cache is enabled, the library checks whether a prompt was already sent to the LLM before making each call. This check uses an exact-match lookup after removing whitespace.

If there is a cache hit (that is, the same prompt was sent to the same LLM earlier and the response was stored in the cache), the library returns the response without calling the LLM.

If there is a cache miss (that is, there is no stored LLM response for this prompt in the cache), the library calls the LLM as usual. After the response is received, the library stores it in the cache.

For security reasons, user prompts are not stored directly. After removing whitespace, the library hashes the user prompt using SHA256 and uses the hash as a cache key.

If a new cache record needs to be added and the cache already has `maxsize` entries, the Least-Frequently Used (LFU) algorithm decides which cache record to evict.
The LFU algorithm ensures that the most frequently accessed cache entries remain in the cache, improving the probability of a cache hit.

---

## Telemetry and Logging

The NVIDIA NeMo Guardrails library supports OTEL telemetry to trace client requests through the library and any calls to LLMs or APIs. The cache operation is reflected in these traces:

- **Cache hits** have a far shorter duration with no LLM call
- **Cache misses** include an LLM call

This OTEL telemetry is suited for operational dashboards.

The cache statistics are also logged at a configurable interval if `cache.stats.enabled` is set to `true`. Every `log_interval` seconds, the cache statistics are logged with the format shown below.

The most important metric is the *Hit Rate*, which represents the proportion of LLM calls returned from the cache. If this value remains low, the exact-match approach might not be a good fit for your use case.

These statistics accumulate while the library is running.

```text
Cache Stats :: Size: 23/10000 | Hits: 20 | Misses: 3 | Hit Rate: 87.00% | Evictions: 0 | Puts: 21 | Updates: 4
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

The cache is implemented in-memory by the NVIDIA NeMo Guardrails library.
If a Guardrails library instance is restarted, the contents of the cache are lost.
This causes high miss rates due to compulsory or cold-start cache misses.
This section describes techniques that are out of scope for the library but can improve caching performance in a horizontally scaled backend.

You can operate the library as a horizontally scalable service to meet availability and performance Service Level Objectives (SLOs).
A typical deployment has multiple Guardrails instances running in parallel behind an API gateway and load balancer.
The API gateway implements authentication and authorization, rate limiting and throttling, and any required protocol translation.
The load balancer distributes load evenly over nodes in the cluster.
As a result, highly requested prompts can spread across nodes over time rather than concentrate on a single node.

### Cache Fragmentation

With a default round-robin load balancing strategy, incoming traffic routes to each node in turn.
The nodes build their own partial view of traffic, which reduces cache hit rates compared to a single-node deployment.
This effect is called *cache fragmentation* and becomes more pronounced as the number of nodes increases.

You can address cache fragmentation in one of two ways:

1. Use a stateful load balancer to inspect the incoming request and route it to the same backend node on every request.
2. Use a cluster-wide in-memory store to store and read cache entries from all compute nodes in the cluster. This approach also helps when nodes restart because they can pull the cache state on startup.

#### Improving Cache Hit Rates with Consistent Hashing

Stateful load balancing strategies route repeated identical requests to the same backend node rather than spreading them evenly.
This approach increases cache hit rates and does not require any modifications to the library code.
In consistent hashing, a property of the request, such as the request body or a header value, is hashed and used to select a backend node.
Requests with identical properties always route to the same node, which improves cache hit rates.
However, over time, nodes are added or removed due to scaling, deployments, or node failures.
This causes some requests to remap to different nodes, and hit rates drop until the incoming traffic repopulates cache entries.

- [Consistent Hashing and Random Trees: Distributed Caching Protocols for Relieving Hot Spots on the World Wide Web](https://www.cs.princeton.edu/courses/archive/fall09/cos518/papers/chash.pdf)
- [Web Caching with Consistent Hashing](https://cs.brown.edu/courses/csci2950-u/f09/papers/chash99www.pdf)
- [A Fast, Minimal Memory, Consistent Hash Algorithm](https://arxiv.org/pdf/1406.2294)
- [Maglev: A Fast and Reliable Software Network Load Balancer](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/44824.pdf)

#### Improving Cache Hit Rates with Cluster Storage

An alternative to stateful load balancing is to use a cluster-wide in-memory store such as [Redis](https://redis.io/).
Supporting a cluster-wide in-memory store requires modifications to the library to read and write to the cluster storage.
Instead of each node maintaining its own isolated cache, all nodes read from and write to a shared store.
This approach eliminates cache fragmentation because a prompt cached by any node is available to all nodes regardless of how the load balancer routes requests.
A cluster-wide store also improves resilience.
When a node restarts, it does not start with an empty cache.
Instead, the node can load previously cached entries from the shared store and benefits from cache hits immediately.

The trade-off is added infrastructure complexity and a network hop for each cache lookup.
The shared store must be highly available and sized to handle the throughput of all nodes.
Refer to the documentation for your chosen in-memory store for guidance on clustering, replication, and sizing.

For general background on cluster-wide caching, refer to:

- [Redis: Clustering](https://redis.io/docs/latest/operate/oss_and_stack/management/scaling/)
- [Memcached: Configuration](https://github.com/memcached/memcached/wiki/ConfiguringServer)
- [AWS ElastiCache: Choosing a Cache](https://docs.aws.amazon.com/AmazonElastiCache/latest/dg/SelectEngine.html)
- [Google Cloud Memorystore](https://docs.cloud.google.com/memorystore/docs/redis/memorystore-for-redis-overview)
