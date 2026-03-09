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

The cache is implemented in-memory by the Guardrails toolkit.
If a Guardrails instance is restarted, the contents of the cache will be lost.
This causes high miss-rates due to compulsory or cold-start cache misses.

Guardrails may be operated as a horizontally-scalable service to meet availability and performance Service Level Objectives (SLOs).
A typical deployment has multiple Guardrails nodes running in-parallel behind an API gateway and Load Balancer.
The API Gateway implements authentication and authorization, rate limiting and throttling, and any required protocol translation.
The load-balancer distributes load evenly over nodes in the cluster.
The load-balancer ensures that over time, highly-requested prompts will be stored over all nodes in the cluster.

### Cache Fragmentation

With a default round-robin load balancing strategy, incoming traffic is routed to each node in-turn.
The nodes build their own partial view of traffic, reducing cache hit-rates compared to a single-node deployment.
This effect is called *cache fragmentation* and becomes more pronounced as the number of nodes increases.

Cache fragmentation may be addressed in one of two ways.
1. A stateful load-balancer may be used to inspect the incoming request and route it to the same backend node on every request.
2. A cluster-wide in-memory store may be used to store and read cache entries from all compute-nodes in the cluster. This also helps if any nodes are restarted, since they can pull the cache state on startup.

#### Improving Cache Hit Rates with Stateful Load Balancing

Stateful load balancing strategies route repeated similar requests to the same backend node rather than spreading them evenly.
This increases cache hit-rates.
Two commonly used approaches are:

- **Sticky sessions (session affinity)**: The load balancer routes all requests from the same client or session to the same node.
This is effective when individual users tend to send similar prompts within a session.
The trade-off is that high-traffic users may overwhelm an individual node and lead to performance and availability issues.
In this case, consistent hashing may help.
- **Consistent hashing**: The load balancer hashes a property of the request (such as the request body or a header value) and uses the hash to select a backend node. Requests with identical properties are always routed to the same node. This approach can distribute traffic more evenly than sticky sessions while still improving cache hit rates. However, when nodes are added or removed, some requests are remapped to different nodes, which temporarily reduces hit rates until caches are repopulated.

Both strategies involve trade-offs between cache efficiency and even load distribution.
The right approach depends on your traffic patterns, scaling requirements, and infrastructure capabilities.
Consult your API gateway or load balancer documentation for configuration details.

For general background on these strategies, refer to:

- [NGINX: Using Sticky Sessions](https://docs.nginx.com/nginx/admin-guide/load-balancer/http-load-balancer/#enabling-session-persistence)
- [HAProxy: Load Balancing Algorithms](https://docs.haproxy.org/3.0/configuration.html#4-balance)
- [AWS Elastic Load Balancing: Sticky Sessions](https://docs.aws.amazon.com/elasticloadbalancing/latest/application/sticky-sessions.html)

#### Improving Cache Hit Rates with Cluster Storage

An alternative to stateful load balancing is to use a cluster-wide in-memory store such as [Redis](https://redis.io/) or [Memcached](https://memcached.org/).
Instead of each node maintaining its own isolated cache, all nodes read from and write to a shared store.
This eliminates cache fragmentation entirely as a prompt cached by any node is available to all nodes regardless of how the load balancer routes requests.
A cluster-wide store also improves resilience.
When a node is restarted, it does not start with an empty cache.
Instead, it can load previously cached entries from the shared store and benefits from cache hits immediately.

The trade-off is added infrastructure complexity and a network hop for each cache lookup.
The shared store itself must be highly available and sized to handle the throughput of all nodes.
Consult the documentation for your chosen in-memory store for guidance on clustering, replication, and sizing.

For general background on cluster-wide caching, refer to:

- [Redis: Clustering](https://redis.io/docs/latest/operate/oss_and_stack/management/scaling/)
- [Memcached: Configuration](https://github.com/memcached/memcached/wiki/ConfiguringServer)
- [AWS ElastiCache: Choosing a Cache](https://docs.aws.amazon.com/AmazonElastiCache/latest/dg/SelectEngine.html)
- [Google Cloud Memorystore](https://docs.cloud.google.com/memorystore/docs/redis/memorystore-for-redis-overview)
