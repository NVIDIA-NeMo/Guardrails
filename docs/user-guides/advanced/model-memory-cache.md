(model-memory-cache)=

# In-Memory Model Cache

Guardrails supports an in-memory cache to store user-prompts and the LLM response to them.
This can be applied to any model, using the `Model.cache` field

## Example Configuration

Let's walk through an example of adding caching to a Content-Safety Guardrails application. The initial `config.yml` is shown below.

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

The yaml file below shows the same configuration, but this time with caching enabled on the main LLM and Content-Safety Nemoguard model.
The `cache` section controls the caching. The `meta/llama-3.3-70b-instruct` model has a cache with a maximum size of 1,000 entries, while the `nvidia/llama-3.1-nemoguard-8b-content-safety` has a cache maximum size of 10,000 entries.
Both caches have telemetry reporting enabled.

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

  - type: content_safety
    engine: nim
    model: nvidia/llama-3.1-nemoguard-8b-content-safety
    cache:
      enabled: true
      maxsize: 10000
      stats:
        enabled: true
rails:
  input:
    flows:
      - content safety check input $model=content_safety
  output:
    flows:
      - content safety check output $model=content_safety
```


## Least Frequently Used Cache


## Telemetry

## Horizontal scaling and caching
