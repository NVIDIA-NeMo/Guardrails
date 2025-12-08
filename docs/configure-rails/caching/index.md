# Caching Instructions and Prompts

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Memory Model Cache
:link: model-memory-cache
:link-type: doc

Guardrails supports an in-memory cache that avoids making LLM calls for repeated prompts. The cache stores user prompts and their corresponding LLM responses. Prior to making an LLM call,...
:::

:::{grid-item-card} KV Cache Reuse for NemoGuard NIM
:link: kv-cache-reuse
:link-type: doc

When you configure NeMo Guardrails to call NemoGuard NIMs in response to a client request, every NIM call interjecting the input and response adds to the inference latency. The application LLM can...
:::

::::

```{toctree}
:maxdepth: 1
:hidden:

Caching Instructions <model-memory-cache.md>
KV Cache Reuse for LLM NIM <kv-cache-reuse.md>
```
