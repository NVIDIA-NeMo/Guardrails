# Troubleshooting

This page covers common issues you may encounter while configuring or running the NVIDIA NeMo Guardrails library, along with resolution steps.

::::{admonition} Get Help
:class: tip

If your issue is not listed here, [open an issue](https://github.com/NVIDIA-NeMo/Guardrails/issues) on GitHub.
::::

## Runtime

### Nested AsyncIO Loop

The NVIDIA NeMo Guardrails library is async-first. The core runtime uses async functions. To provide a blocking API, the library invokes async functions inside synchronous code using `asyncio.run`.

Python does not allow nested event loops. In notebooks, async web servers, and other environments that already run an event loop, nested loop behavior can cause runtime errors or unexpected behavior.

Meanwhile, the NVIDIA NeMo Guardrails library makes use of [nest_asyncio](https://github.com/erdewit/nest_asyncio). The patching is applied when the `nemoguardrails` package is loaded the first time.

If you do not need the blocking API, or if the `nest_asyncio` patching causes unexpected problems, disable it before loading `nemoguardrails`:

```console
$ export DISABLE_NEST_ASYNCIO=True
```

Then restart the Python process and retry the application.

## LLM Framework Routing

Starting with 0.22, two controls select whether an engine is handled by the built-in OpenAI-compatible client or by LangChain:

- `NEMOGUARDRAILS_LLM_FRAMEWORK` environment variable. Read once at process startup. Default value `default` (built-in). Set to `langchain` to opt into the LangChain path, or to any name you register with `register_framework(name, instance)` before initialization.
- `nemoguardrails.set_default_framework(name)`. Switches the active selection at runtime. Raises `KeyError` if the name is unknown and is not one of the lazy built-ins (`default`, `langchain`).

Use the environment variable when every model in a deployment uses the same path. Use `set_default_framework` from Python when you switch dynamically (for example in tests or when bootstrapping a custom framework).

For migration recipes, see [Migrating to 0.22](migration/0.22.md). For the engine-by-engine matrix, see [Supported LLMs](about/supported-llms.md#inference-providers).

### Error: No Default `base_url` for Provider

```text
ValueError: No default base_url for provider 'cohere'.
If your endpoint is OpenAI-compatible, set parameters.base_url.
Otherwise, set NEMOGUARDRAILS_LLM_FRAMEWORK=langchain and install
the matching langchain-<provider> package (see migration guide).
```

This error appears when the engine name you used (`cohere` in the example) isn't a built-in OpenAI-compatible engine and you haven't opted into LangChain. The error itself names both fix paths; pick the one that matches your provider. For migration recipes, see [Migrating to 0.22](migration/0.22.md).

### Error: Framework Already Registered

`register_framework` does not allow rebinding. If you see this error, the framework name is already registered in the current process.

Pick a different name. In tests, call the registry-reset hook before re-registering the same name.

### Error: Unknown Framework

The `set_default_framework` call used a name that is not registered and is not one of the lazy built-ins `default` or `langchain`.

Register the framework first, or correct the name:

```python
from nemoguardrails import register_framework, set_default_framework
from my_pkg import MyFramework

register_framework("my-framework", MyFramework())
set_default_framework("my-framework")
```

### Error: Unsupported Parameter on First Call

Starting with 0.22, the built-in client forwards `parameters` from `config.yml` directly to the OpenAI-compatible HTTP request. Keys that LangChain accepted as Python flags (`streaming`, `disable_streaming`, `verbose`, `cache`, `callbacks`, `tags`, `metadata`, `name`, `model_kwargs`) and provider-prefixed credential aliases (`openai_api_base`, `nim_base_url`, `*_api_key`, and others) are not part of the OpenAI wire shape, so the provider rejects them. The library detects recognizable shapes at boot and on the first 400/422 response, and appends a migration hint to the underlying provider error.

Fix the configuration by choosing one path:

- Adapt the configuration to OpenAI-compatible shape. Rename `openai_api_base` to `base_url`, drop LangChain Python flags, and remove provider-prefixed aliases. The migration recipe in [Migrating to 0.22](migration/0.22.md#mixed-shape-configs) covers the common case.
- Keep the 0.21 config. Set `NEMOGUARDRAILS_LLM_FRAMEWORK=langchain` for the process and install LangChain plus the matching upstream provider integration. The legacy field names continue to work when you opt into LangChain.
