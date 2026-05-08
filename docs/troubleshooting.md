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

Version 0.22 of the NVIDIA NeMo Guardrails library introduces a framework registry that decides whether an engine is handled by the new `DefaultFramework`, which uses `httpx` and does not require LangChain, or by `LangChainFramework`. Two controls select the active framework:

- `NEMOGUARDRAILS_LLM_FRAMEWORK` environment variable. Read once when the registry initializes. Default value `default`. Accepted values: `default`, `langchain`, or any name you register with `register_framework(name, instance)` before initialization.
- `nemoguardrails.set_default_framework(name)`. Changes the active framework at runtime. Raises `KeyError` if the name is unknown and is not one of the lazy built-ins (`default`, `langchain`).

Use the environment variable when every model in a deployment should be resolved through the same framework. Use `set_default_framework` from Python when you switch frameworks dynamically (for example in tests or when bootstrapping a custom framework).

For a detailed walkthrough, including which engines route to each framework, refer to [LLM Framework Routing](configure-rails/yaml-schema/llm-framework-routing.md).

### Error: No Default `base_url` for Provider

```text
ValueError: No default base_url for provider 'cohere'.
Set it explicitly in model parameters: parameters.base_url
```

This error comes from `_resolve_base_url` in `nemoguardrails/llm/frameworks/default.py`. It means the engine name you used, such as `cohere`, is not in `DefaultFramework`'s routing table and does not identify an OpenAI-compatible endpoint.

Fix the configuration by choosing one path:

- For an OpenAI-compatible endpoint, set `parameters.base_url` explicitly.
- For a LangChain-only provider, set `NEMOGUARDRAILS_LLM_FRAMEWORK=langchain` and install the upstream LangChain provider integration. Keep the bare engine name in `config.yml`, such as `engine: cohere`.

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

Version 0.22 forwards `parameters` from `config.yml` directly to the OpenAI-compatible HTTP client when an engine routes through `DefaultFramework`. Keys that LangChain accepted as Python flags (`streaming`, `disable_streaming`, `verbose`, `cache`, `callbacks`, `tags`, `metadata`, `name`, `model_kwargs`) and provider-prefixed credential aliases (`openai_api_base`, `nim_base_url`, `*_api_key`, and others) are not part of the OpenAI wire shape, so the provider rejects them. The NVIDIA NeMo Guardrails library detects recognizable shapes at boot and on the first 400/422 response, and appends a migration hint to the underlying provider error.

Fix the configuration by choosing one path:

- Adapt the configuration to OpenAI-compatible shape. Rename `openai_api_base` to `base_url`, drop LangChain Python flags, and remove provider-prefixed aliases. The migration recipe in [Configure OpenAI-Compatible Self-Hosted and Third-Party Endpoints](configure-rails/yaml-schema/llm-framework-routing.md#configure-openai-compatible-self-hosted-and-third-party-endpoints) covers the common case.
- Keep the 0.21 config. Set `NEMOGUARDRAILS_LLM_FRAMEWORK=langchain` for the process and install LangChain plus the matching upstream provider integration. The legacy field names continue to work under the LangChain framework.
