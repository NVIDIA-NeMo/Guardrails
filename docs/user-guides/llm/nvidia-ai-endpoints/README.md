# Using LLMs hosted on NVIDIA API Catalog

This guide teaches you how to use NeMo Guardrails with LLMs hosted on NVIDIA API Catalog. It uses the [ABC Bot configuration](../../../../examples/bots/abc) and with the `meta/llama-3.1-70b-instruct` model. Similarly, you can use `meta/llama-3.1-405b-instruct`, `meta/llama-3.1-8b-instruct` or any other [AI Foundation Model](https://build.nvidia.com/explore/discover).

## Prerequisites

Before you begin, ensure you have the following prerequisites in place:

1. Install the [langchain-nvidia-ai-endpoints](https://github.com/langchain-ai/langchain-nvidia/tree/main/libs/ai-endpoints) package:

```bash
pip install -U --quiet langchain-nvidia-ai-endpoints
```

2. An NVIDIA NGC account to access AI Foundation Models. To create a free account go to [NVIDIA NGC website](https://ngc.nvidia.com/).

3. An API key from NVIDIA API Catalog:
   - Generate an API key by navigating to the [AI Foundation Models](https://build.nvidia.com/explore/discover) section on the NVIDIA NGC website, selecting a model with an API endpoint, and generating an API key. You can use this API key for all models available in the NVIDIA API Catalog.
   - Export the NVIDIA API key as an environment variable:

```bash
export NVIDIA_API_KEY=$NVIDIA_API_KEY # Replace with your own key
```

4. If you're running this inside a notebook, patch the AsyncIO loop.

```python
import nest_asyncio

nest_asyncio.apply()
```

## Configuration

To get started, copy the ABC bot configuration into a subdirectory called `config`:

```bash
cp -r ../../../../examples/bots/abc config
```

Update the `models` section of the `config.yml` file to the desired model supported by NVIDIA API Catalog:

```yaml
...
models:
  - type: main
    engine: nvidia_ai_endpoints
    model: meta/llama-3.1-70b-instruct
...
```

## Usage

Load the guardrail configuration:

```python
from nemoguardrails import LLMRails, RailsConfig

config = RailsConfig.from_path("./config")
rails = LLMRails(config)
```

Test that it works:

```python
response = rails.generate(messages=[
{
    "role": "user",
    "content": "How many vacation days do I have per year?"
}])
print(response['content'])
```

```
According to our company policy, you are eligible for 20 days of vacation per year, accrued monthly.
```

You can see that the bot responds correctly.

## Conclusion

In this guide, you learned how to connect a NeMo Guardrails configuration to an NVIDIA API Catalog LLM model. This guide uses `meta/llama-3.1-70b-instruct`, however, you can connect any other model by following the same steps.
