# Using LLMs hosted on Vertex AI

This guide teaches you how to use NeMo Guardrails with LLMs hosted on Vertex AI. It uses the [ABC Bot configuration](https://github.com/NVIDIA/NeMo-Guardrails/tree/develop/examples/bots/abc/README.md) and changes the model to `gemini-1.0-pro`.

This guide assumes you have configured and tested working with Vertex AI models. If not, refer to [this guide](../../advanced/vertexai-setup.md).

## Prerequisites

You need to install the following Python libraries:

1. Install the `google-cloud-aiplatform` and `langchain-google-vertexai` packages:

```bash
pip install --quiet "google-cloud-aiplatform>=1.38.0" langchain-google-vertexai==0.1.0
```

2. Set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable:

```bash
export GOOGLE_APPLICATION_CREDENTIALS=$GOOGLE_APPLICATION_CREDENTIALS # Replace with your own key
```

3. If you're running this inside a notebook, patch the AsyncIO loop.

```python
import nest_asyncio
nest_asyncio.apply()
```

## Configuration

To get started, copy the ABC bot configuration into a subdirectory called `config`:

```bash
cp -r ../../../../examples/bots/abc config
```

Update the `config/config.yml` file to use the `gemini-1.0-pro` model with the `vertexai` provider:

```
...

models:
  - type: main
    engine: vertexai
    model: gemini-1.0-pro

...
```

Load the guardrails configuration:

```python
from nemoguardrails import RailsConfig
from nemoguardrails import LLMRails

config = RailsConfig.from_path("./config")
rails = LLMRails(config)
```

Test that it works:

```python
response = rails.generate(messages=[{
    "role": "user",
    "content": "Hi! How are you?"
}])
print(response)
```

```yaml
{'role': 'assistant', 'content': "I'm doing great! Thank you for asking. I'm here to help you with any questions you may have about the ABC Company."}
```

You can see that the bot responds correctly. To see in more detail what LLM calls have been made, you can use the `print_llm_calls_summary` method as follows:

```python
info = rails.explain()
info.print_llm_calls_summary()
```

```
Summary: 5 LLM call(s) took 3.99 seconds .

1. Task `self_check_input` took 0.58 seconds .
2. Task `generate_user_intent` took 1.19 seconds .
3. Task `generate_next_steps` took 0.71 seconds .
4. Task `generate_bot_message` took 0.88 seconds .
5. Task `self_check_output` took 0.63 seconds .
```

## Evaluation

The `gemini-1.0-pro` and `text-bison` models have been evaluated for topical rails, and `gemini-1.0-pro` has also been evaluated as a self-checking model for hallucination and content moderation. Evaluation results can be found [here](../../../evaluation/README.md).

## Conclusion

In this guide, you learned how to connect a NeMo Guardrails configuration to a Vertex AI LLM model. This guide uses `gemini-1.0-pro`, however, you can connect any other model following the same steps.
