# Nemotron Message-Based Prompts

This directory contains configurations for using Nemotron models with NeMo Guardrails.

## Message-Based Prompts with Detailed Thinking

NeMo Guardrails implements message-based prompts for Nemotron models with _detailed thinking_ enabled for specific internal tasks:

### Tasks with Detailed Thinking Enabled

The following internal tasks include a `detailed thinking on` system message:

- `generate_bot_message` - When generating the final response
- `generate_value` - When extracting information from user input
- Other complex reasoning tasks like flow generation and continuation

### Tasks without Detailed Thinking

The following tasks use standard system messages without detailed thinking:

- `generate_user_intent` - When detecting user intent
- `generate_next_steps` - When determining what bot actions to take

## Usage

To use Nemotron with NeMo Guardrails:

```python
from nemoguardrails import LLMRails, RailsConfig

# Load the configuration
config = RailsConfig.from_path("examples/configs/nemotron")

# Create the LLMRails instance
rails = LLMRails(config)

# Generate a response
response = rails.generate(messages=[
    {"role": "user", "content": "What is NeMo Guardrails?"}
])
print(response)
```

When using a task that has "detailed thinking on" enabled, the response includes reasoning in a `<think>...</think>` wrapper. The following illustrative example shows the response structure; the reasoning is omitted:

```
{'role': 'assistant',
 'content': '<think>...</think>\n\nThe NeMo Guardrails library adds programmable guardrails to LLM-based applications.'}
```

### Controlling Detailed Thinking in Final Responses

As Nemotron is a hybrid reasoning model, users can toggle the "detailed thinking" feature for final responses similar to how it works on build.nvidia.com as long as you are not using dialog rails:

#### Enabling Detailed Thinking via System Message

To enable detailed thinking in the response, include a system message with "detailed thinking on":

```python
response = rails.generate(messages=[
    {"role": "system", "content": "detailed thinking on"},
    {"role": "user", "content": "How is the weather today?"}
])
```

This will include the model's reasoning process in a `<think>...</think>` wrapper:

```
{'role': 'assistant',
 'content': '<think>\n</think>I\'m sorry, but I don\'t know the weather. I\'m a large language model, I don\'t have access to real-time information or your location. However, I can guide you on how to check the weather! You can check the weather forecast for your area by:...'}
```

#### Standard Mode (No Detailed Thinking)

Without the special system message, the model provides direct responses without showing its reasoning:

```python
response = rails.generate(messages=[
    {"role": "user", "content": "How is the weather today?"}
])
```

Response:

```
{'role': 'assistant',
 'content': 'The weather! Unfortunately, I don\'t have real-time access to current weather conditions or your location. I\'m a large language model...'}
```

To remove the reasoning traces from the internal tasks, you can use the `remove_thinking_traces` configuration option:

For more information see [LLMs with Reasoning Traces](docs/user-guides/configuration-guide.md#using-llms-with-reasoning-traces)

```yaml
remove_thinking_traces: true
```

## Configuration Details

The `config.yml` file sets:

```yaml
models:
  - type: main
    engine: nim
    model: nvidia/llama-3.1-nemotron-ultra-253b-v1

```
