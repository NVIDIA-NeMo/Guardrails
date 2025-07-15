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

When using a task that has "detailed thinking on" enabled, the model will show its reasoning process:

```
{'role': 'assistant', 'content': '<think>\nOkay, the user is asking about NeMo Guardrails. Let me start by recalling what I know. NeMo is part of NVIDIA\'s tools, right? So, Guardrails must be a component related to that. I remember that NVIDIA has been working on AI frameworks and model development. Maybe Guardrails is part of the NeMo toolkit, which is used for building and training neural networks, especially for speech and language processing.\n\nWait, I think Guardrails are safety features or constraints that prevent models from generating harmful or inappropriate content. So, if NeMo Guardrails exist, they probably integrate these safety mechanisms into the model training or inference process. But I need to be precise here. I should check if NeMo Guardrails are specifically designed for their models like the ones in the NGC catalog.\n\nI remember that NVIDIA has LMOps tools, which might include Guardrails. Oh right, they announced RAPIDS Guardrails earlier, which is a library for adding safety features. Maybe NeMo Guardrails are a similar concept but tailored for the NeMo framework. So, they would allow developers to apply filters, classifiers, or rules to ensure the outputs are safe and comply with policies.\n\nBut wait, I should make sure not to confuse it with other guardrails. For example, some models use RLHF (Reinforcement Learning from Human Feedback), but Guardrails might be more about pre-defined rules or modular safety layers. The key points would be that NeMo Guardrails provide tools to manage model outputs, making it easier to deploy safe and responsible AI applications.\n\nAlso, considering the components of NeMo, which include pre-trained models and a framework for research, adding Guardrails would make sense as a feature to support ethical AI practices. They might include things like toxicity detection, fact-checking, or steering models away from sensitive topics.\n\nI should structure the answer to explain what NeMo is, introduce Guardrails as a safety component, mention their purpose, how they work (like modular components, customizable policies), and their benefits. Also, note that it\'s part of NVIDIA\'s efforts to provide tools for responsible AI deployment. Make sure to clarify if they are part of the main NeMo toolkit or an additional module, and maybe mention associated tools like RAPIDS Guardrails if relevant.\n\nWait, but I should verify if NeMo Guardrails are indeed a released feature or part of a specific version. If the information is not up-to-date, I need to state that clearly. The user might be referring to recent updates. Since my knowledge is current up to July 2024, if there were announcements around that time, include them. Otherwise, stick to known information.\n\nSo, putting it all together: NeMo Guardrails are safety features in NVIDIA\'s NeMo toolkit, designed to enforce ethical guidelines and prevent harmful outputs. They offer customizable policies, integration of safety models, and tools for auditing, helping developers deploy responsible AI applications. They might work alongside other NVIDIA tools to streamline the process of adding safety measures without compromising performance.\n</think>\n\nNeMo Guardrails is an open-source toolkit developed by NVIDIA that provides programmable guardrails for Large Language Models (LLMs). These guardrails are designed to make LLM-based applications safer and more reliable by controlling the output of the models in specific ways...'}
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
