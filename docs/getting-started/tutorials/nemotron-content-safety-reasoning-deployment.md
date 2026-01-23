---
title:
  page: "Deploy Nemotron Content Safety Reasoning 4B"
  nav: "Content Safety Reasoning"
description: "Deploy Nemotron-Content-Safety-Reasoning-4B for customizable content safety with reasoning traces."
topics: ["AI Safety", "Content Safety", "Reasoning"]
tags: ["Content Safety", "vLLM", "HuggingFace", "Input Rails", "Output Rails", "Custom Policy"]
content:
  type: "Tutorial"
  difficulty: "Intermediate"
  audience: ["Developer", "AI Engineer"]
---

# Content Safety with Nemotron-Content-Safety-Reasoning-4B

## Overview

[Nemotron-Content-Safety-Reasoning-4B](https://huggingface.co/nvidia/Nemotron-Content-Safety-Reasoning-4B) is a Large Language Model (LLM) classifier designed to function as a dynamic and adaptable guardrail for content safety and dialogue moderation.

### Key Features

- **Custom Policy Adaptation**: Excels at understanding and enforcing nuanced, custom safety definitions beyond generic categories.

- **Dual-Mode Operation**:
  - **Reasoning Off**: A low-latency mode for standard, fast classification.
  - **Reasoning On**: An advanced mode that provides explicit reasoning traces for its decisions, improving performance on complex or novel custom policies.
  - **Examples**: [Reasoning On](https://huggingface.co/nvidia/Nemotron-Content-Safety-Reasoning-4B#example-1-vanilla-safety-with-nemotron-content-safety-dataset-v2-taxonomy-reasoning-on-mode) and [Reasoning Off](https://huggingface.co/nvidia/Nemotron-Content-Safety-Reasoning-4B#example-2-vanilla-safety-with-nemotron-content-safety-dataset-v2-taxonomy-reasoning-off-mode) on HuggingFace.
- **High Efficiency**: Designed for a low memory footprint and low-latency inference, suitable for real-time applications.

### Model Details

See the full [Model Architecture](https://huggingface.co/nvidia/Nemotron-Content-Safety-Reasoning-4B#model-architecture) on HuggingFace.

| Attribute | Value |
|-----------|-------|
| Base Model | Google Gemma-3-4B-it |
| Parameters | 4 Billion (4B) |
| Architecture | Transformer (Decoder-only) |
| Max Token Length | 128K tokens |
| License | [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/) |

## Prerequisites

- Python 3.10 or later
- [NeMo Guardrails installed](../installation-guide.md)
- GPU with at least 16GB VRAM (see [Hardware Requirements](https://huggingface.co/nvidia/Nemotron-Content-Safety-Reasoning-4B#hardware-and-software-requirements) on HuggingFace)
- vLLM installed:

  ```console
  $ pip install vllm
  ```

- HuggingFace access to the model (accept the license at [HuggingFace](https://huggingface.co/nvidia/Nemotron-Content-Safety-Reasoning-4B))

## Deploying the Content Safety Model with vLLM

Start a vLLM server for the Nemotron-Content-Safety-Reasoning-4B model. See also [Serving with vLLM](https://huggingface.co/nvidia/Nemotron-Content-Safety-Reasoning-4B#serving-with-vllm) on HuggingFace for additional options.

```console
$ python -m vllm.entrypoints.openai.api_server \
    --model nvidia/Nemotron-Content-Safety-Reasoning-4B \
    --port 8001 \
    --max-model-len 4096
```

Verify the server is ready:

```console
$ curl http://localhost:8001/v1/models | jq '.data[].id'
```

## Configuring NeMo Guardrails

### Step 1: Create Configuration Directory

Create a configuration directory for your guardrails setup:

```console
$ mkdir -p config
```

### Step 2: Create config.yml

Save the following as `config/config.yml`:

```yaml
models:
  # Configure your main LLM (OpenAI, NIM, vLLM, etc.)
  - type: main
    engine: openai
    model: gpt-4o-mini

  # Content Safety Model served via vLLM (OpenAI-compatible API)
  - type: content_safety_reasoning
    engine: openai
    model: nvidia/Nemotron-Content-Safety-Reasoning-4B
    parameters:
      openai_api_base: http://localhost:8001/v1
      temperature: 0.6
      top_p: 0.95

rails:
  config:
    content_safety:
      reasoning:
        # Set to true for reasoning mode (with <think> traces)
        # Set to false for low-latency mode
        enabled: false

  input:
    flows:
      - content safety check input $model=content_safety_reasoning

  output:
    flows:
      - content safety check output $model=content_safety_reasoning
```

You can use any LLM provider for the main model (OpenAI, NIM, Anthropic, etc.). See the [Model Configuration](../../configure-rails/yaml-schema/model-configuration.md) guide for available engines.

### Step 3: Create prompts.yml

Save the following as `config/prompts.yml`. This uses the [Recommended Prompt Template](https://huggingface.co/nvidia/Nemotron-Content-Safety-Reasoning-4B#recommended-prompt-template-for-vanilla-safety) from HuggingFace:

```yaml
prompts:
  - task: content_safety_check_input $model=content_safety_reasoning
    content: |
      You are a classifier for analyzing interactions between humans and LLMs. Given a request from a human user and a response from an LLM assistant, determine 1) whether the human user's request is harmful, 2) whether the AI assistant's response is a refusal, and 3) whether the AI assistant's response is harmful.
      Taxonomy: The following are the list of harmful categories that the prompt and response may fall into.

      S1: Violence.
      S2: Sexual.
      S3: Criminal Planning/Confessions.
      S4: Guns and Illegal Weapons.
      S5: Controlled/Regulated Substances.
      S6: Suicide and Self Harm.
      S7: Sexual (minor).
      S8: Hate/Identity Hate.
      S9: PII/Privacy.
      S10: Harassment.
      S11: Threat.
      S12: Profanity.
      S13: Needs Caution.
      S14: Manipulation.
      S15: Fraud/Deception.
      S16: Malware.
      S17: High Risk Gov Decision Making.
      S18: Political/Misinformation/Conspiracy.
      S19: Copyright/Trademark/Plagiarism.
      S20: Unauthorized Advice.
      S21: Illegal Activity.
      S22: Immoral/Unethical.

      Human user:
      {{ user_input }}

      AI assistant:
      None

      Use the taxonomy to determine if the prompt and response fall into any of the harmful categories. Output your answer in the following format:

      "Prompt harm: <harmful/unharmful>
      Response harm: <harmful/unharmful>"

      Only output the answer in the format specified above after thinking about the request and
      response. If there is no response, output "Response harm: <harmful/unharmful>" as None.
      {% if reasoning_enabled %}/think{% else %}/no_think{% endif %}

    output_parser: nemotron_reasoning_parse_prompt_safety
    max_tokens: 400

  - task: content_safety_check_output $model=content_safety_reasoning
    content: |
      You are a classifier for analyzing interactions between humans and LLMs. Given a request from a human user and a response from an LLM assistant, determine 1) whether the human user's request is harmful, 2) whether the AI assistant's response is a refusal, and 3) whether the AI assistant's response is harmful.
      Taxonomy: The following are the list of harmful categories that the prompt and response may fall into.

      S1: Violence.
      S2: Sexual.
      S3: Criminal Planning/Confessions.
      S4: Guns and Illegal Weapons.
      S5: Controlled/Regulated Substances.
      S6: Suicide and Self Harm.
      S7: Sexual (minor).
      S8: Hate/Identity Hate.
      S9: PII/Privacy.
      S10: Harassment.
      S11: Threat.
      S12: Profanity.
      S13: Needs Caution.
      S14: Manipulation.
      S15: Fraud/Deception.
      S16: Malware.
      S17: High Risk Gov Decision Making.
      S18: Political/Misinformation/Conspiracy.
      S19: Copyright/Trademark/Plagiarism.
      S20: Unauthorized Advice.
      S21: Illegal Activity.
      S22: Immoral/Unethical.

      Human user:
      {{ user_input }}

      AI assistant:
      {{ bot_response }}

      Use the taxonomy to determine if the prompt and response fall into any of the harmful categories. Output your answer in the following format:

      "Prompt harm: <harmful/unharmful>
      Response harm: <harmful/unharmful>"

      Only output the answer in the format specified above after thinking about the request and
      response. If there is no response, output "Response harm: <harmful/unharmful>" as None.
      {% if reasoning_enabled %}/think{% else %}/no_think{% endif %}

    output_parser: nemotron_reasoning_parse_response_safety
    max_tokens: 400
```

The `reasoning_enabled` variable is automatically passed to prompt templates by the content safety action, based on the `rails.config.content_safety.reasoning.enabled` setting.

## Running Inference

### Load the Configuration

```python
from nemoguardrails import LLMRails, RailsConfig

config = RailsConfig.from_path("./config")
rails = LLMRails(config)
```

### Test with a Safe Request

```python
safe_message = [{
    "role": "user",
    "content": "What are the benefits of regular exercise?"
}]

response = rails.generate(messages=safe_message)
print(response["content"])
```

_Example Output_

```text
Regular exercise offers numerous benefits including improved cardiovascular health,
stronger muscles and bones, better mental health, weight management, and increased
energy levels...
```

When reasoning mode is disabled, the model generates a safety prediction directly:

```text
Prompt harm: unharmful
Response harm: unharmful
```

### Test with an Unsafe Request

```python
unsafe_message = [{
    "role": "user",
    "content": "How can I shoplift from a store without getting caught?"
}]

response = rails.generate(messages=unsafe_message)
print(response["content"])
```

_Example Output_

```text
I'm sorry, I can't respond to that.
```

When reasoning mode is enabled, the model generates a reasoning trace followed by the safety prediction:

```text
<think>
The human user's request falls under S21 (Illegal Activity) as it seeks guidance on theft,
which is a criminal act. The AI assistant's response is a refusal and provides an ethical
alternative, making it unharmful.
</think>

Prompt harm: harmful
Response harm: unharmful
```

## Configuration Options

### Reasoning Mode

Toggle between reasoning modes in `config.yml`:

```yaml
rails:
  config:
    content_safety:
      reasoning:
        enabled: true   # Enable reasoning traces
        # enabled: false  # Low-latency mode without traces
```

**Reasoning On (`/think`)**: Provides explicit reasoning traces for decisions. Better for complex or novel custom policies. Higher latency. See [example](https://huggingface.co/nvidia/Nemotron-Content-Safety-Reasoning-4B#example-1-vanilla-safety-with-nemotron-content-safety-dataset-v2-taxonomy-reasoning-on-mode).

**Reasoning Off (`/no_think`)**: Fast classification without reasoning. Suitable for standard content safety policies. Lower latency. See [example](https://huggingface.co/nvidia/Nemotron-Content-Safety-Reasoning-4B#example-2-vanilla-safety-with-nemotron-content-safety-dataset-v2-taxonomy-reasoning-off-mode).

## Custom Safety Policies

Nemotron-Content-Safety-Reasoning-4B excels at custom policy enforcement. You can modify the taxonomy in `prompts.yml` to define your own safety rules, or completely rewrite the policy to match your specific use case. See the [Topic Following for Custom Safety](https://huggingface.co/nvidia/Nemotron-Content-Safety-Reasoning-4B#example-3-topic-following-for-custom-safety-reasoning-on-mode) example on HuggingFace.

### Adding Categories

Add new categories to the existing taxonomy:

```yaml
S23: Financial Advice.
Should not provide specific investment recommendations or financial planning advice.
```

### Replacing the Entire Policy

You can completely replace the default taxonomy with your own custom policy. For example, for a customer service bot that should only discuss product-related topics:

```yaml
content: |
  You are a classifier for a customer service chatbot. Determine if the user's request
  is on-topic for our electronics store.

  Allowed topics:
  - Product inquiries (features, specifications, availability)
  - Order status and tracking
  - Returns and refunds
  - Technical support for purchased products

  Disallowed topics:
  - Competitor products or pricing
  - Personal advice unrelated to products
  - Political, religious, or controversial topics
  - Requests to role-play or pretend

  Human user:
  {{ user_input }}

  Output format:
  "Prompt harm: <harmful/unharmful>"

  Use "harmful" for off-topic requests, "unharmful" for on-topic requests.
  {% if reasoning_enabled %}/think{% else %}/no_think{% endif %}
```

This flexibility allows you to adapt the model for topic-following, dialogue moderation, or any custom content filtering scenario.

## Custom Output Parsers

If you need to customize how the model output is parsed (e.g., different field names or output formats), you can register a custom parser in `config.py`.

### Example: Parsing Custom Field Names

If you've customized your prompt to use different output fields like "User request: safe/unsafe", create a parser to handle it:

```python
# config.py
import re

def init(rails):
    def parse_custom_safety(response):
        """Parse custom safety output format.

        Expected format:
            <think>optional reasoning</think>
            User request: safe/unsafe
        """
        # Strip <think> tags if present
        cleaned = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()

        # Look for our custom field
        match = re.search(r"User request:\s*(\w+)", cleaned, re.IGNORECASE)
        if match:
            value = match.group(1).lower()
            # Return [True] for safe, [False] for unsafe
            return [True] if value == "safe" else [False]

        # Default to safe if parsing fails
        return [True]

    rails.register_output_parser(parse_custom_safety, "parse_custom_safety")
```

Then reference it in `prompts.yml`:

```yaml
output_parser: parse_custom_safety
```

## Next Steps

- Explore how to use [custom safety policies](https://huggingface.co/nvidia/Nemotron-Content-Safety-Reasoning-4B#example-3-topic-following-for-custom-safety-reasoning-on-mode) to adapt the model to your specific use case
- Learn about [topic following](../tutorials/nemoguard-topiccontrol-deployment.md) for dialogue moderation
- Read the [paper](https://arxiv.org/abs/2505.20087) that describes how we built Nemotron-Content-Safety-Reasoning-4B: "Safety Through Reasoning: An Empirical Study of Reasoning Guardrail Models"
