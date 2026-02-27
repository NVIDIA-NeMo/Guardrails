---
title:
  page: "Topic Control"
  nav: "Topic Control"
description: "Reference for topic control guardrails that ensure conversations stay within predefined subject boundaries."
topics: ["Configuration", "AI Safety"]
tags: ["Rails", "Topic Control", "YAML"]
content:
  type: "Reference"
  difficulty: "Intermediate"
  audience: ["Developer", "AI Engineer"]
---

# Topic Control

The topic safety feature allows you to define and enforce specific conversation rules and boundaries using NVIDIA's Topic Control model. This model helps ensure that conversations stay within predefined topics and follow specified guidelines.

## Usage

To use the topic safety check, you should:

1. Include the topic control model in the models section of your `config.yml` file:

    ```yaml
    models:
      - type: "topic_control"
        engine: nim
        parameters:
          base_url: "http://localhost:8123/v1"
          model_name: "llama-3.1-nemoguard-8b-topic-control"
    ```

2. Include the topic safety check in your rails configuration:

    ```yaml
    rails:
      input:
        flows:
          - topic safety check input $model=topic_control
    ```

3. Define your topic rules in the system prompt. Here's an example prompt that enforces specific conversation boundaries:

    ```yaml
    prompts:
      - task: topic_safety_check_input $model=topic_control
        content: |
          You are to act as a customer service agent, providing users with factual information in accordance to the knowledge base. Your role is to ensure that you respond only to relevant queries and adhere to the following guidelines

          Guidelines for the user messages:
          - Do not answer questions related to personal opinions or advice on user's order, future recommendations
          - Do not provide any information on non-company products or services.
          - Do not answer enquiries unrelated to the company policies.
    ```

The system prompt must end with the topic safety output restriction - `If any of the above conditions are violated, please respond with "off-topic". Otherwise, respond with "on-topic". You must respond with "on-topic" or "off-topic".` This condition is automatically added to the system prompt by the topic safety check input flow. If you want to customize the output restriction, you can do so by modifying the `TOPIC_SAFETY_OUTPUT_RESTRICTION` variable in the [`topic_safety_check_input`](../../../nemoguardrails/library/topic_safety/actions.py) action.

## Customizing Topic Rules

You can customize the topic boundaries by modifying the rules in your prompt. For example, let's add more guidelines specifying additional boundaries:

```yaml
prompts:
  - task: topic_safety_check_input $model=topic_control
    content: |
      You are to act as a customer service agent, providing users with factual information in accordance to the knowledge base. Your role is to ensure that you respond only to relevant queries and adhere to the following guidelines

      Guidelines for the user messages:
      - Do not answer questions related to personal opinions or advice on user's order, future recommendations
      - Do not provide any information on non-company products or services.
      - Do not answer enquiries unrelated to the company policies.
      - Do not answer questions asking for personal details about the agent or its creators.
      - Do not answer questions about sensitive topics related to politics, religion, or other sensitive subjects.
      - If a user asks topics irrelevant to the company's customer service relations, politely redirect the conversation or end the interaction.
      - Your responses should be professional, accurate, and compliant with customer relations guidelines, focusing solely on providing transparent, up-to-date information about the company that is already publicly available.
```

## Implementation Details

The 'topic safety check input' flow uses the [`topic_safety_check_input`](../../../nemoguardrails/library/topic_safety/actions.py) action. The model returns a boolean value indicating whether the user input is on-topic or not. Please refer to the [topic safety example](https://github.com/NVIDIA-NeMo/Guardrails/tree/develop/examples/configs/topic_safety/README.md) for more details.
