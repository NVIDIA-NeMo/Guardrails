# Guardrailing Bot Reasoning Content

Reasoning-capable large language models (LLMs) expose their internal thought process as reasoning traces. These traces reveal how the model arrives at its conclusions, providing transparency into the decision-making process. However, they may also contain sensitive information or problematic reasoning patterns that need to be monitored and controlled.

The NeMo Guardrails toolkit helps you set up guardrails to inspect and control these reasoning traces by extracting them. With this feature, you can configure guardrails that can block responses based on the model's reasoning process, enhance moderation decisions with reasoning context, or monitor reasoning patterns.

```{note}
This guide uses Colang 1.0 syntax. Colang 1.0 currently supports bot reasoning guardrails only.
```

```{important}
The examples in this guide range from minimal toy examples (for understanding concepts) to complete reference implementations. These examples teach you how to access and work with `bot_thinking` in different contexts, not as production-ready code to copy-paste. Adapt these patterns to your specific use case with appropriate validation, error handling, and business logic for your application.
```

---

## Accessing Reasoning Content

When an LLM generates a response with reasoning traces, the NeMo Guardrails toolkit extracts the reasoning and makes it available through the `bot_thinking` variable. You can use this variable in the following ways.

### In Colang Flows

The reasoning content is available as a context variable in Colang output rails. For example, in `config/rails.co`, you can set up a flow to capture the reasoning content by setting the `$captured_reasoning` variable to `$bot_thinking`.

```{code-block}
define flow check_reasoning
  if $bot_thinking
    $captured_reasoning = $bot_thinking
```

### In Custom Actions

When you write Python action functions in  `config/actions.py`, you can access the reasoning through the context dictionary. For example, the following is an example action function that checks if the reasoning retrieved through `context.get("bot_thinking")` contains the word `"sensitive"`. It returns `False` if the bot reasoning contains the word `"sensitive"`.

```{code-block} python
@action(is_system_action=True)
async def check_reasoning(context: Optional[dict] = None):
    bot_thinking = context.get("bot_thinking")
    if bot_thinking and "sensitive" in bot_thinking:
        return False
    return True
```

### In Prompt Templates

When you render prompts for LLM tasks such as `self check output`, the reasoning is available as a Jinja2 template variable. For example, in `prompts.yml`, you can set up a prompt to check if the reasoning contains the word `"sensitive"` and block the response if it does.

```yaml
prompts:
  - task: self_check_output
    content: |
      Bot message: "{{ bot_response }}"

      {% if bot_thinking %}
      Bot reasoning: "{{ bot_thinking }}"
      {% endif %}

      Should this be blocked (Yes or No)?
```

```{important}
Always check if reasoning exists before using it, as not all models provide reasoning traces.
```

---

## Guardrailing with Output Rails

You can use the `$bot_thinking` variable in output rails to inspect and control responses based on reasoning content.

1. Write a basic pattern matching flow that uses the `$bot_thinking` variable in `config/rails.co` as follows:

    ```{code-block}
    define bot refuse to respond
      "I'm sorry, I can't respond to that."

    define flow block_sensitive_reasoning
      if $bot_thinking
        if "confidential" in $bot_thinking or "internal only" in $bot_thinking
          bot refuse to respond
          stop
    ```

2. Add the flow to your output rails in `config.yml` as follows:

    ```{code-block}
    rails:
      output:
        flows:
          - block_sensitive_reasoning
    ```

```{note}
This demonstrates how to set up a basic pattern matching flow for learning purposes. Production implementations must use more comprehensive validation and consider edge cases.
```

---

## Guardrailing with Custom Actions

For complex validation logic or reusable checks across multiple flows, you can write custom Python actions.
This approach provides better code organization and makes it easier to share validation logic across different guardrails.

1. Write the custom action function in `config/actions.py` as follows:

    ```{code-block} python
    from typing import Optional
    from nemoguardrails.actions import action

    @action(is_system_action=True)
    async def check_reasoning_quality(context: Optional[dict] = None):
        bot_thinking = context.get("bot_thinking")

        if not bot_thinking:
            return True

        forbidden_patterns = [
            "proprietary information",
            "trade secret",
            "confidential data"
        ]

        for pattern in forbidden_patterns:
            if pattern.lower() in bot_thinking.lower():
                return False

        return True
    ```

2. Write the flow that uses the custom action function in `config/rails.co` as follows:

    ```{code-block}
    define bot refuse to respond
      "I'm sorry, I can't respond to that."

    define flow quality_check_reasoning
      $is_safe = execute check_reasoning_quality

      if not $is_safe
        bot refuse to respond
        stop
    ```

3. Add the flow to your output rails in `config.yml`.

    ```{code-block}
    rails:
      output:
        flows:
          - quality_check_reasoning
    ```

---

## Using Reasoning in Self-Check Output

The following steps show how to use `bot_thinking` in a self-check output rail. This pattern provides reasoning traces to your moderation LLM, allowing it to make more informed decisions by evaluating both the response and the reasoning process.

This extends the [self check thinking configuration examples](https://github.com/NVIDIA/NeMo-Guardrails/tree/develop/examples/configs/self_check_thinking) in the NeMo Guardrails toolkit repository.

1. Write the `config.yml` file as follows:

    ```yaml
    models:
      - type: main
        engine: <your_engine>
        model: <your_reasoning_model>
      - type: self_check_output
        model: <your_moderation_model>
        engine: <your_engine>

    rails:
      output:
        flows:
          - self check output
    ```

2. Write the `prompts.yml` file as follows:

    ```yaml
    prompts:
      - task: self_check_output
        content: |
          Your task is to check if the bot message below complies with the company policy.

          Company policy for the bot:
          - messages should not contain any explicit content
          - messages should not contain abusive language or offensive content
          - messages should not contain any harmful content
          - messages should not contain racially insensitive content
          - if a message is a refusal, should be polite

          Bot message: "{{ bot_response }}"

          {% if bot_thinking %}
          Bot thinking/reasoning: "{{ bot_thinking }}"
          {% endif %}

          Question: Should the message be blocked (Yes or No)?
          Answer:
    ```

The `{% if bot_thinking %}` conditional ensures that the prompt works with both reasoning and non-reasoning models. When reasoning is available, the self-check LLM can evaluate both the final response and the reasoning process.

---

## Related Guides

Use the following guides to learn more about the features used in this guide.

- [LLM Configuration - Using LLMs with Reasoning Traces](../configuration-guide/llm-configuration.md#using-llms-with-reasoning-traces): API response handling and breaking changes.
- [Output Rails](../../getting-started/5-output-rails/README.md): General guide on output rails.
- [Self-Check Output Example](https://github.com/NVIDIA/NeMo-Guardrails/tree/develop/examples/configs/self_check_thinking): Complete working configuration example in the NeMo Guardrails toolkit repository.
- [Custom Actions](../../colang-language-syntax-guide.md#actions): Guide on writing custom actions.
