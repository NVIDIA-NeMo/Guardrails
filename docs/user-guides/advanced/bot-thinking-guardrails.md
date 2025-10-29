# Guardrailing Bot Reasoning Content

Modern reasoning-capable LLMs expose their internal thought process as reasoning traces. These traces reveal how the model arrives at its conclusions, which can be valuable for transparency but may also contain sensitive information or problematic reasoning patterns.

NeMo Guardrails allows you to inspect and control these reasoning traces by extracting them and making them available throughout your guardrails configuration. This enables you to write guardrails that can block responses based on the model's reasoning process, enhance moderation decisions with reasoning context, or monitor reasoning patterns.

```{note}
This guide uses Colang 1.0 syntax. Bot reasoning guardrails are currently supported in Colang 1.0 only.
```

```{important}
The examples in this guide range from minimal toy examples (for understanding concepts) to complete reference implementations. They are designed to teach you how to access and work with `bot_thinking` in different contexts, not as production-ready code to copy-paste. Adapt these patterns to your specific use case with appropriate validation, error handling, and business logic for your application.
```

## Accessing Reasoning Content

When an LLM generates a response with reasoning traces, NeMo Guardrails automatically extracts the reasoning and makes it available in three ways:

### In Colang Flows: `$bot_thinking` Variable

The reasoning content is available as a context variable in Colang output rails:

```colang
define flow check_reasoning
  if $bot_thinking
    $captured_reasoning = $bot_thinking
```

### In Custom Actions: `context.get("bot_thinking")`

When writing Python actions, you can access the reasoning via the context dictionary:

```python
@action(is_system_action=True)
async def check_reasoning(context: Optional[dict] = None):
    bot_thinking = context.get("bot_thinking")
    if bot_thinking and "sensitive" in bot_thinking:
        return False
    return True
```

### In Prompt Templates: `{{ bot_thinking }}`

When rendering prompts for LLM tasks (like `self check output`), the reasoning is available as a Jinja2 template variable:

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

**Important**: Always check if reasoning exists before using it, as not all models provide reasoning traces.

## Guardrailing with Output Rails

Output rails can use the `$bot_thinking` variable to inspect and control responses based on reasoning content.

### Basic Pattern Matching

```colang
define bot refuse to respond
  "I'm sorry, I can't respond to that."

define flow block_sensitive_reasoning
  if $bot_thinking
    if "confidential" in $bot_thinking or "internal only" in $bot_thinking
      bot refuse to respond
      stop
```

Add this flow to your output rails in `config.yml`:

```yaml
rails:
  output:
    flows:
      - block_sensitive_reasoning
```

```{note}
This demonstrates basic pattern matching for learning purposes. Real implementations should use more comprehensive validation and consider edge cases.
```

## Guardrailing with Custom Actions

For complex validation logic or reusable checks across multiple flows, write custom Python actions:

**config/actions.py**:

```python
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

**config/rails.co**:

```colang
define bot refuse to respond
  "I'm sorry, I can't respond to that."

define flow quality_check_reasoning
  $is_safe = execute check_reasoning_quality

  if not $is_safe
    bot refuse to respond
    stop
```

Add to your output rails:

```yaml
rails:
  output:
    flows:
      - quality_check_reasoning
```

## Using Reasoning in Self-Check Output

This is the **complete reference implementation** from [examples/configs/self_check_thinking/](https://github.com/NVIDIA/NeMo-Guardrails/tree/develop/examples/configs/self_check_thinking), showing how `bot_thinking` is used in practice. This pattern provides reasoning traces to your self-check output rail, allowing the moderation LLM to make more informed decisions.

### Configuration

**config.yml**:

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

**prompts.yml**:

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

The `{% if bot_thinking %}` conditional ensures the prompt works with both reasoning and non-reasoning models. When reasoning is available, the self-check LLM can evaluate both the final response and the reasoning process.

**Explore the complete implementation**: You can find the full working configuration in [examples/configs/self_check_thinking/](https://github.com/NVIDIA/NeMo-Guardrails/tree/develop/examples/configs/self_check_thinking) with all files ready to use as a reference for your own implementation.

## See Also

- [LLM Configuration - Using LLMs with Reasoning Traces](../configuration-guide/llm-configuration.md#using-llms-with-reasoning-traces) - API response handling and breaking changes
- [Output Rails](../../getting-started/5-output-rails/README.md) - General guide on output rails
- [Self-Check Output Example](https://github.com/NVIDIA/NeMo-Guardrails/tree/develop/examples/configs/self_check_thinking) - Complete working configuration
- [Custom Actions](../../colang-language-syntax-guide.md#actions) - Guide on writing custom actions
