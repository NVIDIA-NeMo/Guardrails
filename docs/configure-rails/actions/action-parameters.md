---
title:
  page: Action Parameters Reference
  nav: Action Parameters
description: Reference for special parameters like context, llm, and config provided to actions.
keywords:
  - nemo guardrails action parameters
  - guardrails context parameter
  - guardrails llm parameter
  - action events parameter
topics:
  - generative_ai
  - developer_tools
tags:
  - llms
  - ai_inference
  - security_for_ai
content:
  type: reference
  difficulty: technical_intermediate
  audience:
    - engineer
---

# Action Parameters

This section describes the special parameters automatically provided to actions by the NeMo Guardrails library.

## Special Parameters

When you include these parameters in your action's function signature, they are automatically populated:

| Parameter | Type | Description |
|-----------|------|-------------|
| `context` | `dict` | Context data available to the action |
| `events` | `List[dict]` | History of events in the conversation |
| `llm` | `BaseLLM` | Access to the LLM instance |
| `config` | `RailsConfig` | The full configuration instance |
| `llm_task_manager` | `LLMTaskManager` | Access to the LLM task manager for prompt rendering |
| `state` | `State` | The runtime state object (Colang 2.x only) |

```{important}
Special parameters are only injected for actions that run locally. When an `actions_server_url` is configured, non-system actions are sent to the remote server and do **not** receive these parameters. To ensure an action always receives special parameters, mark it with `is_system_action=True`. When no actions server is configured, all actions run locally and receive these parameters regardless of `is_system_action`.
```

## The `context` Parameter

The `context` parameter provides access to conversation state and variables:

```python
from typing import Optional
from nemoguardrails.actions import action

@action(is_system_action=True)
async def my_action(context: Optional[dict] = None):
    # Access context variables
    user_message = context.get("last_user_message")
    bot_message = context.get("bot_message")
    relevant_chunks = context.get("relevant_chunks")

    return True
```

### Common Context Variables

| Variable | Description | Availability |
|----------|-------------|--------------|
| `last_user_message` | The most recent user message | Always available after user input |
| `bot_message` | The current bot message (in output rails) | Available in output rails |
| `last_bot_message` | The previous bot message | Available after first bot response |
| `relevant_chunks` | Retrieved knowledge base chunks | Only after `retrieve_relevant_chunks` runs |
| `user_intent` | The canonical user intent | Only after `generate_user_intent` runs |
| `bot_intent` | The canonical bot intent | Only after `generate_next_steps` runs |

### Accessing Custom Context

Custom context variables set in flows are also accessible:

```text
# In a Colang flow
$user_preference = "dark_mode"
execute check_preference
```

```python
@action()
async def check_preference(context: Optional[dict] = None):
    preference = context.get("user_preference")
    return preference == "dark_mode"
```

## The `events` Parameter

The `events` parameter provides the complete event history:

```python
from typing import List, Optional
from nemoguardrails.actions import action

@action()
async def analyze_conversation(events: Optional[List[dict]] = None):
    # Count user messages
    user_messages = [
        e for e in events
        if e.get("type") == "UtteranceUserActionFinished"
    ]

    return {"message_count": len(user_messages)}
```

### Event Types

| Event Type | Description |
|------------|-------------|
| `UtteranceUserActionFinished` | User sent a message |
| `StartUtteranceBotAction` | Bot started responding |
| `UtteranceBotActionFinished` | Bot finished responding |
| `StartInternalSystemAction` | System action started |
| `InternalSystemActionFinished` | System action completed |
| `UserIntent` | User intent was determined |
| `BotIntent` | Bot intent was determined |

### Event Structure Example

```python
{
    "type": "UtteranceUserActionFinished",
    "uid": "abc123",
    "final_transcript": "Hello, how are you?",
    "action_uid": "action_001",
    "is_success": True
}
```

## The `llm` Parameter

The `llm` parameter provides direct access to the LLM instance:

```python
from typing import Optional
from langchain_core.language_models import BaseLLM
from nemoguardrails.actions import action

@action()
async def custom_llm_call(
    prompt: str,
    llm: Optional[BaseLLM] = None
):
    """Make a custom LLM call."""
    if llm is None:
        return "LLM not available"

    response = await llm.agenerate([prompt])
    return response.generations[0][0].text
```

### Use Cases for LLM Access

- Custom prompt engineering
- Multiple LLM calls within a single action
- Specialized text processing

```python
@action()
async def summarize_and_validate(
    text: str,
    llm: Optional[BaseLLM] = None
):
    """Summarize text and validate the summary."""
    # First call: summarize
    summary_prompt = f"Summarize this text: {text}"
    summary = await llm.agenerate([summary_prompt])
    summary_text = summary.generations[0][0].text

    # Second call: validate
    validation_prompt = f"Is this summary accurate? {summary_text}"
    validation = await llm.agenerate([validation_prompt])

    return {
        "summary": summary_text,
        "validation": validation.generations[0][0].text
    }
```

### Action-Specific LLM

You can register a dedicated LLM for a specific action using the `{action_name}_llm` naming convention. When registered, it overrides the default `llm` parameter for that action:

```python
from nemoguardrails import LLMRails, RailsConfig

config = RailsConfig.from_path("config")
rails = LLMRails(config)

rails.register_action_param("my_custom_action_llm", specialized_llm)
```

When `my_custom_action` runs and requests the `llm` parameter, it receives `specialized_llm` instead of the default LLM.

## The `config` Parameter

The `config` parameter provides access to the full configuration:

```python
from typing import Optional
from nemoguardrails import RailsConfig
from nemoguardrails.actions import action

@action()
async def check_config_setting(config: Optional[RailsConfig] = None):
    """Access configuration settings."""
    # Access model configuration
    models = config.models
    main_model = next(
        (m for m in models if m.type == "main"),
        None
    )

    # Access custom config data
    custom_data = config.custom_data

    return {
        "model_engine": main_model.engine if main_model else None,
        "custom_data": custom_data
    }
```

### Configuration Access Examples

```python
@action()
async def get_active_rails(config: Optional[RailsConfig] = None):
    """Get list of active rails."""
    rails_config = config.rails

    return {
        "input_rails": rails_config.input.flows if rails_config.input else [],
        "output_rails": rails_config.output.flows if rails_config.output else []
    }
```

## The `llm_task_manager` Parameter

The `llm_task_manager` parameter provides access to prompt rendering and LLM task management:

```python
from typing import Optional
from nemoguardrails.actions import action
from nemoguardrails.llm.taskmanager import LLMTaskManager

@action()
async def custom_task(llm_task_manager: Optional[LLMTaskManager] = None):
    """Use the LLM task manager for prompt rendering."""
    pass
```

## Combining Multiple Parameters

You can use multiple special parameters together:

```python
@action(is_system_action=True)
async def advanced_check(
    context: Optional[dict] = None,
    events: Optional[List[dict]] = None,
    llm: Optional[BaseLLM] = None,
    config: Optional[RailsConfig] = None
):
    """Advanced action using multiple special parameters."""
    # Get current message from context
    message = context.get("last_user_message", "")

    # Count previous interactions from events
    interaction_count = len([
        e for e in events
        if e.get("type") == "UtteranceUserActionFinished"
    ])

    # Check config for thresholds
    max_interactions = config.custom_data.get("max_interactions", 100)

    if interaction_count > max_interactions:
        return False

    # Use LLM for complex validation if needed
    if needs_llm_check(message):
        result = await llm.agenerate([f"Is this safe? {message}"])
        return "yes" in result.generations[0][0].text.lower()

    return True
```

## Parameter Type Annotations

Always use proper type annotations for special parameters:

```python
from typing import Optional, List
from langchain_core.language_models import BaseLLM
from nemoguardrails import RailsConfig
from nemoguardrails.actions import action
from nemoguardrails.llm.taskmanager import LLMTaskManager

@action()
async def properly_typed_action(
    # Regular parameters
    query: str,
    limit: int = 10,
    # Special parameters with correct types
    context: Optional[dict] = None,
    events: Optional[List[dict]] = None,
    llm: Optional[BaseLLM] = None,
    config: Optional[RailsConfig] = None,
    llm_task_manager: Optional[LLMTaskManager] = None,
):
    """Action with proper type annotations."""
    pass
```

## Related Topics

- [Registering Actions](registering-actions) - Ways to register actions
- [Built-in Actions](built-in-actions) - Default actions in the library
- [Creating Custom Actions](creating-actions) - Create your own actions
