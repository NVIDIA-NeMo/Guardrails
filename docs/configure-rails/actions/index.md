---
title:
  page: "Configuring Custom Actions"
  nav: "Custom Actions"
description: "Create Python actions to extend guardrails with external APIs and validation logic."
topics: ["Configuration", "AI Safety"]
tags: ["Actions", "Python", "API", "Validation", "Customization"]
content:
  type: "How-To"
  difficulty: "Intermediate"
  audience: ["Developer", "AI Engineer"]
---

# Configuring Custom Actions

This section describes the `actions.py` file used to define custom Python actions for the NeMo Guardrails library.
By configuring custom actions, you can execute Python code within guardrails flows, extending the library with custom logic, external API integrations, and complex validation.

An `actions.py` file defines custom action functions using the `@action` decorator. A decorator is a callable that takes a function and returns a new function, usually adding behavior or attaching metadata.

```python
from typing import Optional
from nemoguardrails.actions import action

@action()
async def check_custom_policy(context: Optional[dict] = None):
    """Check if the input complies with custom policy."""
    user_message = context.get("last_user_message", "")

    # Custom validation logic
    forbidden_words = ["spam", "phishing"]
    for word in forbidden_words:
        if word in user_message.lower():
            return False

    return True

@action(name="fetch_user_data")
async def get_user_info(user_id: str):
    """Fetch user data from external service."""
    # External API call
    return {"user_id": user_id, "status": "active"}
```

## Configuration Sections

The following sections provide detailed documentation for creating and using custom actions:

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Creating Custom Actions
:link: creating-actions
:link-type: doc

Create custom actions using the @action decorator to integrate Python logic into guardrails flows.
:::

:::{grid-item-card} Built-in Actions
:link: built-in-actions
:link-type: doc

Reference for default actions included in the NeMo Guardrails library for common operations.
:::

:::{grid-item-card} Action Parameters
:link: action-parameters
:link-type: doc

Reference for special parameters like context, llm, and config automatically provided to actions.
:::

:::{grid-item-card} Registering Actions
:link: registering-actions
:link-type: doc

Register custom actions via actions.py, LLMRails.register_action(), or init.py for different use cases.
:::

::::

## File Organization

Custom actions can be organized in two ways:

**Option 1: Single `actions.py` file**

```text
.
├── config
│   ├── config.yml
│   ├── actions.py        # All custom actions
│   └── rails/
│       └── ...
```

**Option 2: `actions/` sub-package**

```text
.
├── config
│   ├── config.yml
│   ├── actions/
│   │   ├── __init__.py
│   │   ├── validation.py
│   │   ├── external_api.py
│   │   └── ...
│   └── rails/
│       └── ...
```

## Quick Example

### 1. Define the Action

Create `config/actions.py`:

```python
from typing import Optional
from nemoguardrails.actions import action

@action(is_system_action=True)
async def check_blocked_terms(context: Optional[dict] = None):
    """Check if bot response contains blocked terms."""
    bot_response = context.get("bot_message", "")

    blocked_terms = ["confidential", "proprietary", "secret"]

    for term in blocked_terms:
        if term in bot_response.lower():
            return True  # Term found, block the response

    return False  # No blocked terms found
```

### 2. Create a Flow Using the Action

Create `config/rails/output.co`:

```colang
define bot refuse to respond
  "I apologize, but I cannot provide that information."

define flow check_output_terms
  $contains_blocked = execute check_blocked_terms

  if $contains_blocked
    bot refuse to respond
    stop
```

### 3. Configure the Rail

Add to `config/config.yml`:

```yaml
rails:
  output:
    flows:
      - check_output_terms
```

For detailed information about each topic, refer to the individual pages linked above.

```{toctree}
:hidden:
:maxdepth: 2

Creating Custom Actions <creating-actions>
Built-in Actions <built-in-actions>
Action Parameters <action-parameters>
Registering Actions <registering-actions>
```
