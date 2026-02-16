---
title:
  page: LangChain Integration
  nav: LangChain Integration
description: Add guardrails to LangChain chains or use chains as actions inside guardrails configurations.
topics:
- Integration
- AI Safety
tags:
- LangChain
- RunnableRails
- Actions
- LangSmith
content:
  type: how_to
  difficulty: technical_intermediate
  audience:
  - engineer
  - AI Engineer
---

# LangChain Integration

There are three main ways in which you can use the NeMo Guardrails library with LangChain:

1. Add guardrails to a LangChain agent through the middleware hooks (LangChain v1).
2. Add guardrails to a LangChain chain (or `Runnable`).
3. Use a LangChain chain (or `Runnable`) inside a guardrails configuration.

## Add Guardrails to an Agent

For tool-calling agents built with `create_agent`, the `GuardrailsMiddleware` hooks into the agent loop to run safety checks before and after every model call:

```python
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from nemoguardrails.integrations.langchain.middleware import GuardrailsMiddleware

guardrails = GuardrailsMiddleware(config_path="path/to/config")
model = ChatOpenAI(model="gpt-4o")

agent = create_agent(model, tools=[...], middleware=[guardrails])
result = agent.invoke({"messages": [{"role": "user", "content": "Hello!"}]})
```

For more details, check out the [Agent Middleware Guide](agent-middleware.md).

## Add Guardrails to a Chain

You can easily add guardrails to a chain using the `RunnableRails` class:

```python
from nemoguardrails import RailsConfig
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails

# ... initialize `some_chain`

config = RailsConfig.from_path("path/to/config")

# Using LCEL, you first create a RunnableRails instance, and "apply" it using the "|" operator
guardrails = RunnableRails(config)
chain_with_guardrails = guardrails | some_chain

# Alternatively, you can specify the Runnable to wrap
# when creating the RunnableRails instance.
chain_with_guardrails = RunnableRails(config, runnable=some_chain)
```

For more details, check out the [RunnableRails Guide](runnable-rails.md) and the [Chain with Guardrails Guide](chain-with-guardrails/index.md).

## Using a Chain inside Guardrails

To use a chain (or `Runnable`) inside a guardrails configuration, you can register it as an action.

```python
from nemoguardrails import RailsConfig, LLMRails

config = RailsConfig.from_path("path/to/config")
rails = LLMRails(config)

rails.register_action(SampleChainOrRunnable(), "sample_action")
```

Once registered, the chain (or `Runnable`) can be invoked from within a flow:

```colang
define flow
  ...
  $result = execute sample_action
  ...
```

For a complete example, check out the [Runnable as Action Guide](runnable-as-action/index.md).

## LangSmith Integration

The NeMo Guardrails library integrates out-of-the-box with [LangSmith](https://www.langchain.com/langsmith). To start sending trace information to LangSmith, you have to configure the following environment variables:

```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
export LANGCHAIN_API_KEY=<your-api-key>
export LANGCHAIN_PROJECT=<your-project>  # if not specified, defaults to "default"
```

For more details on configuring LangSmith check out the [LangSmith documentation](https://docs.smith.langchain.com/).
