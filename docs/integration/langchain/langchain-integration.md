# LangChain Integration

There are two main ways in which you can use NeMo Guardrails with LangChain:

1. Add guardrails to a LangChain chain (or `Runnable`).
2. Use a LangChain chain (or `Runnable`) inside a guardrails configuration.

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

For more details, check out the [RunnableRails Guide](runnable-rails.md) and the [Chain with Guardrails Guide](chain-with-guardrails/README.md).

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

For a complete example, check out the [Runnable as Action Guide](runnable-as-action/README.md).

## LangSmith Integration

NeMo Guardrails integrates out-of-the-box with [LangSmith](https://www.langchain.com/langsmith). To start sending trace information to LangSmith, you have to configure the following environment variables:

```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
export LANGCHAIN_API_KEY=<your-api-key>
export LANGCHAIN_PROJECT=<your-project>  # if not specified, defaults to "default"
```

For more details on configuring LangSmith check out the [LangSmith documentation](https://docs.smith.langchain.com/).
