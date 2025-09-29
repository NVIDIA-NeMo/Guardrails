# RunnableRails

This guide demonstrates how to integrate NeMo Guardrails into LangChain applications using the `RunnableRails` class. The class implements the full [Runnable Protocol](https://python.langchain.com/docs/concepts/runnables/) with comprehensive support for synchronous and asynchronous operations, streaming, and batch processing.

---

## Overview

`RunnableRails` provides a complete LangChain-native interface that wraps guardrail configurations around LLMs or entire chains. It supports all Runnable methods including `invoke()`, `ainvoke()`, `stream()`, `astream()`, `batch()`, and `abatch()` with full metadata preservation.

---

## Getting Started

To get started, load a guardrail configuration and create a `RunnableRails` instance.

```python
from nemoguardrails import RailsConfig
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails

config = RailsConfig.from_path("path/to/config")
guardrails = RunnableRails(config)
```

To add guardrails around an LLM model inside a chain, wrap the LLM model with a `RunnableRails` instance. For example, `(guardrails | ...)`.

The following is an example of using a prompt, model, and output parser:

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template("tell me a short joke about {topic}")
model = ChatOpenAI()
output_parser = StrOutputParser()

chain = prompt | model | output_parser
```

Add guardrails around the LLM model in the example above with the following code:

```python
chain_with_guardrails = prompt | (guardrails | model) | output_parser
```

```{note}
Using the extra parenthesis is essential to enforce the order in which the `|` (pipe) operator is applied.
```

To add guardrails to an existing chain or any `Runnable`, wrap it similarly.

```python
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

rag_chain_with_guardrails = guardrails | rag_chain
```

You can also use the same approach to add guardrails only around certain parts of your chain. The following example from the [RunnableBranch Documentation](https://python.langchain.com/docs/expression_language/how_to/routing) adds guardrails around the `"anthropic"` and `"general"` branches inside a `RunnableBranch`.

```python
from langchain_core.runnables import RunnableBranch

branch = RunnableBranch(
    (lambda x: "anthropic" in x["topic"].lower(), guardrails | anthropic_chain),
    (lambda x: "langchain" in x["topic"].lower(), langchain_chain),
    guardrails | general_chain,
)
```

In general, you can wrap any part of a runnable chain with guardrails.

```python
chain = runnable_1 | runnable_2 | runnable_3 | runnable_4 | ...
chain_with_guardrails = runnable_1 | (guardrails | (runnable_2 | runnable_3)) | runnable_4 | ...
```

---

## Streaming Support

`RunnableRails` provides full streaming support with both synchronous and asynchronous methods. This enables responsive applications that stream LLM outputs as they are generated.

```python
# Synchronous streaming
for chunk in guardrails.stream("What is machine learning?"):
    print(chunk, end="", flush=True)

# Asynchronous streaming
async def stream_example():
    async for chunk in guardrails.astream("What is machine learning?"):
        print(chunk, end="", flush=True)
```

**Metadata in Streaming**: `RunnableRails` preserves all metadata during streaming, including `response_metadata`, `usage_metadata`, and `additional_kwargs` in `AIMessageChunk` objects.

---

## Batch Processing

`RunnableRails` supports efficient batch processing for multiple inputs. The following example shows how to use the `batch` and `abatch` methods.

```python
inputs = [
    "What is Python?",
    "Explain machine learning",
    "How does AI work?"
]

# Synchronous batch processing
results = guardrails.batch(inputs)

# Asynchronous batch processing
results = await guardrails.abatch(inputs)

# Control concurrency
from langchain_core.runnables import RunnableConfig
config = RunnableConfig(max_concurrency=3)
results = await guardrails.abatch(inputs, config=config)
```

---

## Input/Output Formats

`RunnableRails` intelligently handles various input and output formats with automatic transformation.

### LLM Wrapping Formats

| Input Format | Output Format | Description |
|-------------|---------------|-------------|
| `str` | `AIMessage` | String prompts → AI messages with full metadata |
| `StringPromptValue` | `AIMessage` | Prompt values → AI messages |
| `ChatPromptValue` | `AIMessage` | Chat prompts → AI messages |
| `List[BaseMessage]` | `AIMessage` | Message lists → AI messages |
| `HumanMessage` | `AIMessage` | Human messages → AI messages |

### Chain Wrapping Formats

| Input Format | Output Format | Behavior |
|-------------|---------------|----------|
| `dict` with `input` key | `dict` with `output` key | Dictionary passthrough |
| `dict` with custom key | `dict` with custom key | Configurable via `input_key`/`output_key` |
| `str` | `str` | String passthrough |
| Mixed formats | Intelligently detected | Automatic format detection |

---

## Metadata Preservation

`RunnableRails` maintains complete metadata compatibility with LangChain components. All `AIMessage` responses include the following:

- **`response_metadata`**: Token usage, model info, finish reasons.
- **`usage_metadata`**: Input/output token counts, total tokens.
- **`additional_kwargs`**: Custom fields from the LLM provider.
- **`id`**: Unique message identifiers.
- **`tool_calls`**: Tool call information when applicable.

```python
result = guardrails.invoke("Hello world")
print(result.response_metadata)  # {'token_usage': {...}, 'model_name': '...', ...}
print(result.usage_metadata)     # {'input_tokens': 10, 'output_tokens': 5, ...}
print(result.additional_kwargs)  # Provider-specific fields
print(result.id)                 # 'msg_abc123...'
```

This ensures seamless integration with LangChain components that depend on message metadata.

---

## Configuration Options

### Passthrough Mode

The role of a guardrail configuration is to validate user input, check LLM output, and guide the LLM model on how to respond. See the [Configuration Guide](../configuration-guide.md#guardrails-definitions) for more details on the different types of rails.

To achieve this, the guardrail configuration might make additional calls to the LLM or other models/APIs (for example, for fact-checking and content moderation).

By default, when the guardrail configuration decides that it is safe to prompt the LLM, it uses the exact prompt that was provided as the input, such as a string, `StringPromptValue` or `ChatPromptValue`. However, to enforce specific rails, for example, dialog rails, general instructions, the guardrails configuration needs to alter the prompt used to generate the response.

The `passthrough` parameter controls this behavior.

- **`passthrough=True`** (default): Uses the exact input prompt with minimal guardrail intervention.
- **`passthrough=False`**: Allows guardrails to modify prompts for enhanced protection.

```python
# Minimal intervention (required for tool calling)
guardrails = RunnableRails(config, passthrough=True)

# Enhanced guardrails (modifies prompts as needed)
guardrails = RunnableRails(config, passthrough=False)
```

**Tool Calling Requirement**: Set `passthrough=True` for proper tool call handling.

### Custom Input/Output Keys

When you use a guardrail configuration to wrap a chain or a `Runnable`, the input and output are either dictionaries or strings. However, a guardrail configuration always operates on a text input from the user and a text output from the LLM. To achieve this, when dictionaries are used, one of the keys from the input dictionary must be designated as the `"input text"` and one of the keys from the output as the `"output text"`.

By default, these keys are `input` and `output`. To customize these keys, provide the `input_key` and `output_key` parameters when creating the `RunnableRails` instance.

The following examples show how to customize the input and output keys with "question" and "answer" keys.

```python
# Custom keys for specialized chains
guardrails = RunnableRails(
    config,
    input_key="question",    # Default: "input"
    output_key="answer"      # Default: "output"
)

# Usage with RAG chain
rag_chain_with_guardrails = guardrails | rag_chain
```

When a guardrail is triggered and predefined messages must be returned instead of the output from the LLM, only a dictionary with the output key is returned.

```json
{"answer": "I can't assist with that request."}
```

---

## Tool Calling

`RunnableRails` supports LangChain tool calling with full metadata preservation and streaming. Tool calling requires `passthrough=True` to work properly.

The following steps are required to use tool calling with `RunnableRails`:

- Set `passthrough=True` when creating `RunnableRails` instance.
- Use `bind_tools()` to attach tools to your model.
- Handle tool execution in your application logic.

### Basic Tool Setup

```python
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from nemoguardrails import RailsConfig
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails

@tool
def calculator(expression: str) -> str:
    """Evaluates mathematical expressions like '2 + 2' or 'sqrt(16)'."""
    try:
        safe_dict = {'sqrt': __import__('math').sqrt, 'pow': pow, '__builtins__': {}}
        return str(eval(expression, safe_dict))
    except Exception as e:
        return f"Error: {e}"

tools = [calculator]
model = ChatOpenAI(model="gpt-5").bind_tools(tools)
config = RailsConfig.from_path("path/to/config")
guardrails = RunnableRails(config=config, passthrough=True)
guarded_model = guardrails | model
```

### Two-Call Tool Pattern

The standard flow for two-call tool calling is to get tool calls, execute them, and synthesize results.

```python
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

# First call: Get tool calls
messages = [HumanMessage(content="What is 2 + 2?")]
result = guarded_model.invoke(messages)

# Execute tools
tools_by_name = {tool.name: tool for tool in tools}
messages_with_tools = [
    messages[0],
    AIMessage(content=result.content or "", tool_calls=result.tool_calls),
]

for tool_call in result.tool_calls:
    tool_result = tools_by_name[tool_call["name"]].invoke(tool_call["args"])
    messages_with_tools.append(
        ToolMessage(
            content=str(tool_result),
            name=tool_call["name"],
            tool_call_id=tool_call["id"],
        )
    )

# Second call: Synthesize results
final_result = guarded_model.invoke(messages_with_tools)
print(final_result.content)
```

### Single-Call with Pre-processed Messages

Use single-call tool calling when you already have a complete message history with tool results.

```python
messages = [
    HumanMessage(content="What is 2 + 2?"),
    AIMessage(
        content="",
        tool_calls=[
            {
                "name": "calculator",
                "args": {"expression": "2 + 2"},
                "id": "call_001",
                "type": "tool_call",
            }
        ],
    ),
    ToolMessage(
        content="4",
        name="calculator",
        tool_call_id="call_001",
    ),
]

result = guarded_model.invoke(messages)
print(result.content)  # "2 + 2 equals 4."
```

---

## Composition and Chaining

`RunnableRails` integrates with complex LangChain compositions. The following example shows how to use `RunnableRails` with a conditional branching chain.

```python
from langchain_core.runnables import RunnablePassthrough, RunnableBranch
from langchain_core.output_parsers import StrOutputParser

# Complex chain with guardrails
chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | (guardrails | llm)
    | StrOutputParser()
)

# Conditional branching with guardrails
branch = RunnableBranch(
    (lambda x: "technical" in x["topic"], guardrails | technical_chain),
    (lambda x: "creative" in x["topic"], creative_chain),
    guardrails | general_chain,
)
```

**Key Benefits of `RunnableRails`:**

- Maintains full Runnable protocol compatibility.
- Preserves metadata throughout the chain.
- Supports all async/sync operations.
- Works with streaming and batch processing.
