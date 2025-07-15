# RunnableRails

This guide will teach you how to integrate guardrail configurations built with NeMo Guardrails into your LangChain applications. The examples in this guide will focus on using the [LangChain Expression Language](https://python.langchain.com/docs/expression_language/) (LCEL).

## Overview

NeMo Guardrails provides a LangChain native interface that implements the [Runnable Protocol](https://python.langchain.com/docs/expression_language/interface), through the `RunnableRails` class. To get started, you must first load a guardrail configuration and create a `RunnableRails` instance:

```python
from nemoguardrails import RailsConfig
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails

config = RailsConfig.from_path("path/to/config")
guardrails = RunnableRails(config)
```

To add guardrails around an LLM model inside a chain, you have to "wrap" the LLM model with a `RunnableRails` instance, i.e., `(guardrails | ...)`.

Let's take a typical example using a prompt, a model, and an output parser:

```python
from langchain.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template("tell me a short joke about {topic}")
model = ChatOpenAI()
output_parser = StrOutputParser()

chain = prompt | model | output_parser
```

To add guardrails around the LLM model in the above example:

```python
chain_with_guardrails = prompt | (guardrails | model) | output_parser
```
```{note}
Using the extra parenthesis is essential to enforce the order in which the `|` (pipe) operator is applied.
```

To add guardrails to an existing chain (or any `Runnable`) you must wrap it similarly:

```python
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

rag_chain_with_guardrails = guardrails | rag_chain
```

You can also use the same approach to add guardrails only around certain parts of your chain. The example below (extracted from the [RunnableBranch Documentation](https://python.langchain.com/docs/expression_language/how_to/routing)), adds guardrails around the "anthropic" and "general" branches inside a `RunnableBranch`:

```python
from langchain_core.runnables import RunnableBranch

branch = RunnableBranch(
    (lambda x: "anthropic" in x["topic"].lower(), guardrails | anthropic_chain),
    (lambda x: "langchain" in x["topic"].lower(), langchain_chain),
    guardrails | general_chain,
)
```

In general, you can wrap any part of a runnable chain with guardrails:

```python
chain = runnable_1 | runnable_2 | runnable_3 | runnable_4 | ...
chain_with_guardrails = runnable_1 | (guardrails | (runnable_2 | runnable_3)) | runnable_4 | ...
```

## Input/Output Formats

The supported input/output formats when wrapping an LLM model are:

| Input Format                           | Output Format                   |
|----------------------------------------|---------------------------------|
| Prompt (i.e., `StringPromptValue`)     | Completion string               |
| Chat history (i.e., `ChatPromptValue`) | New message (i.e., `AIMessage`) |

The supported input/output formats when wrapping a chain (or a `Runnable`) are:

| Input Format                | Output Format                |
|-----------------------------|------------------------------|
| Dictionary with `input` key | Dictionary with `output` key |
| Dictionary with `input` key | String output                |
| String input                | Dictionary with `output` key |
| String input                | String output                |

## Prompt Passthrough

The role of a guardrail configuration is to validate the user input, check the LLM output, guide the LLM model on how to respond, etc. (see [Configuration Guide](../configuration-guide.md#guardrails-definitions) for more details on the different types of rails). To achieve this, the guardrail configuration might make additional calls to the LLM or other models/APIs (e.g., for fact-checking and content moderation).

By default, when the guardrail configuration decides that it is safe to prompt the LLM, **it will use the exact prompt that was provided as the input** (i.e., string, `StringPromptValue` or `ChatPromptValue`). However, to enforce specific rails (e.g., dialog rails, general instructions), the guardrails configuration needs to alter the prompt used to generate the response. To enable this behavior, which provides more robust rails, you must set the `passthrough` parameter to `False` when creating the `RunnableRails` instance:

```python
guardrails = RunnableRails(config, passthrough=False)
```

## Input/Output Keys for Chains with Guardrails

When a guardrail configuration is used to wrap a chain (or a `Runnable`) the input and output are either dictionaries or strings. However, a guardrail configuration always operates on a text input from the user and a text output from the LLM. To achieve this, when dicts are used, one of the keys from the input dict must be designated as the "input text" and one of the keys from the output as the "output text". By default, these keys are `input` and `output`. To customize these keys, you must provide the `input_key` and `output_key` parameters when creating the `RunnableRails` instance.

```python
guardrails = RunnableRails(config, input_key="question", output_key="answer")
rag_chain_with_guardrails = guardrails | rag_chain
```

When a guardrail is triggered, and predefined messages must be returned, instead of the output from the LLM, only a dict with the output key is returned:

```json
{
  "answer": "I'm sorry, I can't assist with that"
}
```

## Using Tools

A guardrail configuration can also use tools as part of the dialog rails. The following snippet defines the `Calculator` tool using the `LLMMathChain`:

```python
from langchain.chains import LLMMathChain

tools = []

class CalculatorInput(BaseModel):
    question: str = Field()

llm_math_chain = LLMMathChain(llm=model, verbose=True)
tools.append(
    Tool.from_function(
        func=llm_math_chain.run,
        name="Calculator",
        description="useful for when you need to answer questions about math",
        args_schema=CalculatorInput,
    )
)
```

To make sure that all math questions are answered using this tool, you can create a rail like the one below and include it in your guardrail configuration:

```colang
define user ask math question
  "What is the square root of 7?"
  "What is the formula for the area of a circle?"

define flow
  user ask math question
  $result = execute Calculator(tool_input=$user_message)
  bot respond
```

Finally, you pass the `tools` array to the `RunnableRails` instance:

```python
guardrails = RunnableRails(config, tools=tools)

prompt = ChatPromptTemplate.from_template("{question}")
chain = prompt | (guardrails | model)

print(chain.invoke({"question": "What is 5+5*5/5?"}))
```

## Limitations

The current implementation of the `RunnableRails` interface does not support streaming. This will be addressed in a future release.
