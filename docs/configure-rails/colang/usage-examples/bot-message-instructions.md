---
title: Bot Message Instructions
description: Provide custom instructions to control how the LLM generates bot messages in Colang 1.0 and 2.0.
---

# Bot Message Instructions

You can provide instructions to the LLM on how to generate bot messages. The approach differs between Colang 1.0 and Colang 2.0.

## Overview

````{tab-set}
```{tab-item} Colang 2.0
In Colang 2.0, you use **flow docstrings** (Natural Language Descriptions) to provide instructions to the LLM. These docstrings are included in the prompt when the generation operator (`...`) is invoked.
```

```{tab-item} Colang 1.0
In Colang 1.0, you place a **comment** above a `bot something` statement. The comment is included in the prompt, instructing the LLM on how to generate the message.
```
````

## Formal Greeting Example

The following example instructs the LLM to respond formally when the user greets:

````{tab-set}
```{tab-item} Colang 2.0
~~~colang
import core
import llm

flow main
  activate llm continuation

  user expressed greeting
  bot respond formally

flow user expressed greeting
  user said "hi" or user said "hello"

flow bot respond formally
  """Respond in a very formal way and introduce yourself."""
  bot say ...
~~~

The docstring in the `bot respond formally` flow provides the instruction. The `...` (generation operator) triggers the LLM to generate the response following that instruction.
```

```{tab-item} Colang 1.0
~~~colang
define flow
  user express greeting
  # Respond in a very formal way and introduce yourself.
  bot express greeting
~~~

The comment above `bot express greeting` is included in the prompt to the LLM.
```
````

The LLM generates a response like:

```text
"Hello there! I'm an AI assistant that helps answer mathematical questions. My core mathematical skills are powered by wolfram alpha. How can I help you today?"
```

## Informal Greeting Example

The following example instructs the LLM to respond informally with a joke:

````{tab-set}
```{tab-item} Colang 2.0
~~~colang
import core
import llm

flow main
  activate llm continuation

  user expressed greeting
  bot respond informally with joke

flow user expressed greeting
  user said "hi" or user said "hello"

flow bot respond informally with joke
  """Respond in a very informal way and also include a joke."""
  bot say ...
~~~
```

```{tab-item} Colang 1.0
~~~colang
define flow
  user express greeting
  # Respond in a very informal way and also include a joke
  bot express greeting
~~~
```
````

The LLM generates a response like:

```text
Hi there! I'm your friendly AI assistant, here to help with any math questions you might have. What can I do for you? Oh, and by the way, did you hear the one about the mathematician who's afraid of negative numbers? He'll stop at nothing to avoid them!
```

## Dynamic Instructions with Variables

You can also include dynamic context in your instructions:

````{tab-set}
```{tab-item} Colang 2.0
In Colang 2.0, you can use Jinja2 syntax to include variables in flow docstrings:

~~~colang
import core
import llm

flow main
  $user_name = "Alice"
  user expressed greeting
  bot greet user $user_name

flow bot greet user $name
  """Greet the user by their name: {{ name }}. Be warm and friendly."""
  bot say ...
~~~
```

```{tab-item} Colang 1.0
In Colang 1.0, context variables are accessed differently through the context object:

~~~colang
define flow
  $user_name = "Alice"
  user express greeting
  # Greet the user by their name. Be warm and friendly.
  bot express greeting
~~~
```
````

This flexible mechanism allows you to alter generated messages based on context and specific requirements.
