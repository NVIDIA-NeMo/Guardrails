# Colang Guide

This document is a brief introduction Colang 1.0.

Colang is a modeling language enabling the design of guardrails for conversational systems.

```{warning}
Colang can be used to perform complex activities, such as calling python scripts and performing multiple calls to the underlying language model. You should avoid loading Colang files from untrusted sources without careful inspection.
```

## Why a New Language

Creating guardrails for conversational systems requires some form of understanding of how the dialogue between the user and the bot unfolds. Existing dialog management techniques such us flow charts, state machines, frame-based systems, etc. are not well suited for modeling highly flexible conversational flows like the ones we expect when interacting with an LLM-based system like ChatGPT.

However, since learning a new language is not an easy task, Colang was designed as a mix of natural language and python. If you are familiar with python, you should feel confident using Colang after seeing a few examples, even without any explanation.

## Concepts

Below are the main concepts behind the language:

- **LLM-based Application**: a software application that uses an LLM to drive
- **Bot**: synonym for *LLM-based application*.
- **Utterance**: the raw text coming from the user or the bot.
- **Intent**: the canonical form (i.e. structured representation) of a user/bot utterance.
- **Event**: something that has happened and is relevant to the conversation e.g. user is silent, user clicked something, user made a gesture, etc.
- **Action**: a custom code that the bot can invoke; usually for connecting to third-party API.
- **Context**: any data relevant to the conversation (i.e. a key-value dictionary).
- **Flow**: a sequence of messages and events, potentially with additional branching logic.
- **Rails**: specific ways of controlling the behavior of a conversational system (a.k.a. bot) e.g. not talk about politics, respond in a specific way to certain user requests, follow a predefined dialog path, use a specific language style, extract data etc.

## Syntax

Colang has a "pythonic" syntax in the sense that most constructs resemble their python equivalent and indentation is used as a syntactic element.

```{note}
Unlike python, the recommended indentation in Colang is **two spaces**, rather than four.
```

### Core Syntax Elements

The core syntax elements are: blocks, statements, expressions, keywords and variables. There are three main types of blocks: *user message blocks* (`define user ...`), *flow blocks* (`define flow ...`) and *bot message blocks* (`define bot ...`).

#### User Messages

User message definition blocks define the canonical form message that should be associated with various user utterances e.g.:

```colang
define user express greeting
  "hello"
  "hi"

define user request help
  "I need help with something."
  "I need your help."
```

#### Bot Messages

Bot message definition blocks define the utterances that should be associated with various bot message canonical forms:

```colang
define bot express greeting
  "Hello there!"
  "Hi!"

define bot ask welfare
  "How are you feeling today?"
```

If more than one utterance is specified per bot message, the meaning is that one of them should be chosen randomly.

##### Bot Messages with Variables

The utterance definition can also include reference to variables (see the [Variables](#variables) section below).

```colang
define bot express greeting
  "Hello there, $name!"
```

Alternatively, you can also use the Jinja syntax:

```colang
define bot express greeting
  "Hello there, {{ name }}!"
```

```{note}
For more advanced use cases you can also use other Jinja features like `{% if ... %} ... {% endif %}`.
```

#### Flows

Flows represent how you want the conversation to unfold. It includes sequences of user messages, bot messages and potentially other events.

```colang
define flow hello
  user express greeting
  bot express greeting
  bot ask welfare
```

Additionally, flows can contain additional logic which can be modeled using `if` and `when`.

For example, to alter the greeting message based on whether the user is talking to the bot for the first time or not, we can do the following (we can model this using `if`):

```colang
define flow hello
  user express greeting
  if $first_time_user
    bot express greeting
    bot ask welfare
  else
    bot expess welcome back
```

The `$first_time_user` context variable would have to be set by the host application.

As another example, after asking the user how they feel (`bot ask welfare`) we can have different paths depending on the user response (we can model this using `when`):

```colang
define flow hello
  user express greeting
  bot express greeting
  bot ask welfare

  when user express happiness
    bot express happiness
  else when user express sadness
    bot express empathy
```

The `if/else` statement can be used to evaluate expressions involving context variables and alter the flow accordingly. The `when/else` statement can be used to branch the flow based on next user message/event.

#### Subflows

Subflows are a particular type of flows. While flows are meant to be applied automatically to the current conversation (when there is a match), subflows are meant to be called explicitly by other flows/subflows. A subflow can be invoked using the `do` keyword and the name of the subflow:

```colang

define subflow check user authentication
  if not $user_auth
    bot inform authentication required
    bot ask name
    ...

define flow greeting
  """We first authenticate the user, before continuing."""
  user express greeting
  do check user authentication
  bot express greeting
```

Subflows should be used for reusable pieces of conversational logic, e.g., authentication, form filling.

#### Variables

References to context variables always start with a `$` sign e.g. `$name`. All variables are global and accessible in all flows.

Each conversation is associated with a global context which contains a set of variables and their respective values (key-value pairs). The value for a context variable can be set either directly, or as the return value from an action execution.

```colang
define flow
  ...
  $name = "John"
  $allowed = execute check_if_allowed
```

Context variables are dynamically typed, and they can be: booleans, integers, floats and strings. Variables can also hold complex types such as lists and dictionaries, but they can't be initialized directly to this type of values i.e. the value would come from the return value of an action.

#### Expressions

Expressions can be used to set values for context variables.

Types of supported expressions:

- arithmetic operations
- array indexing using `[...]`
- `len(...)` for arrays and strings
- property accessor using "." for dict objects

#### Actions

Actions are custom functions available to be invoked from flows. Action execution can be invoked in a flow using the following syntax:

```colang
define flow ...
  ...
  $result = execute some_action(some_param_1=some_value_1, ...)
```

All action parameters must be passed like keyword arguments in python.

Actions **are not defined** in Colang. They are made available to the guardrails configuration at runtime by the host application.

## Conclusion

This was a brief introduction to Colang 1.0. For more details, check out the [Examples folder](https://github.com/NVIDIA/NeMo-Guardrails/tree/develop/examples) document.
