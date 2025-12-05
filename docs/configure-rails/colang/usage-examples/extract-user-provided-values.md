---
title: Extract User-provided Values
description: Extract and store user-provided values like names, dates, and queries in context variables.
---

# Extract User-provided Values

This guide teaches you how to extract user-provided values (for example, a name, a date, a query) from a user utterance and store them in context variables. You can then use these values in bot responses or follow-up logic.

## Overview

````{tab-set}
```{tab-item} Colang 2.0
In Colang 2.0, you use **Natural Language Descriptions (NLD)** with the generation operator (`...`) to extract values. The NLD is placed inline after the `...` operator:

~~~colang
$variable_name = ..."Instructions on how to extract the value."
~~~

The NLD together with the variable name is interpreted by the LLM directly. Be specific about the format and type you expect.
```

```{tab-item} Colang 1.0
In Colang 1.0, you place a **comment** above the variable assignment with the `...` operator:

~~~colang
# Comment with instructions on how to extract the value.
# Can span multiple lines.
$variable_name = ...
~~~

The comment is included in the prompt, instructing the LLM on how to compute the variable's value.
```
````

```{note}
`...` is not a placeholder; it is the actual syntax (the generation operator).
```

## Single Values

You can extract single values from user input:

````{tab-set}
```{tab-item} Colang 2.0
~~~colang
import core
import llm

flow main
  activate llm continuation

  user provided name
  $name = ..."Extract the name of the user. Return the name as a single string."
  bot say "Hello, {$name}!"

flow user provided name
  user said "my name is" or user said "I am" or user said "call me"
~~~
```

```{tab-item} Colang 1.0
~~~colang
define user provide name
  "My name is John"
  "I am Alice"
  "Call me Bob"

define flow
  user provide name
  # Extract the name of the user.
  $name = ...
  bot express greeting
~~~
```
````

## Lists of Values

You can instruct the LLM to extract a list of values:

````{tab-set}
```{tab-item} Colang 2.0
~~~colang
import core
import llm

flow main
  activate llm continuation

  user requested add items to cart
  $item_list = ..."Generate a list of the menu items that the user requested to be added to the cart, e.g. ['french fries', 'double protein burger', 'lemonade']. If user specifies no menu items, return an empty list []."

  # Process the items
  bot say "Adding {$item_list} to your cart."

flow user requested add items to cart
  user said "add to cart"
    or user said "I want to order"
    or user said "can I get"
~~~
```

```{tab-item} Colang 1.0
~~~colang
define flow add to cart
  user request add items to cart

  # Generate a list of the menu items that the user requested to be added to the cart
  # e.g. ["french fries", "double protein burger", "lemonade"].
  # If user specifies no menu items, just leave this empty, i.e. [].

  $item_list = ...
~~~
```
````

## Multiple Values

You can extract values for multiple variables from the same user input:

````{tab-set}
```{tab-item} Colang 2.0
~~~colang
import core
import llm

flow main
  activate llm continuation

  user requested book flight
  $origin_city = ..."Extract the origin city from the user's request. If not specified, return 'unknown'."
  $destination_city = ..."Extract the destination city from the user's request. If not specified, return 'unknown'."

  bot say "Booking flight from {$origin_city} to {$destination_city}."

flow user requested book flight
  user said "I want to book a flight"
    or user said "I want to fly"
    or user said "I need a flight"
~~~
```

```{tab-item} Colang 1.0
~~~colang
define user request book flight
  "I want to book a flight."
  "I want to fly from Bucharest to San Francisco."
  "I want a flight to Paris."

define flow
  user request book flight

  # Extract the origin from the user's request. If not specified, say "unknown".
  $origin_city = ...

  # Extract the destination city from the user's request. If not specified, say "unknown".
  $destination_city = ...
~~~
```
````

## Contextual Queries

This mechanism can enable contextual queries. For example, to answer math questions using Wolfram Alpha with follow-up context:

**Example conversation:**

```text
user: "What is the largest prime factor for 1024?"
bot: "The largest prime factor is 2."
user: "And its square root?"
bot: "The square root for 1024 is 32"
```

````{tab-set}
```{tab-item} Colang 2.0
~~~colang
import core
import llm

flow main
  activate llm continuation

  user asked math question
  $math_query = ..."Extract the math question from the user's input. Include any contextual references from the conversation."
  $result = await WolframAlphaAction(query=$math_query)
  bot say $result

flow user asked math question
  user said "what is"
    or user said "calculate"
    or user said "and its"
~~~
```

```{tab-item} Colang 1.0
~~~colang
define flow
  user ask math question

  # Extract the math question from the user's input.
  $math_query = ...

  execute wolfram alpha request(query=$math_query)
  bot respond to math question
~~~
```
````

## Best Practices

````{tab-set}
```{tab-item} Colang 2.0
**Be specific in your NLDs:**

~~~colang
# Good - specific format and fallback
$user_name = ..."Return the user name as a single string between quotes. If no user name is available, return 'friend'."

# Good - specific list format
$items = ..."Return the items as a Python list, e.g. ['item1', 'item2']. Return [] if no items found."

# Avoid - too vague
$value = ..."Get the value."
~~~

**Use variables in NLDs for context:**

~~~colang
$order_info = ..."Extract the order details."
$summary = ..."Provide a brief summary of the current order. Order Information: '{$order_info}'"
~~~
```

```{tab-item} Colang 1.0
**Be specific in your comments:**

~~~colang
# Good - specific format and fallback
# Extract the user's name. If not specified, return "friend".
$name = ...

# Good - specific list format
# Generate a list of items, e.g. ["item1", "item2"]. Return [] if empty.
$items = ...

# Avoid - too vague
# Get the value.
$value = ...
~~~
```
````

## Key Differences

| Feature | Colang 2.0 | Colang 1.0 |
|---------|------------|------------|
| Instruction placement | Inline after `...` | Comment above assignment |
| Syntax | `$var = ..."instruction"` | `# instruction`<br>`$var = ...` |
| String interpolation | `{$var}` in strings | Context variable access |
| Flow definition | `flow name` | `define flow` |
| Action execution | `await ActionName()` | `execute action_name()` |
