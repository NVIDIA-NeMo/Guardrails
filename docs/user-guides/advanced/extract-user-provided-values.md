# Extract User-provided Values

## Overview

This guide will teach you how to extract user-provided values (e.g., a name, a date, a query) from a user utterance and store them in context variables. You can then use these bot responses or follow-up logic.

The general syntax is the following:

```colang
# Comment with instructions on how to extract the value.
# Can span multiple lines.
$variable_name = ...
```

**Note**: `...` is not a placeholder here; it is the actual syntax, i.e., ellipsis.

At any point in a flow, you can include a `$variable_name = ...`, instructing the LLM to compute the variable's value.

## Single Values or Lists

You can extract single values.

```colang
user provide name
# Extract the name of the user.
$name = ...
```

Or, you can also instruct the LLM to extract a list of values.

```colang
define flow add to cart
  user request add items to cart

  # Generate a list of the menu items that the user requested to be added to the cart
  # e.g. ["french fries", "double protein burger", "lemonade"].
  # If user specifies no menu items, just leave this empty, i.e. [].

  $item_list = ...
```

## Multiple Values

If you extract the values for multiple variables from the same user input.

```colang
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
```

## Contextual Queries

This mechanism can be applied to enable contextual queries. For example, let's assume you want to answer math questions using Wolfram Alpha and support a flow like the following:

```colang
user "What is the largest prime factor for 1024?"
bot "The largest prime factor is 2."
user "And its square root?"
bot "The square root for 1024 is 32"
```

To achieve this, you can use the following flow:

```colang
define flow
  user ask math question

  # Extract the math question from the user's input.
  $math_query = ...

  execute wolfram alpha request(query=$math_query)
  bot respond to math question
```
