# Bot Message Instructions

If you place a comment above a `bot somethig` statement, the comment will be included in the prompt, instructing the LLM further on how to generate the message.

For example:

```colang
define flow
  user express greeting
  # Respond in a very formal way and introduce yourself.
  bot express greeting
```

The above flow would generate a prompt (using the default prompt templates) that looks like this:

```
... (content removed for readability) ...
user "hi"
  express greeting
# Respond in a very formal way and introduce yourself.
bot express greeting
```

And in this case, the completion from the LLM will be:
```
 "Hello there! I'm an AI assistant that helps answer mathematical questions. My core mathematical skills are powered by wolfram alpha. How can I help you today?"
```

Whereas if we change the flow to:

```colang
define flow
  user express greeting
  # Respond in a very informal way and also include a joke
  bot express greeting
```

Then the completion will be something like:

```
Hi there! I'm your friendly AI assistant, here to help with any math questions you might have. What can I do for you? Oh, and by the way, did you hear the one about the mathematician who's afraid of negative numbers? He'll stop at nothing to avoid them!
```

This is a very flexible mechanism for altering the generated messages.
