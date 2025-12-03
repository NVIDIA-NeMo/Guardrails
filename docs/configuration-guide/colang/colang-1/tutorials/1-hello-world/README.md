# Hello World

This guide shows you how to create a "Hello World" guardrails configuration that controls the greeting behavior. Before you begin, make sure you have [installed NeMo Guardrails](../../getting-started/installation-guide.md).

## Prerequisites

This "Hello World" guardrails configuration uses the OpenAI `gpt-3.5-turbo-instruct` model.

1. Install the `openai` package:

```bash
pip install openai
```

2. Set the `OPENAI_API_KEY` environment variable:

```bash
export OPENAI_API_KEY=$OPENAI_API_KEY    # Replace with your own key
```

3. If you're running this inside a notebook, patch the AsyncIO loop.

```python
import nest_asyncio

nest_asyncio.apply()
```

## Step 1: create a new guardrails configuration

Every guardrails configuration must be stored in a folder. The standard folder structure is as follows:

```
.
├── config
│   ├── actions.py
│   ├── config.py
│   ├── config.yml
│   ├── rails.co
│   ├── ...
```

See the [Configuration Guide](../../user-guides/configuration-guide.md) for information about the contents of these files.

1. Create a folder, such as *config*, for your configuration:

```bash
mkdir config
```

2. Create a *config.yml* file with the following content:

```yaml
models:
 - type: main
   engine: openai
   model: gpt-3.5-turbo-instruct
```

The `models` key in the *config.yml* file configures the LLM model. For a complete list of supported LLM models, see [Supported LLM Models](../../user-guides/configuration-guide.md#supported-llm-models).

## Step 2: load the guardrails configuration

To load a guardrails configuration from a path, you must create a `RailsConfig` instance using the `from_path` method in your Python code:

```python
from nemoguardrails import RailsConfig

config = RailsConfig.from_path("./config")
```

## Step 3: use the guardrails configuration

Use this empty configuration by creating an `LLMRails` instance and using the `generate_async` method in your Python code:

```python
from nemoguardrails import LLMRails

rails = LLMRails(config)

response = rails.generate(messages=[{
    "role": "user",
    "content": "Hello!"
}])
print(response)
```

```yaml
{'role': 'assistant', 'content': "Hello! It's nice to meet you. My name is Assistant. How can I help you today?"}
```

The format for the input `messages` array as well as the response follow the [OpenAI API](https://platform.openai.com/docs/guides/text-generation/chat-completions-api) format.

## Step 4: add your first guardrail

To control the greeting response, define the user and bot messages, and the flow that connects the two together. See [Core Colang Concepts](../2-core-colang-concepts/README.md) for definitions of *messages* and *flows*.

1. Define the `greeting` user message by creating a *config/rails.co* file with the following content:

```colang
define user express greeting
  "Hello"
  "Hi"
  "Wassup?"
```

2. Add a greeting flow that instructs the bot to respond back with "Hello World!" and ask how they are doing by adding the following content to the *rails.co* file:

```python
define flow greeting
  user express greeting
  bot express greeting
  bot ask how are you
```

3. Define the messages for the response by adding the following content to the *rails.co* file:

```python
define bot express greeting
  "Hello World!"

define bot ask how are you
  "How are you doing?"
```

4. Reload the config and test it:

```python
config = RailsConfig.from_path("./config")
rails = LLMRails(config)

response = rails.generate(messages=[{
    "role": "user",
    "content": "Hello!"
}])
print(response["content"])
```

```
Hello World!
How are you doing?
```

**Congratulations!** You've just created you first guardrails configuration!

### Other queries

What happens if you ask another question, such as "What is the capital of France?":

```python
response = rails.generate(messages=[{
    "role": "user",
    "content": "What is the capital of France?"
}])
print(response["content"])
```

```
The capital of France is Paris.
```

For any other input that is not a greeting, the LLM generates the response as usual. This is because the rail that we have defined is only concerned with how to respond to a greeting.

## CLI Chat

You can also test this configuration in interactive mode using the NeMo Guardrails CLI Chat command:

```bash
$ nemoguardrails chat
```

Without any additional parameters, the CLI chat loads the configuration from the *config.yml* file in the *config* folder in the current directory.

### Sample session
```
$ nemoguardrails chat
Starting the chat (Press Ctrl+C to quit) ...

> Hello there!
Hello World!
How are you doing?

> What is the capital of France?
The capital of france is Paris.

> And how many people live there?
According to the latest estimates, the population of Paris is around 2.2 million people.
```

## Server and Chat UI

You can also test a guardrails configuration using the NeMo Guardrails server and the Chat UI.

To start the server:

```bash
$ nemoguardrails server --config=.

INFO:     Started server process [27509]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

The Chat UI interface is now available at `http://localhost:8000`:

![hello-world-server-ui.png](../../_static/images/hello-world-server-ui.png)

## Next

The next guide, [Core Colang Concepts](../2-core-colang-concepts/README.md), explains the Colang concepts *messages* and *flows*.
