# Prompt Customization

**NOTE**: this documentation is intended for developers that want to extend/improve the support for different LLM engines.

## Task-oriented Prompting

The interaction with the LLM is designed in a task-oriented way, i.e., each time the LLM is called, it must perform a specific task. The most important tasks, which are part of the [guardrails process](../../architecture/README.md#the-guardrails-process), are:

1. `generate_user_intent`: generate the canonical user message from the raw utterance (e.g., "Hello there" -> `express greeting`);
2. `generate_next_steps`: decide what the bot should say or what action should be executed (e.g., `bot express greeting`, `bot respond to question`);
3. `generate_bot_message`: decide the exact bot message that should be returned.
4. `general`: generate the next bot message based on the history of user and bot messages; this task is used when there are no dialog rails defined (i.e., no user message canonical forms).

Check out the [Task type](https://github.com/NVIDIA/NeMo-Guardrails/blob/develop/nemoguardrails/llm/types.py) for the complete list of tasks.

## Prompt Configuration

The toolkit provides predefined prompts for each task and for certain LLM models. They are located in the [nemoguardrails/llm/prompts](https://github.com/NVIDIA/NeMo-Guardrails/tree/develop/nemoguardrails/llm/prompts) folder. You can customize the prompts further by including a `prompts.yml` file in a guardrails configuration (technically, the file name is not essential, and you can also include the `prompts` key in the general `config.yml` file).

Additionally, if the environment variable `PROMPTS_DIR` is set, the toolkit will also load any prompts defined in the specified directory. The loading is performed once, when the python module is loaded. The folder must contain one or more `.yml` files which contain prompt definitions (inside the `prompts` key).

To override the prompt for a specific model, you need to specify the `models` key:

```yaml
prompts:
  - task: general
    models:
      - databricks/dolly-v2-3b
    content: |-
      ...

  - task: generate_user_intent
    models:
      - databricks/dolly-v2-3b
    content: |-
      ...

  - ...
```

You can associate a prompt for a specific task with multiple LLM models:

```yaml
prompts:
  - task: generate_user_intent
    models:
      - openai/gpt-3.5-turbo
      - openai/gpt-4

...
```

To override the prompt for any other custom purpose, you can specify the `mode` key. If the corresponding task configuration is run with the same `prompting_mode`, the custom prompt will be used.

As an example of this, let's consider the case of compacting. Some applications might need concise prompts, for instance to avoid handling long contexts, and lower latency at the risk of slightly degraded performance due to the smaller context. For this, you might want to have multiple versions of a prompt for the same task and same model. This can be achieved as follows:

Task configuration:

```yaml
models:
  - type: main
    engine: openai
    model: gpt-3.5-turbo

prompting_mode: "compact"  # Default value is "standard"
```

Prompts configuration:

```yaml
prompts:
  - task: generate_user_intent
    models:
      - openai/gpt-3.5-turbo
      - openai/gpt-4
    content: |-
      Default prompt tailored for high accuracy with the given models for example by adding the fill {{ history }}

  - task: generate_user_intent
    models:
      - openai/gpt-3.5-turbo
      - openai/gpt-4
    content: |-
      Smaller prompt tailored for high accuracy by reducing number of few shot examples or other means
    mode: compact
...
```

You can have as many different modes as you like for a given task and model, as long as the `mode` key inside the prompt configuration matches the `prompting_mode` key in the top-level task configuration, thus enabling an easy setup for prompt engineering experiments.

Note that if you specify a custom `prompting_mode` but no prompt definition with the same custom `mode` is defined, then, the `standard` prompt template for that task is used.

### Prompt Templates

Depending on the type of LLM, there are two types of templates you can define: **completion** and **chat**. For completion models (e.g., `gpt-3.5-turbo-instruct`), you need to include the `content` key in the configuration of a prompt:

```yaml
prompts:
  - task: generate_user_intent
    models:
      - openai/gpt-3.5-turbo-instruct
    content: |-
      ...
```

For chat models (e.g., `gpt-3.5-turbo`), you need to include the `messages` key in the configuration of a prompt:

```yaml
prompts:
  - task: generate_user_intent
    models:
      - openai/gpt-3.5-turbo
    messages:
      - type: system
        content: ...
      - type: user
        content: ...
      - type: bot
        content: ...
      # ...
```

For each task, you can also specify the maximum length of the prompt to be used for the LLM call in terms of the number of characters. This is useful if you want to limit the number of tokens used by the LLM or when you want to make sure that the prompt length does not exceed the maximum context length. When the maximum length is exceeded, the prompt is truncated by removing older turns from the conversation history until length of the prompt is less than or equal to the maximum length. The default maximum length is 16000 characters.

For example, for the `generate_user_intent` task, you can specify the following:

```yaml
prompts:
  - task: generate_user_intent
    models:
      - openai/gpt-3.5-turbo
    max_length: 3000
```

### Content Template

The content for a completion prompt or the body for a message in a chat prompt is a string that can also include variables and potentially other types of constructs. NeMo Guardrails uses [Jinja2](https://jinja.palletsprojects.com/) as the templating engine. Check out the [Jinja Synopsis](https://jinja.palletsprojects.com/en/3.1.x/templates/#synopsis) for a quick introduction.

As an example, the default template for the `generate_user_intent` task is the following:

```
"""
{{ general_instructions }}
"""

# This is how a conversation between a user and the bot can go:
{{ sample_conversation }}

# This is how the user talks:
{{ examples }}

# This is the current conversation between the user and the bot:
{{ sample_conversation | first_turns(2) }}
{{ history | colang }}
```

#### Variables

There are three types of variables available to be included in the prompt:

1. System variables
2. Prompt variables
3. Context variables

##### System Variables

The following is the list of system variables:

- `general_instructions`: the content corresponds to the [general instructions](../configuration-guide.md#general-instructions) specified in the configuration;
- `sample_conversation`: the content corresponds to the [sample conversation](../configuration-guide.md#sample-conversation) specified in the configuration;
- `examples`: depending on the task, this variable will contain the few-shot examples that the LLM should take into account;
- `history`: contains the history of events (see the [complete example](../../architecture/README.md#complete-example))
- `relevant_chunks`: (only available for the `generate_bot_message` task) if a knowledge base is used, this variable will contain the most relevant chunks of text based on the user query.

##### Prompt Variables

Prompt variables can be registered using the `LLMRails.register_prompt_context(name, value_or_fn)` method. If a function is provided, the value of the variable will be computed for each rendering.

##### Context Variables

Flows included in a guardrails configuration can define (and update) various [context variables](../colang-language-syntax-guide.md#variables). These can also be included in a prompt if needed.

#### Filters

The concept of filters is the same as in Jinja (see [Jinja filters](https://jinja.palletsprojects.com/en/3.1.x/templates/#filters)). Filters can modify the content of a variable, and you can apply multiple filters using the pipe symbol (`|`).

The list of predefined filters is the following:

- `colang`: transforms an array of events into the equivalent colang representation;
- `remove_text_messages`: removes the text messages from a colang history (leaving only the user intents, bot intents and other actions);
- `first_turns(n)`: limits a colang history to the first `n` turns;
- `user_assistant_sequence`: transforms an array of events into a sequence of "User: .../Assistant: ..." sequence;
- `to_messages`: transforms a colang history of into a sequence of user and bot messages (intended for chat models);
- `verbose_v1`: transforms a colang history into a more verbose and explicit form.

#### Output Parsers

Optionally, the output from the LLM can be parsed using an *output parser*. The list of predefined parsers is the following:

- `user_intent`: parse the user intent, i.e., removes the "User intent:" prefix if present;
- `bot_intent`: parse the bot intent, i.e., removes the "Bot intent:" prefix if present;
- `bot_message`: parse the bot message, i.e., removes the "Bot message:" prefix if present;
- `verbose_v1`: parse the output of the `verbose_v1` filter.

## Predefined Prompts

Currently, the NeMo Guardrails toolkit includes prompts for `openai/gpt-3.5-turbo-instruct`, `openai/gpt-3.5-turbo`, `openai/gpt-4`, `databricks/dolly-v2-3b`, `cohere/command`, `cohere/command-light`, `cohere/command-light-nightly`.

**DISCLAIMER**: Evaluating and improving the provided prompts is a work in progress. We do not recommend deploying this alpha version using these prompts in a production setting.

## Custom Tasks and Prompts

In the scenario where you would like to create a custom task beyond those included in
[the default tasks](https://github.com/NVIDIA/NeMo-Guardrails/tree/develop/nemoguardrails/llm/types.py), you can include the task and associated prompt as provided in the example below:

```yaml
prompts:
- task: summarize_text
  content: |-
      Text: {{ user_input }}
      Summarize the above text.
```

Refer to ["Prompt Customization"](#prompt-customization) on where to include this custom task and prompt.

Within an action, this prompt can be rendered via the `LLMTaskManager`:

```python
prompt = llm_task_manager.render_task_prompt(
    task="summarize_text",
    context={
        "user_input": user_input,
    },
)

check = await llm_call(llm, prompt, llm_params={"temperature": 0.0})
...
```

With this approach, you can quickly modify custom tasks' prompts in your configuration files.
