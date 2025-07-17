# General Options

The following subsections describe all the configuration options you can use in the `config.yml` file.

## General Instructions

The general instructions (similar to a system prompt) get appended at the beginning of every prompt, and you can configure them as shown below:

```yaml
instructions:
  - type: general
    content: |
      Below is a conversation between the NeMo Guardrails bot and a user.
      The bot is talkative and provides lots of specific details from its context.
      If the bot does not know the answer to a question, it truthfully says it does not know.
```

In the future, multiple types of instructions will be supported, hence the `type` attribute and the array structure.

## Sample Conversation

The sample conversation sets the tone for how the conversation between the user and the bot should go. It will help the LLM learn better the format, the tone of the conversation, and how verbose responses should be. This section should have a minimum of two turns. Since we append this sample conversation to every prompt, it is recommended to keep it short and relevant.

```yaml
sample_conversation: |
  user "Hello there!"
    express greeting
  bot express greeting
    "Hello! How can I assist you today?"
  user "What can you do for me?"
    ask about capabilities
  bot respond about capabilities
    "As an AI assistant, I can help provide more information on NeMo Guardrails toolkit. This includes question answering on how to set it up, use it, and customize it for your application."
  user "Tell me a bit about the what the toolkit can do?"
    ask general question
  bot response for general question
    "NeMo Guardrails provides a range of options for quickly and easily adding programmable guardrails to LLM-based conversational systems. The toolkit includes examples on how you can create custom guardrails and compose them together."
  user "what kind of rails can I include?"
    request more information
  bot provide more information
    "You can include guardrails for detecting and preventing offensive language, helping the bot stay on topic, do fact checking, perform output moderation. Basically, if you want to control the output of the bot, you can do it with guardrails."
  user "thanks"
    express appreciation
  bot express appreciation and offer additional help
    "You're welcome. If you have any more questions or if there's anything else I can help you with, please don't hesitate to ask."
```

## Actions Server URL

If an actions server is used, the URL must be configured in the `config.yml`:

```yaml
actions_server_url: ACTIONS_SERVER_URL
```

## LLM Prompts

You can customize the prompts that are used for the various LLM tasks (e.g., generate user intent, generate next step, generate bot message) using the `prompts` key. For example, to override the prompt used for the `generate_user_intent` task for the `openai/gpt-3.5-turbo` model:

```yaml
prompts:
  - task: generate_user_intent
    models:
      - openai/gpt-3.5-turbo
    max_length: 3000
    output_parser: user_intent
    content: |-
      <<This is a placeholder for a custom prompt for generating the user intent>>
```

For each task, you can also specify the maximum length of the prompt to be used for the LLM call in terms of the number of characters. This is useful if you want to limit the number of tokens used by the LLM or when you want to make sure that the prompt length does not exceed the maximum context length. When the maximum length is exceeded, the prompt is truncated by removing older turns from the conversation history until the length of the prompt is less than or equal to the maximum length. The default maximum length is 16000 characters.

The full list of tasks used by the NeMo Guardrails toolkit is the following:

- `general`: generate the next bot message, when no canonical forms are used;
- `generate_user_intent`: generate the canonical user message;
- `generate_next_steps`: generate the next thing the bot should do/say;
- `generate_bot_message`: generate the next bot message;
- `generate_value`: generate the value for a context variable (a.k.a. extract user-provided values);
- `self_check_facts`: check the facts from the bot response against the provided evidence;
- `self_check_input`: check if the input from the user should be allowed;
- `self_check_output`: check if bot response should be allowed;
- `self_check_hallucination`: check if the bot response is a hallucination.

You can check the default prompts in the [prompts](https://github.com/NVIDIA/NeMo-Guardrails/tree/develop/nemoguardrails/llm/prompts) folder.

## Multi-step Generation

With a large language model (LLM) that is fine-tuned for instruction following, particularly those exceeding 100 billion parameters, it's possible to enable the generation of complex, multi-step flows.

**EXPERIMENTAL**: this feature is experimental and should only be used for testing and evaluation purposes.

```yaml
enable_multi_step_generation: True
```

## Lowest Temperature

This temperature will be used for the tasks that require deterministic behavior (e.g., `dolly-v2-3b` requires a strictly positive one).

```yaml
lowest_temperature: 0.1
```

## Event Source ID

This ID will be used as the `source_uid` for all events emitted by the Colang runtime. Setting this to something else than the default value (default value is `NeMoGuardrails-Colang-2.x`) is useful if you need to distinguish multiple Colang runtimes in your system (e.g. in a multi-agent scenario).

```yaml
event_source_uid : colang-agent-1
```

## Custom Data

If you need to pass additional configuration data to any custom component for your configuration, you can use the `custom_data` field.

```yaml
custom_data:
  custom_config_field: "some_value"
```

For example, you can access the custom configuration inside the `init` function in your `config.py` (see [Custom Initialization](custom-initialization.md)).

```python
def init(app: LLMRails):
    config = app.config

    # Do something with config.custom_data
```
