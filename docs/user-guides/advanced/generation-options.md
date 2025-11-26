# Generation Options

NeMo Guardrails exposes a set of **generation options** that give you fine-grained control over how the LLM generation is performed (e.g., what rails are enabled, additional parameters that should be passed to the LLM, what context data should be returned, what logging information should be returned).

The **generation options** can be used both in the Python API and through the server API.

To use the generation options through the Python API, you must provide the `options` keyword argument:

```python
messages = [{
    "role": "user",
    "content": "..."
}]
rails.generate(messages=messages, options={...})
```

To use the generation options through the server API, you must provide the `options` as part of the request body:

```
POST /v1/chat/completions
```

```json
{
    "config_id": "...",
    "messages": [{
      "role":"user",
      "content":"..."
    }],
    "options": {
      ...
    }
}
```

## Output Variables

Some rails can store additional information in [context variables](../colang-language-syntax-guide.md#variables). You can return the content of these variables by setting the `output_vars` generation option to the list of names for all the variables that you are interested in. If you want to return the complete context (this will also include some predefined variables), you can set `output_vars` to `True`.

```python
rails.generate(messages=messages, options={
    "output_vars": ["some_input_rail_score", "some_output_rail_score"]
})
```

The returned data will be included in the `output_data` key of the response:

```json
{
  "response": [...],
  "output_data": {
    "some_input_rail_score": 0.7,
    "some_output_rail_score": 0.8
  }
}
```

## Additional LLM Parameters

You can pass additional parameters to the LLM call that is used to generate the final message by using the `llm_params` generation option. For example, to use a lower temperature than the default one:

```python
rails.generate(messages=messages, options={
    "llm_params": {
        "temperature": 0.2
    }
})
```

The supported parameters depend on the underlying LLM engine. NeMo Guardrails passes them "as is".

## Additional LLM Output

You can receive the additional output from the LLM generation by using the `llm_output` generation options.

```python
rails.generate(messages=messages, options={
    "llm_output": True
})
```

**NOTE**: The data that is returned is highly dependent on the underlying implementation of the LangChain connector for the LLM provider. For example, for OpenAI, it only returns `token_usage` and `model_name`.

## Detailed Logging Information

You can obtain detailed information about what happened under the hood during the generation process by setting the `log` generation option. This option has four different inner-options:

- `activated_rails`: Include detailed information about the rails that were activated during generation.
- `llm_calls`: Include information about all the LLM calls that were made. This includes: prompt, completion, token usage, raw response, etc.
- `internal_events`: Include the array of internal generated events.
- `colang_history`: Include the history of the conversation in Colang format.

```python
res = rails.generate(messages=messages, options={
    "log": {
        "activated_rails": True,
        "llm_calls": True,
        "internal_events": True,
        "colang_history": True
    }
})
```

```json
{
  "response": [...],
  "log": {
    "activated_rails": {
      ...
    },
    "stats": {...},
    "llm_calls": [...],
    "internal_events": [...],
    "colang_history": "..."
  }
}
```

When using the Python API, the `log` is an object that also has a `print_summary` method. When called, it will print a simplified version of the log information. Below is a sample output.

```python
res.log.print_summary()
```

```markdown
# General stats

- Total time: 2.85s
  - [0.56s][19.64%]: INPUT Rails
  - [1.40s][49.02%]: DIALOG Rails
  - [0.58s][20.22%]: GENERATION Rails
  - [0.31s][10.98%]: OUTPUT Rails
- 5 LLM calls, 2.74s total duration, 1641 total prompt tokens, 103 total completion tokens, 1744 total tokens.

# Detailed stats

- [0.56s] INPUT (self check input): 1 actions (self_check_input), 1 llm calls [0.56s]
- [0.43s] DIALOG (generate user intent): 1 actions (generate_user_intent), 1 llm calls [0.43s]
- [0.96s] DIALOG (generate next step): 1 actions (generate_next_step), 1 llm calls [0.95s]
- [0.58s] GENERATION (generate bot message): 2 actions (retrieve_relevant_chunks, generate_bot_message), 1 llm calls [0.49s]
- [0.31s] OUTPUT (self check output): 1 actions (self_check_output), 1 llm calls [0.31s]
```

## Disabling Rails

You can choose which categories of rails you want to apply by using the `rails` generation option. The four supported categories are: `input`, `dialog`, `retrieval` and `output`. By default, all are enabled.

```python
res = rails.generate(messages=messages)
```

is equivalent to:

```python
res = rails.generate(messages=messages, options={
    "rails": ["input", "dialog", "retrieval", "output"]
})
```

### Input Rails Only

If you only want to check a user's input by running the input rails from a guardrails configuration, you must disable all the others:

```python
res = rails.generate(messages=[{
    "role": "user",
    "content": "Some user input."
}], options={
    "rails": ["input"]
})
```

The response will be the same string if the input was allowed "as is":

```json
{
  "role": "assistant",
  "content": "Some user input."
}
```

If some of the rails alter the input, e.g., to mask sensitive information, then the returned value is the altered input.

```json
{
  "role": "assistant",
  "content": "Some altered user input."
}
```

If the input was blocked, you will get the predefined response `bot refuse to respond` (by default "I'm sorry, I can't respond to that").

```json
{
  "role": "assistant",
  "content": "I'm sorry, I can't respond to that."
}
```

For more details on what rails was triggered, use the `log.activated_rails` generation option.

### Input and Output Rails Only

If you want to check both the user input and an output that was generated outside of the guardrails configuration, you must disable the dialog rails and the retrieval rails, and provide a bot message as well when making the call:

```python
res = rails.generate(messages=[{
    "role": "user",
    "content": "Some user input."
}, {
    "role": "assistant",
    "content": "Some bot output."
}], options={
    "rails": ["input", "output"]
})
```

The response will be the exact bot message provided, if allowed, an altered version if an output rail decides to change it, e.g., to remove sensitive information, or the predefined message for `bot refuse to respond`, if the message was blocked.

For more details on what rails was triggered, use the `log.activated_rails` generation option.

### Output Rails Only

If you want to apply only the output rails to an LLM output, you must disable the input rails as well and provide an empty input.

```python
res = rails.generate(messages=[{
    "role": "user",
    "content": ""
}, {
    "role": "assistant",
    "content": "Some bot output."
}], options={
    "rails": ["output"]
})
```

## Limitations

- Only supported for the `generate`/`generate_async` methods (not for `generate_events`/`generate_events_async`).
- Specifying which individual rails of a particular type to activate is not yet supported.
