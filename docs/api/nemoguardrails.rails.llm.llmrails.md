<!-- markdownlint-disable -->

<a href="../../nemoguardrails/rails/llm/llmrails.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

# <kbd>module</kbd> `nemoguardrails.rails.llm.llmrails`
LLM Rails entry point.

**Global Variables**
---------------
- **explain_info_var**
- **streaming_handler_var**


---

<a href="../../nemoguardrails/rails/llm/llmrails.py#L45"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `LLMRails`
Rails based on a given configuration.

<a href="../../nemoguardrails/rails/llm/llmrails.py#L48"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

### <kbd>method</kbd> `LLMRails.__init__`

```python
__init__(
    config: nemoguardrails.rails.llm.config.RailsConfig,
    llm: Optional[langchain.llms.base.BaseLLM] = None,
    verbose: bool = False
)
```

Initializes the LLMRails instance.



**Args:**

 - <b>`config`</b>:  A rails configuration.
 - <b>`llm`</b>:  An optional LLM engine to use.
 - <b>`verbose`</b>:  Whether the logging should be verbose or not.




---

<a href="../../nemoguardrails/rails/llm/llmrails.py#L560"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

### <kbd>method</kbd> `LLMRails.explain`

```python
explain() → ExplainInfo
```

Helper function to return the latest ExplainInfo object.

---

<a href="../../nemoguardrails/rails/llm/llmrails.py#L464"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

### <kbd>method</kbd> `LLMRails.generate`

```python
generate(prompt: Optional[str] = None, messages: Optional[List[dict]] = None)
```

Synchronous version of generate_async.

---

<a href="../../nemoguardrails/rails/llm/llmrails.py#L347"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

### <kbd>method</kbd> `LLMRails.generate_async`

```python
generate_async(
    prompt: Optional[str] = None,
    messages: Optional[List[dict]] = None,
    streaming_handler: Optional[nemoguardrails.streaming.StreamingHandler] = None
) → Union[str, dict]
```

Generate a completion or a next message.

The format for messages is the following:

```python
     [
         {"role": "context", "content": {"user_name": "John"}},
         {"role": "user", "content": "Hello! How are you?"},
         {"role": "assistant", "content": "I am fine, thank you!"},
         {"role": "event", "event": {"type": "UserSilent"}},
         ...
     ]
```



**Args:**

 - <b>`prompt`</b>:  The prompt to be used for completion.
 - <b>`messages`</b>:  The history of messages to be used to generate the next message.
 - <b>`streaming_handler`</b>:  If specified, and the config supports streaming, the  provided handler will be used for streaming.



**Returns:**
 The completion (when a prompt is provided) or the next message.

---

<a href="../../nemoguardrails/rails/llm/llmrails.py#L513"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

### <kbd>method</kbd> `LLMRails.generate_events`

```python
generate_events(events: List[dict]) → List[dict]
```

Synchronous version of `LLMRails.generate_events_async`.

---

<a href="../../nemoguardrails/rails/llm/llmrails.py#L477"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

### <kbd>method</kbd> `LLMRails.generate_events_async`

```python
generate_events_async(events: List[dict]) → List[dict]
```

Generate the next events based on the provided history.

The format for events is the following:

```python
     [
         {"type": "...", ...},
         ...
     ]
```



**Args:**

 - <b>`events`</b>:  The history of events to be used to generate the next events.



**Returns:**
 The newly generate event(s).

---

<a href="../../nemoguardrails/rails/llm/llmrails.py#L524"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

### <kbd>method</kbd> `LLMRails.register_action`

```python
register_action(
    action: <built-in function callable>,
    name: Optional[str] = None
)
```

Register a custom action for the rails configuration.

---

<a href="../../nemoguardrails/rails/llm/llmrails.py#L528"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

### <kbd>method</kbd> `LLMRails.register_action_param`

```python
register_action_param(name: str, value: Any)
```

Registers a custom action parameter.

---

<a href="../../nemoguardrails/rails/llm/llmrails.py#L548"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

### <kbd>method</kbd> `LLMRails.register_embedding_search_provider`

```python
register_embedding_search_provider(
    name: str,
    cls: Type[nemoguardrails.embeddings.index.EmbeddingsIndex]
) → None
```

Register a new embedding search provider.



**Args:**

 - <b>`name`</b>:  The name of the embedding search provider that will be used.
 - <b>`cls`</b>:  The class that will be used to generate and search embedding

---

<a href="../../nemoguardrails/rails/llm/llmrails.py#L532"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

### <kbd>method</kbd> `LLMRails.register_filter`

```python
register_filter(
    filter_fn: <built-in function callable>,
    name: Optional[str] = None
)
```

Register a custom filter for the rails configuration.

---

<a href="../../nemoguardrails/rails/llm/llmrails.py#L536"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

### <kbd>method</kbd> `LLMRails.register_output_parser`

```python
register_output_parser(output_parser: <built-in function callable>, name: str)
```

Register a custom output parser for the rails configuration.

---

<a href="../../nemoguardrails/rails/llm/llmrails.py#L540"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

### <kbd>method</kbd> `LLMRails.register_prompt_context`

```python
register_prompt_context(name: str, value_or_fn: Any)
```

Register a value to be included in the prompt context.

:name: The name of the variable or function that will be used. :value_or_fn: The value or function that will be used to generate the value.

---

<a href="../../nemoguardrails/rails/llm/llmrails.py#L446"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

### <kbd>method</kbd> `LLMRails.stream_async`

```python
stream_async(
    prompt: Optional[str] = None,
    messages: Optional[List[dict]] = None
) → AsyncIterator[str]
```

Simplified interface for getting directly the streamed tokens from the LLM.
