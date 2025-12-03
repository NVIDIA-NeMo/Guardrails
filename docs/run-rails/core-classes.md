# Core Classes

This guide covers the two fundamental classes in the NeMo Guardrails toolkit: `RailsConfig` for loading configurations and `LLMRails` for generating responses with guardrails.

## RailsConfig

The `RailsConfig` class represents a complete guardrails configuration, including models, rails, flows, prompts, and other settings.

### Loading from a Directory

The most common way to load a configuration is from a directory containing `config.yml` and Colang files:

```python
from nemoguardrails import RailsConfig

config = RailsConfig.from_path("path/to/config")
```

**Expected directory structure:**

```text
config/
├── config.yml          # Main configuration file
├── rails/              # Colang flow files
│   ├── input.co
│   ├── output.co
│   └── ...
├── kb/                 # Knowledge base documents (optional)
│   └── docs.md
├── actions.py          # Custom actions (optional)
└── config.py           # Custom initialization (optional)
```

### Loading from a Single File

You can also load from a single YAML file:

```python
config = RailsConfig.from_path("path/to/config.yml")
```

### Loading from Content

For dynamic configurations or testing, load directly from strings:

```python
from nemoguardrails import RailsConfig

yaml_content = """
models:
  - type: main
    engine: openai
    model: gpt-4

instructions:
  - type: general
    content: |
      You are a helpful assistant.
"""

colang_content = """
define user express greeting
  "hello"
  "hi"

define flow
  user express greeting
  bot express greeting
"""

config = RailsConfig.from_content(
    yaml_content=yaml_content,
    colang_content=colang_content
)
```

### Loading from a Dictionary

You can also provide configuration as a Python dictionary:

```python
config = RailsConfig.from_content(
    config={
        "models": [
            {"type": "main", "engine": "openai", "model": "gpt-4"}
        ],
        "instructions": [
            {"type": "general", "content": "You are a helpful assistant."}
        ]
    }
)
```

### Combining Configurations

Configurations can be combined using the `+` operator:

```python
base_config = RailsConfig.from_path("path/to/base")
additional_config = RailsConfig.from_path("path/to/additional")

combined_config = base_config + additional_config
```

This is useful for:

- Adding rails to a base configuration
- Layering environment-specific settings
- Combining shared and application-specific configurations

### Key Configuration Properties

| Property | Type | Description |
|----------|------|-------------|
| `models` | `List[Model]` | LLM models configuration |
| `instructions` | `List[Instruction]` | System instructions for the LLM |
| `sample_conversation` | `str` | Example conversation for prompts |
| `rails` | `Rails` | Rails configuration (input, output, dialog, etc.) |
| `flows` | `List[Dict]` | Colang flow definitions |
| `prompts` | `List[TaskPrompt]` | Custom prompts for various tasks |
| `streaming` | `bool` | Enable streaming responses |
| `colang_version` | `str` | Colang version ("1.0" or "2.x") |

---

## LLMRails

The `LLMRails` class is the main interface for generating responses with guardrails applied.

### Initialization

```python
from nemoguardrails import LLMRails, RailsConfig

config = RailsConfig.from_path("path/to/config")
rails = LLMRails(config)
```

**Constructor parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `config` | `RailsConfig` | The rails configuration |
| `llm` | `BaseLLM \| BaseChatModel` | Optional pre-configured LLM (overrides config) |
| `verbose` | `bool` | Enable verbose logging |

### Using a Custom LLM

You can provide your own LLM instance:

```python
from langchain_openai import ChatOpenAI
from nemoguardrails import LLMRails, RailsConfig

config = RailsConfig.from_path("path/to/config")
llm = ChatOpenAI(model="gpt-4", temperature=0.7)

rails = LLMRails(config, llm=llm)
```

```{note}
When providing an LLM via the constructor, it takes precedence over any main LLM specified in the configuration.
```

### Generating Responses

#### Using Messages (Chat Format)

```python
response = rails.generate(messages=[
    {"role": "user", "content": "Hello! How are you?"}
])
print(response["content"])
```

#### Using a Prompt (Completion Format)

```python
response = rails.generate(prompt="Complete this sentence: The sky is")
print(response)
```

#### With Conversation History

```python
messages = [
    {"role": "user", "content": "My name is John."},
    {"role": "assistant", "content": "Hello John! How can I help you?"},
    {"role": "user", "content": "What's my name?"}
]

response = rails.generate(messages=messages)
print(response["content"])  # Should remember the name
```

#### Passing Context

You can pass additional context using the `context` role:

```python
response = rails.generate(messages=[
    {
        "role": "context",
        "content": {
            "user_name": "Alice",
            "user_role": "admin"
        }
    },
    {"role": "user", "content": "What permissions do I have?"}
])
```

### Asynchronous Generation

For async contexts, use `generate_async`:

```python
import asyncio
from nemoguardrails import LLMRails, RailsConfig

async def main():
    config = RailsConfig.from_path("path/to/config")
    rails = LLMRails(config)

    response = await rails.generate_async(messages=[
        {"role": "user", "content": "Hello!"}
    ])
    print(response["content"])

asyncio.run(main())
```

### Streaming Responses

For real-time token streaming:

```python
async def stream_response():
    config = RailsConfig.from_path("path/to/config")
    rails = LLMRails(config)

    async for chunk in rails.stream_async(messages=[
        {"role": "user", "content": "Tell me a story."}
    ]):
        print(chunk, end="", flush=True)
```

For detailed streaming configuration, refer to [Streaming](streaming.md).

### Event-based Generation

For low-level control using events:

```python
events = rails.generate_events(events=[
    {
        "type": "UtteranceUserActionFinished",
        "final_transcript": "Hello!"
    }
])

for event in events:
    if event["type"] == "StartUtteranceBotAction":
        print(f"Bot says: {event['script']}")
```

For detailed event-based API usage, refer to [Event-based API](event-based-api.md).

### Generation Options

Fine-tune generation behavior using the `options` parameter:

```python
response = rails.generate(
    messages=[{"role": "user", "content": "Hello!"}],
    options={
        "rails": ["input", "output"],  # Only apply these rails
        "output_vars": ["score"],       # Return context variables
        "log": {
            "activated_rails": True,
            "llm_calls": True
        }
    }
)
```

For detailed options, refer to [Generation Options](generation-options.md).

---

## Registering Custom Actions

You can register custom Python functions as actions:

```python
from nemoguardrails import LLMRails, RailsConfig

async def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"Weather in {city}: Sunny, 22°C"

config = RailsConfig.from_path("path/to/config")
rails = LLMRails(config)

# Register the action
rails.register_action(get_weather, name="get_weather")
```

For detailed action registration, refer to [Actions Guide](../configuration-guide/actions/index.md).

---

## Registering Embedding Search Providers

For custom knowledge base search:

```python
from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.embeddings.index import EmbeddingsIndex

class CustomSearchProvider(EmbeddingsIndex):
    async def search(self, text: str, max_results: int):
        # Custom search logic
        pass

config = RailsConfig.from_path("path/to/config")
rails = LLMRails(config)

# Register the provider
rails.register_embedding_search_provider("custom", CustomSearchProvider)
```

---

## Complete Example

```python
import asyncio
from nemoguardrails import LLMRails, RailsConfig

async def main():
    # Load configuration
    config = RailsConfig.from_content(
        yaml_content="""
models:
  - type: main
    engine: openai
    model: gpt-4

rails:
  input:
    flows:
      - self check input
  output:
    flows:
      - self check output

prompts:
  - task: self_check_input
    content: |
      Check if the following is safe: {{ user_input }}
      Answer (Yes/No):
  - task: self_check_output
    content: |
      Check if the following is safe: {{ bot_response }}
      Answer (Yes/No):
""",
        colang_content="""
define user express greeting
  "hello"
  "hi"

define bot express greeting
  "Hello! How can I help you today?"

define flow
  user express greeting
  bot express greeting
"""
    )

    # Create rails instance
    rails = LLMRails(config, verbose=True)

    # Generate response
    response = await rails.generate_async(
        messages=[{"role": "user", "content": "Hello!"}],
        options={"log": {"activated_rails": True}}
    )

    print(f"Response: {response['content']}")

    # Print what happened
    if hasattr(response, 'log'):
        response.log.print_summary()

asyncio.run(main())
```

---

## Related Resources

- [Generation Options](generation-options.md) - Fine-grained control over generation
- [Streaming](streaming.md) - Real-time token streaming
- [Event-based API](event-based-api.md) - Low-level event control
- [Tools Integration](tools-integration.md) - Integrating LangChain tools
- [Configuration Guide](../configuration-guide/index.md) - Complete configuration reference
