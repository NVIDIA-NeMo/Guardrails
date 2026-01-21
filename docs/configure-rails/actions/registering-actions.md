---
title: Registering Actions
description: Register custom actions via actions.py, LLMRails.register_action(), or init.py for different use cases.
---

# Registering Actions

This section describes the different ways to register custom actions with the NeMo Guardrails library.

## Registration Methods

| Method | Description | Use Case |
|--------|-------------|----------|
| File-based | Actions in `actions.py` are auto-registered | Standard configurations |
| Programmatic | Register via `LLMRails.register_action()` | Dynamic registration |
| LangChain tools | Register LangChain tools as actions | Tool integration |
| Actions server | Remote action execution | Distributed systems |

## File-Based Registration

Actions defined in `actions.py` or the `actions/` package are automatically registered when the configuration is loaded.

### Single File (`actions.py`)

```text
config/
├── config.yml
├── actions.py        # Actions auto-registered
└── rails/
    └── ...
```

```python
# config/actions.py
from nemoguardrails.actions import action

@action()
async def my_action():
    return "result"

@action(name="custom_name")
async def another_action():
    return "another result"
```

### Package (`actions/`)

For larger projects, organize actions in a package:

```text
config/
├── config.yml
├── actions/
│   ├── __init__.py
│   ├── validation.py
│   ├── external.py
│   └── utils.py
└── rails/
    └── ...
```

```python
# config/actions/__init__.py
from .validation import check_input, check_output
from .external import fetch_data, call_api
```

```python
# config/actions/validation.py
from nemoguardrails.actions import action

@action()
async def check_input(text: str):
    return len(text) > 0

@action()
async def check_output(text: str):
    return "error" not in text.lower()
```

## Programmatic Registration

Register actions dynamically using `LLMRails.register_action()`:

```python
from nemoguardrails import LLMRails, RailsConfig

config = RailsConfig.from_path("config")
rails = LLMRails(config)

# Register a function as an action
async def my_dynamic_action(param: str):
    return f"Processed: {param}"

rails.register_action(my_dynamic_action, name="dynamic_action")
```

### Use Cases for Programmatic Registration

1. **Runtime configuration**:

```python
def setup_rails(environment: str):
    config = RailsConfig.from_path("config")
    rails = LLMRails(config)

    if environment == "production":
        rails.register_action(production_validator, "validate")
    else:
        rails.register_action(dev_validator, "validate")

    return rails
```

2. **Dependency injection**:

```python
class DatabaseService:
    async def query(self, sql: str):
        # Database query logic
        pass

db = DatabaseService()

async def db_query_action(query: str):
    return await db.query(query)

rails.register_action(db_query_action, name="query_database")
```

## LangChain Tool Registration

Register LangChain tools as guardrails actions:

### Basic Tool Registration

```python
from langchain_core.tools import tool
from nemoguardrails import LLMRails, RailsConfig

@tool
def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"Weather in {city}: Sunny, 72°F"

config = RailsConfig.from_path("config")
rails = LLMRails(config)

# Register the tool as an action
rails.register_action(get_weather, name="get_weather")
```

### Using Registered Tools in Colang

```colang
define flow weather_flow
  user ask about weather
  $weather = execute get_weather(city=$city_name)
  bot provide weather info
```

### Multiple Tool Registration

```python
from langchain_core.tools import tool

@tool
def search_web(query: str) -> str:
    """Search the web."""
    return f"Results for: {query}"

@tool
def calculate(expression: str) -> str:
    """Calculate a math expression."""
    return str(eval(expression))

# Register multiple tools
tools = [search_web, calculate]
for t in tools:
    rails.register_action(t, name=t.name)
```

## Runnable Registration

Register LangChain Runnables as actions:

```python
from langchain_core.runnables import RunnableLambda
from nemoguardrails import LLMRails, RailsConfig

# Create a runnable
process_text = RunnableLambda(lambda x: x.upper())

config = RailsConfig.from_path("config")
rails = LLMRails(config)

# Register the runnable
rails.register_action(process_text, name="process_text")
```

## Actions Server

For distributed deployments, use an actions server:

### Configure the Actions Server URL

```yaml
# config.yml
actions_server_url: http://actions-server:8080
```

### Start the Actions Server

```bash
nemoguardrails actions-server --config config/
```

### Actions Server Benefits

- Centralized action management
- Horizontal scaling
- Separation of concerns
- Easier updates without redeploying the main service

## Registration in `config.py`

Use `config.py` for custom initialization including action registration:

```python
# config/config.py
from nemoguardrails import LLMRails

def init(app: LLMRails):
    """Custom initialization function."""

    # Register actions
    async def custom_action(param: str):
        return f"Custom: {param}"

    app.register_action(custom_action, name="custom_action")

    # Register action parameters
    db_connection = create_db_connection()
    app.register_action_param("db", db_connection)
```

### Registering Action Parameters

Provide shared resources to actions:

```python
# config/config.py
def init(app: LLMRails):
    # Create shared resources
    http_client = aiohttp.ClientSession()
    cache = RedisCache()

    # Register as action parameters
    app.register_action_param("http_client", http_client)
    app.register_action_param("cache", cache)
```

```python
# config/actions.py
from nemoguardrails.actions import action

@action()
async def fetch_with_cache(
    url: str,
    http_client=None,  # Injected automatically
    cache=None         # Injected automatically
):
    # Check cache first
    cached = await cache.get(url)
    if cached:
        return cached

    # Fetch and cache
    response = await http_client.get(url)
    data = await response.json()
    await cache.set(url, data)

    return data
```

## Best Practices

### 1. Use Descriptive Names

```python
# Good
@action(name="validate_user_age")
async def validate_age(age: int):
    pass

# Avoid
@action(name="v_a")
async def validate_age(age: int):
    pass
```

### 2. Group Related Actions

```text
actions/
├── __init__.py
├── validation/
│   ├── __init__.py
│   ├── input.py
│   └── output.py
├── external/
│   ├── __init__.py
│   ├── weather.py
│   └── search.py
└── utils.py
```

### 3. Document Your Actions

```python
@action()
async def search_knowledge_base(
    query: str,
    top_k: int = 5
) -> list:
    """
    Search the knowledge base for relevant documents.

    Args:
        query: The search query string
        top_k: Maximum number of results to return

    Returns:
        List of relevant document snippets
    """
    pass
```

## Related Topics

- [Creating Custom Actions](creating-actions) - Create your own actions
- [Built-in Actions](built-in-actions) - Default actions in the library
- [Action Parameters](action-parameters) - Special parameters for actions
