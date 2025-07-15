# Multi-config API

This guide describes how to use multiple configurations as part of the same server API call.

## Motivation

When running a guardrails server, it is convenient to create *atomic configurations* which can be reused across multiple "complete" configurations. In this guide, we use [these example configurations](https://github.com/NVIDIA/NeMo-Guardrails/tree/develop/examples/server_configs/atomic):
1. `input_checking`: which uses the self-check input rail.
2. `output_checking`: which uses the self-check output rail.
3. `main`: which uses the `gpt-3.5-turbo-instruct` model with no guardrails.

```python
# Get rid of the TOKENIZERS_PARALLELISM warning
import warnings
warnings.filterwarnings('ignore')
```

## Prerequisites

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

## Setup

In this guide, the server is started programmatically, as shown below. This is equivalent to (from the root of the project):

```sh
nemoguardrails server --config=examples/server_configs/atomic
```

```python
import os
from nemoguardrails.server.api import app
from threading import Thread
import uvicorn

def run_server():
    current_path = %pwd
    app.rails_config_path = os.path.normpath(os.path.join(current_path, "..", "..", "..", "examples", "server_configs", "atomic"))

    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")

# Start the server in a separate thread so that you can still use the notebook
thread = Thread(target=run_server)
thread.start()
```

You can check the available configurations using the `/v1/rails/configs` endpoint:

```python
import requests

base_url = "http://127.0.0.1:8000"

response = requests.get(f"{base_url}/v1/rails/configs")
print(response.json())
```

```
[{'id': 'output_checking'}, {'id': 'main'}, {'id': 'input_checking'}]
```

You can make a call using a single config as shown below:

```python
response = requests.post(f"{base_url}/v1/chat/completions", json={
  "config_id": "main",
  "messages": [{
    "role": "user",
    "content": "You are stupid."
  }]
})
print(response.json())
```

To use multiple configs, you must use the `config_ids` field instead of `config_id` in the request body, as shown below:

```python
response = requests.post(f"{base_url}/v1/chat/completions", json={
  "config_ids": ["main", "input_checking"],
  "messages": [{
    "role": "user",
    "content": "You are stupid."
  }]
})
print(response.json())
```

```yaml
{'messages': [{'role': 'assistant', 'content': "I'm sorry, I can't respond to that."}]}
```

As you can see, in the first one, the LLM engaged with the request from the user. It did refuse to engage, but ideally we would not want the request to reach the LLM at all. In the second call, the input rail kicked in and blocked the request.

## Conclusion

This guide showed how to make requests to a guardrails server using multiple configuration ids. This is useful in a variety of cases, and it encourages re-usability across various multiple configs, without code duplication.
