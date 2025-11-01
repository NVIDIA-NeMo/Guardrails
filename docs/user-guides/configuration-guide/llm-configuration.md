(llm-configuration)=

# LLM Configuration

## The LLM Model

To configure the main LLM model that will be used by the guardrails configuration, you set the `models` key as shown below:

```yaml
models:
  - type: main
    engine: openai
    model: gpt-3.5-turbo-instruct
```

The meaning of the attributes is as follows:

- `type`: is set to _main_ to indicate the model is the application LLM.
- `engine`: the LLM provider, such as `openai`, `huggingface_endpoint`, `self_hosted`, and so on.
- `model`: the name of the model, such as `gpt-3.5-turbo-instruct`.
- `parameters`: arguments to pass to the LangChain class used by the LLM provider.
  For example, when `engine` is set to `openai`, the toolkit loads the `ChatOpenAI` class.
  The [ChatOpenAI class](https://python.langchain.com/api_reference/openai/chat_models/langchain_openai.chat_models.base.ChatOpenAI.html)
  supports `temperature`, `max_tokens`, and other class-specific arguments.

### Supported LLM Providers

You can use any LLM provider that is supported by LangChain, such as `ai21`, `aleph_alpha`, `anthropic`, `anyscale`, `azure`, `cohere`, `huggingface_endpoint`, `huggingface_hub`, `openai`, `self_hosted`, `self_hosted_hugging_face`. Check out the LangChain official documentation for the full list.

In addition to the above LangChain providers, connecting to [NVIDIA NIM microservices](https://docs.nvidia.com/nim/index.html) is supported using the `nim` engine.
The `nvidia_ai_endpoints` engine is an alias for the `nim` engine.
The engine provides access to locally-deployed NIM microservices or NVIDIA hosted models that you can view from <https://build.nvidia.com/models>.

To use any of the LLM providers, you must install the LangChain package for the provider.
When you first try to use a configuration with a new provider, you typically receive an error from LangChain that instructs which packages you should install.

```{important}
Although you can instantiate any of the previously mentioned LLM providers, depending on the capabilities of the model, the NeMo Guardrails toolkit works better with some providers than others.
The toolkit includes prompts that have been optimized for certain types of models, such as models provided by `openai` or `llama3` models.
For others, you can optimize the prompts yourself following the information in the [LLM Prompts](../general-options.md#llm-prompts) section.
```

### Exploring Available Providers

To help you explore and select the right LLM provider for your needs, NeMo Guardrails provides the `find-providers` command. This command offers an interactive interface to discover available providers:

```bash
nemoguardrails find-providers [--list]
```

The command supports two modes:

- Interactive mode (default): Guides you through selecting a provider type (text completion or chat completion) and then shows available providers for that type
- List mode (`--list`): Simply lists all available providers without interactive selection

This can be particularly helpful when you're setting up your configuration and need to explore which providers are available and supported.

For more details about the command and its usage, see the [CLI documentation](../cli.md#find-providers-command).

### Using LLMs with Reasoning Traces

```{deprecated} 0.18.0
The `reasoning_config` field and its options `remove_reasoning_traces`, `start_token`, and `end_token` are deprecated. The `rails.output.apply_to_reasoning_traces` field has also been deprecated. Instead, use output rails to guardrail reasoning traces, as introduced in this section.
```

Reasoning-capable LLMs such as [DeepSeek-R1](https://huggingface.co/collections/deepseek-ai/deepseek-r1-678e1e131c0169c0bc89728d) and [NVIDIA Llama 3.1 Nemotron Ultra 253B V1](https://build.nvidia.com/nvidia/llama-3_1-nemotron-ultra-253b-v1) include reasoning traces in their responses, typically wrapped in tokens such as `<think>` and `</think>`.

The NeMo Guardrails toolkit automatically extracts these traces and makes them available to set up in your guardrails configuration through the following variables:

- In Colang flows, use the `$bot_thinking` variable.
- In Python contexts, use the `bot_thinking` variable.

#### Guardrailing Reasoning Traces with Output Rails

Use output rails to inspect and control reasoning traces. This allows you to:

- Block responses based on problematic reasoning patterns.
- Enhance moderation decisions with reasoning context.
- Monitor and filter sensitive information in reasoning.

##### Prepare Configuration Files

The following configuration files show a minimal configuration for guardrailing reasoning traces with output rails.

1. Configure output rails in `config.yml`:

    ```yaml
    models:
      - type: main
        engine: nim
        model: nvidia/llama-3.1-nemotron-ultra-253b-v1
      - type: self_check_output
        model: <your_moderation_model>
        engine: <your_engine>

    rails:
      output:
        flows:
          - self check output
    ```

1. Configure the prompt to access the reasoning traces in `prompts.yml`:

    ```yaml
    prompts:
      - task: self_check_output
        content: |
          Your task is to check if the bot message complies with company policy.

          Bot message: "{{ bot_response }}"

          {% if bot_thinking %}
          Bot reasoning: "{{ bot_thinking }}"
          {% endif %}

          Should this be blocked (Yes or No)?
          Answer:
    ```

For more detailed examples of guardrailing reasoning traces, refer to [Guardrailing Bot Reasoning Content](../../advanced/bot-thinking-guardrails.md).

#### Accessing Reasoning Traces in API Responses

There are two ways to access reasoning traces in API responses: with generation options and without generation options.

Read the option **With GenerationOptions** when you:

- Need structured access to reasoning and response separately.
- Are building a new application.
- Need access to other structured fields such as state, output_data, or llm_metadata.

Read the option **Without GenerationOptions** when you:

- Need backward compatibility with existing code.
- Want the raw response with inline reasoning tags.
- Are integrating with systems that expect tagged strings.

##### With GenerationOptions for Structured Access

When you pass `GenerationOptions` to the API, the function returns a `GenerationResponse` object with structured fields. This approach provides clean separation between the reasoning traces and the final response content, making it easier to process each component independently.

The `reasoning_content` field contains the extracted reasoning traces, while `response` contains the main LLM response. This structured access pattern is recommended for new applications as it provides type safety and clear access to all response metadata.

The following example demonstrates how to use `GenerationOptions` in an guardrails async generation call `rails.generate_async` to access reasoning traces.

```python
from nemoguardrails import RailsConfig, LLMRails
from nemoguardrails.rails.llm.options import GenerationOptions

# Load the guardrails configuration
config = RailsConfig.from_path("./config")
rails = LLMRails(config)

# Create a GenerationOptions object to enable structured responses
options = GenerationOptions()

# Make an async call with GenerationOptions
result = await rails.generate_async(
    messages=[{"role": "user", "content": "What is 2+2?"}],
    options=options
)

# Access reasoning traces separately from the response
if result.reasoning_content:
    print("Reasoning:", result.reasoning_content)

# Access the main response content
print("Response:", result.response[0]["content"])
```

The following example output shows the reasoning traces and the main response content from the guardrailed generation result.

```
Reasoning: Let me calculate: 2 plus 2 equals 4.
Response: The answer is 4.
```

##### Without GenerationOptions for Tagged String

When calling without `GenerationOptions`, such as by using a dict or string response, reasoning is wrapped in `<think>` tags.

The following example demonstrates how to access reasoning traces without using `GenerationOptions`.

```python
response = rails.generate(
    messages=[{"role": "user", "content": "What is 2+2?"}]
)

print(response["content"])
```

The response is wrapped in `<think>` tags as shown in the following example output.

```
<think>Let me calculate: 2 plus 2 equals 4.</think>
The answer is 4.
```

### NIM for LLMs

[NVIDIA NIM](https://docs.nvidia.com/nim/index.html) is a set of easy-to-use microservices designed to accelerate the deployment of generative AI models across the cloud, data center, and workstations.
[NVIDIA NIM for LLMs](https://docs.nvidia.com/nim/large-language-models/latest/introduction.html) brings the power of state-of-the-art LLMs to enterprise applications, providing unmatched natural language processing and understanding capabilities. [Learn more about NIMs](https://developer.nvidia.com/blog/nvidia-nim-offers-optimized-inference-microservices-for-deploying-ai-models-at-scale/).

NIMs can be self hosted, using downloadable containers, or Nvidia hosted and accessible through an Nvidia AI Enterprise (NVAIE) licesnse.

NeMo Guardrails supports connecting to NIMs as follows:

#### Self-hosted NIMs

To connect to self-hosted NIMs, set the engine to `nim`. Also make sure the model name matches one of the model names the hosted NIM supports (you can get a list of supported models using a GET request to v1/models endpoint).

```yaml
models:
  - type: main
    engine: nim
    model: <MODEL_NAME>
    parameters:
      base_url: <NIM_ENDPOINT_URL>
```

For example, to connect to a locally deployed `meta/llama3-8b-instruct` model, on port 8000, use the following model configuration:

```yaml
models:
  - type: main
    engine: nim
    model: meta/llama3-8b-instruct
    parameters:
      base_url: http://localhost:8000/v1
```

#### NVIDIA AI Endpoints

[NVIDIA AI Endpoints](https://www.nvidia.com/en-us/ai-data-science/foundation-models/) give users easy access to NVIDIA hosted API endpoints for NVIDIA AI Foundation Models such as Llama 3, Mixtral 8x7B, and Stable Diffusion.
These models, hosted on the [NVIDIA API catalog](https://build.nvidia.com/), are optimized, tested, and hosted on the NVIDIA AI platform, making them fast and easy to evaluate, further customize, and seamlessly run at peak performance on any accelerated stack.

To use an LLM model through the NVIDIA AI Endpoints, use the following model configuration:

```yaml
models:
  - type: main
    engine: nim
    model: <MODEL_NAME>
```

For example, to use the `llama3-8b-instruct` model, use the following model configuration:

```yaml
models:
  - type: main
    engine: nim
    model: meta/llama3-8b-instruct
```

```{important}
To use the `nvidia_ai_endpoints` or `nim` LLM provider, you must install the `langchain-nvidia-ai-endpoints` package using the command `pip install langchain-nvidia-ai-endpoints`, and configure a valid `NVIDIA_API_KEY`.
```

For further information, see the [user guide](./llm/nvidia-ai-endpoints/README.md).

Here's an example configuration for using `llama3` model with [Ollama](https://ollama.com/):

```yaml
models:
  - type: main
    engine: ollama
    model: llama3
    parameters:
      base_url: http://your_base_url
```

### TRT-LLM

NeMo Guardrails also supports connecting to a TRT-LLM server.

```yaml
models:
  - type: main
    engine: trt_llm
    model: <MODEL_NAME>
```

Below is the list of supported parameters with their default values. Please refer to TRT-LLM documentation for more details.

```yaml
models:
  - type: main
    engine: trt_llm
    model: <MODEL_NAME>
    parameters:
      server_url: <SERVER_URL>
      temperature: 1.0
      top_p: 0
      top_k: 1
      tokens: 100
      beam_width: 1
      repetition_penalty: 1.0
      length_penalty: 1.0
```

## Configuring LLMs per Task

The interaction with the LLM is structured in a task-oriented manner. Each invocation of the LLM is associated with a specific task. These tasks are integral to the guardrail process and include:

1. `generate_user_intent`: This task transforms the raw user utterance into a canonical form. For instance, "Hello there" might be converted to `express greeting`.
2. `generate_next_steps`: This task determines the bot's response or the action to be executed. Examples include `bot express greeting` or `bot respond to question`.
3. `generate_bot_message`: This task decides the exact bot message to be returned.
4. `general`: This task generates the next bot message based on the history of user and bot messages. It is used when there are no dialog rails defined (i.e., no user message canonical forms).

For a comprehensive list of tasks, refer to the [Task type](https://github.com/NVIDIA/NeMo-Guardrails/blob/develop/nemoguardrails/llm/types.py).

You can use different LLM models for specific tasks. For example, you can use a different model for the `self_check_input` and `self_check_output` tasks from various providers. Here's an example configuration:

```yaml

models:
  - type: main
    model: meta/llama-3.1-8b-instruct
    engine: nim
  - type: self_check_input
    model: meta/llama3-8b-instruct
    engine: nim
  - type: self_check_output
    model: meta/llama-3.1-70b-instruct
    engine: nim
```

In the previous example, the `self_check_input` and `self_check_output` tasks use different models. It is even possible to get more granular and use different models for a task like `generate_user_intent`:

```yaml
models:
  - type: main
    model: meta/llama-3.1-8b-instruct
    engine: nim
  - type: self_check_input
    model: meta/llama3-8b-instruct
    engine: nim
  - type: self_check_output
    model: meta/llama-3.1-70b-instruct
    engine: nim
  - type: generate_user_intent
    model: meta/llama-3.1-8b-instruct
    engine: nim
```

```{tip}
Remember, the best model for your needs will depend on your specific requirements and constraints. It's often a good idea to experiment with different models to see which one works best for your specific use case.
```

## The Embeddings Model

To configure the embedding model used for the various steps in the [guardrails process](../architecture/README.md), such as canonical form generation and next step generation, add a model configuration in the `models` key as shown in the following configuration file:

```yaml
models:
  - ...
  - type: embeddings
    engine: FastEmbed
    model: all-MiniLM-L6-v2
```

The `FastEmbed` engine is the default one and uses the `all-MiniLM-L6-v2` model. NeMo Guardrails also supports using OpenAI models for computing the embeddings, e.g.:

```yaml
models:
  - ...
  - type: embeddings
    engine: openai
    model: text-embedding-ada-002
```

### Supported Embedding Providers

The following tables lists the supported embedding providers:

| Provider Name        | `engine_name`          | `model`                            |
|----------------------|------------------------|------------------------------------|
| FastEmbed (default)  | `FastEmbed`            | `all-MiniLM-L6-v2` (default), etc. |
| OpenAI               | `openai`               | `text-embedding-ada-002`, etc.     |
| SentenceTransformers | `SentenceTransformers` | `all-MiniLM-L6-v2`, etc.           |
| NVIDIA AI Endpoints  | `nvidia_ai_endpoints`  | `nv-embed-v1`, etc.                |

```{note}
You can use any of the supported models for any of the supported embedding providers.
The previous table includes an example of a model that can be used.
```

### Embedding Search Provider

NeMo Guardrails uses embedding search, also called vector databases, for implementing the [guardrails process](../architecture/README.md#the-guardrails-process) and for the [knowledge base](knowledge-base.md) functionality. The default embedding search uses FastEmbed for computing the embeddings (the `all-MiniLM-L6-v2` model) and [Annoy](https://github.com/spotify/annoy) for performing the search. As shown in the previous section, the embeddings model supports both FastEmbed and OpenAI. SentenceTransformers is also supported.

For advanced use cases or integrations with existing knowledge bases, you can [provide a custom embedding search provider](advanced/embedding-search-providers.md).
