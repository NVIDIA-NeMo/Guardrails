# Patronus Lynx Integration

NeMo Guardrails supports hallucination detection in RAG systems using [Patronus AI](www.patronus.ai)'s Lynx model. The model is hosted on Hugging Face and comes in both a 70B parameters (see [here](https://huggingface.co/PatronusAI/Patronus-Lynx-70B-Instruct)) and 8B parameters (see [here](https://huggingface.co/PatronusAI/Patronus-Lynx-8B-Instruct)) variant.

There are three components of hallucination that Lynx checks for:

- Information in the `bot_message` is contained in the `relevant_chunks`
- There is no extra information in the `bot_message` that is not in the `relevant_chunks`
- The `bot_message` does not contradict any information in the `relevant_chunks`

## Setup

Since Patronus Lynx is fully open source, you can host it however you like. You can find a guide to host Lynx using vLLM [here](patronus-lynx-deployment.md).

## Usage

Here is how to configure your bot to use Patronus Lynx to check for RAG hallucinations in your bot output:

1. Add a model of type `patronus_lynx` in `config.yml` - the example below uses vLLM to run Lynx:

```yaml
models:
  ...

  - type: patronus_lynx
    engine: vllm_openai
    parameters:
      openai_api_base: "http://localhost:5000/v1"
      model_name: "PatronusAI/Patronus-Lynx-70B-Instruct" # "PatronusAI/Patronus-Lynx-8B-Instruct"
```

2. Add the guardrail name is `patronus lynx check output hallucination` to your output rails in `config.yml`:

```yaml
rails:
  output:
    flows:
      - patronus lynx check output hallucination
```

3. Add a prompt for `patronus_lynx_check_output_hallucination` in the `prompts.yml` file:

```yaml
prompts:
  - task: patronus_lynx_check_output_hallucination
    content: |
      Given the following QUESTION, DOCUMENT and ANSWER you must analyze ...
      ...
```

We recommend you base your Lynx hallucination detection prompt off of the provided example [here](https://github.com/NVIDIA/NeMo-Guardrails/tree/develop/examples/configs/patronusai/prompts.yml).

Under the hood, the `patronus lynx check output hallucination` rail runs the `patronus_lynx_check_output_hallucination` action, which you can find [here](https://github.com/NVIDIA/NeMo-Guardrails/tree/develop/nemoguardrails/library/patronusai/actions.py). It returns whether a hallucination is detected (`True` or `False`) and potentially a reasoning trace explaining the decision. The bot's response will be blocked if hallucination is `True`. Note: If Lynx's outputs are misconfigured or a hallucination decision cannot be found, the action default is to return `True` for hallucination.

Here's the `patronus lynx check output hallucination` flow, showing how the action is executed:

```colang
define bot inform answer unknown
  "I don't know the answer to that."

define flow patronus lynx check output hallucination
  $patronus_lynx_response = execute patronus_lynx_check_output_hallucination
  $hallucination = $patronus_lynx_response["hallucination"]
  # The Reasoning trace is currently unused, but can be used to modify the bot output
  $reasoning = $patronus_lynx_response["reasoning"]

  if $hallucination
    bot inform answer unknown
    stop
```
