# Patronus Lynx Integration

NeMo Guardrails supports hallucination detection in RAG systems using [Patronus AI](https://www.patronus.ai)'s Lynx model. The model is hosted on Hugging Face and comes in both a 70B parameters (see [here](https://huggingface.co/PatronusAI/Patronus-Lynx-70B-Instruct)) and 8B parameters (see [here](https://huggingface.co/PatronusAI/Patronus-Lynx-8B-Instruct)) variant.

There are three components of hallucination that Lynx checks for:

- Information in the `bot_message` is contained in the `relevant_chunks`
- There is no extra information in the `bot_message` that is not in the `relevant_chunks`
- The `bot_message` does not contradict any information in the `relevant_chunks`

## Setup

Because Patronus Lynx is completely open source, you have the flexibility to deploy it in any environment you prefer. An easy method for hosting is to use vLLM.

1. Get access to Patronus Lynx on HuggingFace. See [here](https://huggingface.co/PatronusAI/Patronus-Lynx-70B-Instruct) for the 70B parameters variant, and [here](https://huggingface.co/PatronusAI/Patronus-Lynx-8B-Instruct) for the 8B parameters variant. The examples below use the `70B` parameters model, but there's no additional configuration to deploy the smaller model, so you can swap the model name references out with `8B`.

2. Log in to Hugging Face

    ```bash
    huggingface-cli login
    ```

3. Install vLLM and spin up a server hosting Patronus Lynx

    ```bash
    pip install vllm
    python -m vllm.entrypoints.openai.api_server --port 5000 --model PatronusAI/Patronus-Lynx-70B-Instruct
    ```

    This will launch the vLLM inference server on `http://localhost:5000/`. You can use the OpenAI API spec to send it a cURL request to make sure it works:

    ```bash
    curl http://localhost:5000/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{
      "model": "PatronusAI/Patronus-Lynx-70B-Instruct",
      "messages": [
       {"role": "user", "content": "What is a hallucination?"},
      ]
    }'
    ```

4. Create a model called `patronus_lynx` in your `config.yml` file, setting the host and port to what you set it as above. If the vLLM is running on a different server from `nemoguardrails`, you'll have to replace `localhost` with the vLLM server's address. Check out the [Patronus Lynx Integration](patronus-lynx.md) guide for more information.

## Ollama

You can also run Patronus Lynx 8B on your personal computer using Ollama!

1. Install [Ollama](https://ollama.com/download).

2. Get access to a GGUF quantized version of Lynx 8B on Huggingface. Check it out [here](https://huggingface.co/PatronusAI/Lynx-8B-Instruct-Q4_K_M-GGUF).

3. Download the gguf model from the repository [here](https://huggingface.co/PatronusAI/Lynx-8B-Instruct-Q4_K_M-GGUF/blob/main/patronus-lynx-8b-instruct-q4_k_m.gguf). This may take a few minutes.

4. Create a file called `Modelfile` with the following contents:

    ```bash
    FROM "./patronus-lynx-8b-instruct-q4_k_m.gguf"
    PARAMETER stop "<|im_start|>"
    PARAMETER stop "<|im_end|>"
    TEMPLATE """
    <|im_start|>system
    {{ .System }}<|im_end|>
    <|im_start|>user
    {{ .Prompt }}<|im_end|>
    <|im_start|>assistant
    ```

    Ensure that the `FROM` field correctly points to the `patronus-lynx-8b-instruct-q4_k_m.gguf` file you downloaded in Step 3.

5. Run `ollama create patronus-lynx-8b -f Modelfile`.

6. Run `ollama run patronus-lynx-8b`. You should now be able to chat with `patronus-lynx-8b`!

7. Create a model called `patronus_lynx` in your `config.yml` file, like this:

    ```yaml
    models:
      ...

      - type: patronus_lynx
        engine: ollama
        model: patronus-lynx-8b
        parameters:
          base_url: "http://localhost:11434"
    ```

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

2. Add the guardrail `patronus lynx check output hallucination` to your output rails in `config.yml`:

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

We recommend you base your Lynx hallucination detection prompt off of the provided example [here](https://github.com/NVIDIA-NeMo/Guardrails/tree/develop/examples/configs/patronusai/prompts.yml).

Under the hood, the `patronus lynx check output hallucination` rail runs the `patronus_lynx_check_output_hallucination` action, which you can find [here](https://github.com/NVIDIA-NeMo/Guardrails/tree/develop/nemoguardrails/library/patronusai/actions.py). It returns whether a hallucination is detected (`True` or `False`) and potentially a reasoning trace explaining the decision. The bot's response will be blocked if hallucination is `True`. Note: If Lynx's outputs are misconfigured or a hallucination decision cannot be found, the action default is to return `True` for hallucination.

Here's the `patronus lynx check output hallucination` flow, showing how the action is executed:

```text
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
