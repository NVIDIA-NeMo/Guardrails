# Patronus Lynx Deployment

## vLLM

Lynx is fully open source, so you can host it however you like. One simple way is using vLLM.

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

4. Create a model called `patronus_lynx` in your `config.yml` file, setting the host and port to what you set it as above. If the vLLM is running on a different server from `nemoguardrails`, you'll have to replace `localhost` with the vLLM server's address. Check out the guide [here](../guardrails-library.md#patronus-lynx-based-rag-hallucination-detection) for more information.

## Ollama

You can also run Patronus Lynx 8B on your personal computer using Ollama!

1. Install Ollama: https://ollama.com/download.

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

Check out the guide [here](../guardrails-library.md#patronus-lynx-based-rag-hallucination-detection) for more information.
