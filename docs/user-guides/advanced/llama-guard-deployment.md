# Self-hosting Llama Guard using vLLM

Detailed below are steps to self-host Llama Guard using vLLM and HuggingFace. Alternatively, you can do this using your own custom inference code with the downloaded model weights, too.

1. Get access to the Llama Guard model from Meta on HuggingFace. See [this page](https://huggingface.co/meta-llama/LlamaGuard-7b) for more details.

2. Log in to Hugging Face with your account token
```sh
huggingface-cli login
```

3. Here, we use vLLM to host a Llama Guard inference endpoint in the OpenAI-compatible mode.

```sh
pip install vllm
python -m vllm.entrypoints.openai.api_server --port 5123 --model meta-llama/LlamaGuard-7b
```

This will serve up the vLLM inference server on `http://localhost:5123/`.

4. Set the host and port in your bot's YAML configuration files ([example config](https://github.com/NVIDIA/NeMo-Guardrails/blob/develop/examples/configs/llama_guard/README.md)). If you're running the `nemoguardrails` app on another server, remember to replace `localhost` with your vLLM server's public IP address.
