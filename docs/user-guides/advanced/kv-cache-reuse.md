# KV Cache Reuse for NemoGuard NIM

The NeMo Guardrails client calls NemoGuard NIMs, which are specialized LLMs for specific guardrail types, such as content safety and topic control. A jailbreak-detection NIM also exists to protect against security bypasses.

Every NIM call interjecting the application LLM adds to the inference latency. The application LLM can only begin generating a response after all input checks, which may [run in parallel](parallel-rails), are complete.

[KV Cache Reuse](https://docs.nvidia.com/nim/large-language-models/latest/kv-cache-reuse.html) (also known as prefix-caching) is a feature of the NVIDIA NIM for LLMs that provides a performance improvement by reusing the decoder layers for the prompt.

## How KV Cache Reuse Works

For example, the NemoGuard Content Safety NIM is a fine-tuned Llama 3.1-Instruct using LoRA, and then merging the LoRA weights back into the model weights. When you send requests to the Guardrails client, it calls the Content Safety NIM with the same prompt used for fine-tuning, and inserts the user-supplied query and optional LLM response. The Content Safety NIM responds with a JSON object that classifies the user and response as safe or unsafe.

KV cache reuse is the most effective for NIMs that use the same system prompt for all calls up to the point where user query and LLM response are injected. For example, the [system prompt for the NemoGuard Content Safety NIM](https://docs.api.nvidia.com/nim/reference/nvidia-llama-3_1-nemoguard-8b-content-safety#prompt-format) is about 370 tokens long before the user and LLM response are added. With KV cache reuse, recomputing the decoder layers for these tokens is only necessary on the first inference call. The NIM's response is an order of magnitude smaller than the prompt. This means that the overall latency is heavily dependent on the prefill stage rather than the generation.

You can enable KV cache reuse by setting the `NIM_ENABLE_KV_CACHE_REUSE` variable to `1` for every Content Safety NIM deployment.

## Code Sample

To enable KV cache reuse for the Content Safety NIM, set the `NIM_ENABLE_KV_CACHE_REUSE` environment variable to `1` when running the Docker container.

To run the Content Safety NIM server with KV cache reuse, use the following commands:

```bash
export MODEL_NAME="llama-3.1-nemoguard-8b-content-safety"
export NIM_IMAGE=<llama-3.1-nemoguard-8b-content-safety-image-uri>
export LOCAL_NIM_CACHE=<local-nim-cache-directory>

docker run -it \
    --name=$MODEL_NAME \
    --network=host \
    --gpus='"device=0"' \
    --memory=16g \
    --cpus=4 \
    --runtime=nvidia \
    -e NIM_ENABLE_KV_CACHE_REUSE=1 \
    -e NGC_API_KEY="$NGC_API_KEY" \
    -e NIM_SERVED_MODEL_NAME=$MODEL_NAME \
    -e NIM_CUSTOM_MODEL_NAME=$MODEL_NAME \
    -v $LOCAL_NIM_CACHE:"/opt/nim/.cache/" \
    -u $(id -u) \
    -p 8000:8000 \
    $NIM_IMAGE
```

To disable KV cache reuse, you can either remove the `-e NIM_ENABLE_KV_CACHE_REUSE=1` line or set the variable to `0`.
