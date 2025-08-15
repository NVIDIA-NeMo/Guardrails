# KV Cache Reuse for NemoGuard NIM

When you configure NeMo Guardrails to call NemoGuard NIMs in response to a client request, every NIM call interjecting the input and response adds to the inference latency.
The application LLM can only begin generating a response after all input checks, which may [run in parallel](parallel-rails), are complete. Additionally, response latency is introduced if you run the guardrail checks on the application LLM's response; the larger the response, the longer it takes to check the response.

[KV Cache Reuse](https://docs.nvidia.com/nim/large-language-models/latest/kv-cache-reuse.html) (also known as prefix-caching) is a feature of the NVIDIA NIM for LLMs that provides a performance improvement by reusing the decoder layers for the prompt.

## How Key-Value Cache Reuse Works

For example, the NemoGuard Content Safety NIM is a fine-tuned Llama 3.1-Instruct using LoRA, and then merging the LoRA weights back into the model weights. When you send requests to the Guardrails client, it calls the Content Safety NIM with the same prompt used for fine-tuning, and inserts the user-supplied query and optional LLM response. The Content Safety NIM responds with a JSON object that classifies the user and response as safe or unsafe.

Key-Value (KV) cache reuse is the most effective for LLM NIMs that use the same system prompt for all calls up to the point where user query and LLM response are injected. For example, the [system prompt for the NemoGuard Content Safety NIM](https://docs.api.nvidia.com/nim/reference/nvidia-llama-3_1-nemoguard-8b-content-safety#prompt-format) is about 370 tokens long before the user and LLM response are added. With KV cache reuse, recomputing the decoder layers for these tokens is only necessary on the first inference call. This means that, when the application LLM's response is typically small, the overall latency is heavily dependent on the prefill stage rather than the generation. For more information about pre-fill and decoding phases in application LLMs, see the blog post [Mastering LLM Techniques: Inference Optimization](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/).

You can enable KV cache reuse by setting the `NIM_ENABLE_KV_CACHE_REUSE` variable to `1`.

## Code Sample

To enable KV cache reuse for the Content Safety NIM, set the `NIM_ENABLE_KV_CACHE_REUSE` environment variable to `1` when you run the Docker container for the NemoGuard NIM microservice.

For example, to run the Content Safety NemoGuard NIM microservice with KV cache reuse, add `NIM_ENABLE_KV_CACHE_REUSE=1` to the `docker run` command as follows:

```bash
export MODEL_NAME="nemoguard-nim-name"
export NIM_IMAGE=<nemoguard-nim-image-uri>
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

If you have an existing Docker container running the NIM, you can update the environment variable by running the following command:

```bash
docker exec -it $MODEL_NAME bash -c "export NIM_ENABLE_KV_CACHE_REUSE=1"
```
