# Llama 3.1 NemoGuard 8B ContentSafety Deployment

The Llama 3.1 NemoGuard 8B ContentSafety model will be available to download as a LoRA adapter module through HuggingFace, and as an [NVIDIA NIM](https://docs.nvidia.com/nim/#nemoguard) for low latency optimized inference with [NVIDIA TensorRT-LLM](https://docs.nvidia.com/tensorrt-llm/index.html).

This guide covers how to deploy [NemoGuard NIMs](https://docs.nvidia.com/nim/#nemoguard), and how to then use the deployed NIM in a NeMo Guardrails configuration.

## NIM Deployment

### Access

The first step is to ensure access to NVIDIA NIM assets through NGC using an NVAIE license.
Once you have the NGC API key with the necessary permissions, set the following environment variables:

```bash
export NGC_API_KEY=<your NGC API key>
docker login nvcr.io -u '$oauthtoken' -p <<< <your NGC API key>
```

Test that you are able to use the NVIDIA NIM assets through by pulling the latest NemoGuard container.

```bash
export NIM_IMAGE=<Path to latest NIM docker container>
export MODEL_NAME="llama-3.1-nemoguard-8b-content-safety"
docker pull $NIM_IMAGE
```

And go!

```bash
docker run -it --name=$MODEL_NAME \
    --gpus=all --runtime=nvidia \
    -e NGC_API_KEY="$NGC_API_KEY" \
    -e NIM_SERVED_MODEL_NAME=$MODEL_NAME \
    -e NIM_CUSTOM_MODEL_NAME=$MODEL_NAME \
    -u $(id -u) \
    -p 8123:8000 \
    $NIM_IMAGE
```

#### Use the running NIM in your Guardrails App

Any locally running NIM exposes the standard OpenAI interface on the `v1/completions` and `v1/chat/completions` endpoints. NeMo Guardrails provides out of the box support engines that support the standard LLM interfaces. One such engine is LangChain's `vllm_openai`.

Thus, your Guardrails configuration file can look like:

```yaml
models:
  - type: main
    engine: openai
    model: gpt-3.5-turbo-instruct

  - type: "content_safety"
    engine: nim
    parameters:
      base_url: "http://localhost:8123/v1"
      model_name: "llama-3.1-nemoguard-8b-content-safety"

rails:
  input:
    flows:
      - content safety check input $model=content_safety
  output:
    flows:
      - content safety check output $model=content_safety
```

A few things to note:

- `parameters.base_url` should contain the IP address of the machine the NIM was hosted on, the port should match the tunnel forwarding port specified in the docker run command.
- `parameters.model_name` in the Guardrails configuration needs to match the `$MODEL_NAME` used when running the NIM container.
- The `rails` definitions should list `llama-3.1-nemoguard-8b-content-safety` as the model.

#### Bonus: Caching the optimized TRTLLM inference engines

If you'd like to not build TRTLLM engines from scratch every time you run the NIM container, you can cache it in the first run by just adding a flag to mount a local directory inside the docker to store the model cache.

To achieve this, you simply need to mount the folder containing the cached TRTLLM assets onto the docker container while running it using `-v $LOCAL_NIM_CACHE:/opt/nim/.cache`. See below instructions for the full command. Important: make sure that docker has permissions to write to the cache folder (`sudo chmod 777 $LOCAL_NIM_CACHE`).

```bash
### To bind a $LOCAL_NIM_CACHE folder to "/opt/nim/.cache"
export LOCAL_NIM_CACHE=<PATH TO DIRECTORY WHERE YOU WANT TO SAVE TRTLLM ENGINE ASSETS>
mkdir -p $LOCAL_NIM_CACHE
sudo chmod 666 $LOCAL_NIM_CACHE
```

Now mount this directory while running the docker container to store cached assets in this directory, so that mounting it subsequently will cause the container to read the cached assets instead of rebuilding them.

```bash
docker run -it --name=$MODEL_NAME \
    --gpus=all --runtime=nvidia \
    -e NGC_API_KEY="$NGC_API_KEY" \
    -e NIM_SERVED_MODEL_NAME=$MODEL_NAME \
    -e NIM_CUSTOM_MODEL_NAME=$MODEL_NAME \
    -v $LOCAL_NIM_CACHE:"/opt/nim/.cache/" \
    -u $(id -u) \
    -p 8123:8000 \
    $NIM_IMAGE
```
