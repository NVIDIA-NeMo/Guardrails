# Llama 3.1 NemoGuard 8B Topic Control Deployment

The TopicControl model is available to download as a LoRA adapter module through Hugging Face or as an [NVIDIA TopicControl NIM microservice](https://docs.nvidia.com/nim/llama-3-1-nemoguard-8b-topiccontrol/latest/index.html) for low-latency optimized inference with [NVIDIA TensorRT-LLM](https://docs.nvidia.com/tensorrt-llm/index.html).

This guide covers how to deploy the TopicControl model as a NIM microservice and use it in a NeMo Guardrails configuration.

## NIM Deployment

Follow the instructions below to deploy the TopicControl NIM microservice and configure it in a NeMo Guardrails application.

### Access

The first step is to ensure access to NVIDIA NIM assets through NGC using an NVAIE license.
Once you have the NGC API key with the necessary permissions, set the following environment variables:

```bash
export NGC_API_KEY=<your NGC API key>
docker login nvcr.io -u '$oauthtoken' -p <<< <your NGC API key>
```

Test that you are able to use the NVIDIA NIM assets through by pulling the latest TopicControl container.

```bash
export NIM_IMAGE=<Path to latest NIM docker container>
export MODEL_NAME="llama-3.1-nemoguard-8b-topic-control"
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

### Use TopicControl NIM Microservice in NeMo Guardrails App

A locally running TopicControl NIM microservice exposes the standard OpenAI interface on the `v1/chat/completions` endpoint. NeMo Guardrails provides out-of-the-box support for engines that support the standard LLM interfaces. In Guardrails configuration, use the engine `nim` for the TopicControl NIM microservice as follows.

```yaml
models:
  - type: main
    engine: openai
    model: gpt-3.5-turbo-instruct

  - type: "topic_control"
    engine: nim
    parameters:
      base_url: "http://localhost:8123/v1"
      model_name: "llama-3.1-nemoguard-8b-topic-control"

rails:
  input:
    flows:
      - topic safety check input $model=topic_control
```

A few things to note:

- `parameters.base_url` should contain the IP address of the machine the NIM was hosted on, the port should match the tunnel forwarding port specified in the docker run command.
- `parameters.model_name` in the Guardrails configuration needs to match the `$MODEL_NAME` used when running the NIM container.
- The `rails` definitions should list `topic_control` as the model.

### Bonus: Caching the optimized TRTLLM inference engines

If you'd like to not build TRTLLM engines from scratch every time you run the NIM container, you can cache it in the first run by just adding a flag to mount a local directory inside the docker to store the model cache.

To achieve this, you simply need to mount the folder containing the cached TRTLLM assets onto the docker container while running it using `-v $LOCAL_NIM_CACHE:/opt/nim/.cache`. See below instructions for the full command. Important: make sure that docker has permissions to write to the cache folder (`sudo chmod 666 $LOCAL_NIM_CACHE`).

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

## More details on TopicControl model

For more details on the TopicControl model, check out the other resources:

- NeMo Guardrails library for [NVIDIA NemoGuard models](../guardrails-library.md#nvidia-models)
- TopicControl topic safety example [configuration and prompts](https://github.com/NVIDIA/NeMo-Guardrails/tree/develop/examples/configs/topic_safety)
- [Paper at EMNLP 2024](https://arxiv.org/abs/2404.03820)
