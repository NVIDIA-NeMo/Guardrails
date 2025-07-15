# Jailbreak Detection Deployment

The recommended way to use Jailbreak Detection Heuristics and models with NeMo Guardrails is using the provided [Dockerfile](https://github.com/NVIDIA/NeMo-Guardrails/blob/develop/nemoguardrails/library/jailbreak_detection/Dockerfile).
For more details, check out how to [build and use the image](using-docker.md).

If you wish to use the NemoGuard JailbreakDetect NIM, please see the [related documentation](nemoguard-jailbreakdetect-deployment.md).

In order to deploy the jailbreak detection server, follow these steps:

1. Install the dependencies

```bash
pip install transformers torch uvicorn nemoguardrails
```

2. Start the jailbreak detection server

```bash
python -m nemoguardrails.library.jailbreak_detection.server --port 1337
```

By default, the jailbreak detection server listens on port `1337`. You can change the port using the `--port` option.

## Running on GPU

To run on GPU, ensure you have the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed.
If you are building a container from the provided dockerfiles, make sure that you specify the correct [Dockerfile](https://github.com/NVIDIA/NeMo-Guardrails/blob/develop/nemoguardrails/library/jailbreak_detection/Dockerfile-GPU) and include the `-f` parameter with `docker build`.
When running docker, ensure you pass the `-e NVIDIA_DRIVER_CAPABILITIES=compute,utility`, `-e NVIDIA_VISIBLE_DEVICES=all` and the `--runtime=nvidia` argument to `docker run`.

``` bash
docker run -ti --runtime=nvidia -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -e NVIDIA_VISIBLE_DEVICES=all <image_name>
```
