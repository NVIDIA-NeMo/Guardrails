---
title:
  page: Deploy the NeMo Guardrails Library with Docker
  nav: Docker
description: Build and run NeMo Guardrails Docker images for deployment and testing.
topics:
- Deployment
- AI Safety
tags:
- Docker
- Container
- Deployment
- AlignScore
content:
  type: how_to
  difficulty: technical_intermediate
  audience:
  - engineer
  - DevOps Engineer
---

# Deploy the NeMo Guardrails Library with Docker

This guide shows how to run the NVIDIA NeMo Guardrails library using Docker. Docker provides a direct deployment method for getting started with the library.

## Prerequisites

Ensure Docker is installed on your machine. If Docker is not installed, follow the [official Docker installation guide](https://docs.docker.com/get-docker/) for your platform.

## LLM Framework Selection

By default, the library uses the lightweight default framework, which is based on `httpx` and does not require LangChain. It serves engines such as `openai`, `nim`, `nvidia_ai_endpoints`, `ollama`, `azure`, and `azure_openai`. It also serves any other OpenAI-compatible provider configured with `engine: openai` and `parameters.base_url`, such as self-hosted vLLM, TGI, OpenRouter, Together.ai, Fireworks.ai, Groq, DeepSeek, or `llama.cpp`.

To use LangChain-only engines whose API is not OpenAI-compatible, set `NEMOGUARDRAILS_LLM_FRAMEWORK=langchain` in the container environment and add `langchain` plus the relevant provider packages to your image. These engines include `vertexai`, `anthropic`, `cohere`, `huggingface_pipeline`, `huggingface_endpoint` with the default text-generation schema, `trt_llm`, `self_hosted`, and the legacy `vllm_openai` LangChain wrapper. For example:

```bash
docker run \
  -p 8000:8000 \
  -e NEMOGUARDRAILS_LLM_FRAMEWORK=langchain \
  -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
  nemoguardrails
```

Replace `ANTHROPIC_API_KEY` with the credential your provider uses, such as `GOOGLE_APPLICATION_CREDENTIALS` for Vertex AI or `COHERE_API_KEY` for Cohere. For file-based credentials such as Vertex AI service-account JSON, mount the credential file into the container and set the environment variable to the in-container path. For example, bind-mount host `service-account.json` to `/secrets/service-account.json` and set `GOOGLE_APPLICATION_CREDENTIALS=/secrets/service-account.json`. Secure the host credential file and configure equivalent bind-mount and environment settings in `docker run` or Docker Compose.

## Build the Docker Images

To build the Docker images, complete the following steps:

1. Clone the repository.

   ```bash
   git clone https://github.com/NVIDIA-NeMo/Guardrails.git nemoguardrails
   ```

1. Change directory into the repository.

   ```bash
   cd nemoguardrails
   ```

1. Build the `nemoguardrails` Docker image.

   ```bash
   docker build -t nemoguardrails .
   ```

1. Optional: Build the AlignScore server image.

   If you want to use AlignScore-based fact-checking, you can also build a Docker image using the provided [Dockerfile](https://github.com/NVIDIA-NeMo/Guardrails/tree/develop/nemoguardrails/library/factchecking/align_score/Dockerfile).

   ```bash
   cd nemoguardrails/library/factchecking/align_score
   docker build -t alignscore-server .
   ```

   ```{note}
   The provided Dockerfile downloads only the `base` AlignScore image. For large model support, uncomment the corresponding line in the Dockerfile.
   ```

1. Optional: Build the jailbreak detection heuristics server image.

   If you want to use the jailbreak detection heuristics server, you can also build a Docker image using the provided [Dockerfile](https://github.com/NVIDIA-NeMo/Guardrails/tree/develop/nemoguardrails/library/jailbreak_detection/Dockerfile).

   ```bash
   cd nemoguardrails/library/jailbreak_detection
   docker build -t jailbreak_detection_heuristics .
   ```

## Run Using Docker

To run the library server using the Docker image, run the following command:

```bash
docker run -p 8000:8000 -e OPENAI_API_KEY=$OPENAI_API_KEY nemoguardrails
```

This command starts the library server with the example configurations. The Chat UI is accessible at `http://localhost:8000`.

```{note}
Because the example configurations use OpenAI models such as `gpt-3.5-turbo-instruct` and `gpt-4`, you must provide an `OPENAI_API_KEY`.
```

To specify your own config folder for the server, mount your local configuration into the `/config` path in the container:

```bash
docker run \
  -p 8000:8000 \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  -v </path/to/local/config/>:/config \
  nemoguardrails
```

To use the Chat CLI interface, run the Docker container in interactive mode:

```bash
docker run -it \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  -v </path/to/local/config/>:/config \
  nemoguardrails chat --config=/config --verbose
```

## AlignScore Fact-Checking

If one of your configurations uses the AlignScore fact-checking model, run the AlignScore server in a separate container:

```bash
docker run -p 5000:5000 alignscore-server
```

This command starts the AlignScore server on port `5000`. You can then specify the AlignScore server URL in your configuration file:

```yaml
rails:
  config:
    fact_checking:
      # Select AlignScore as the provider
      provider: align_score
      parameters:
        # Point to a running instance of the AlignScore server
        endpoint: "http://localhost:5000/alignscore_base"
```
