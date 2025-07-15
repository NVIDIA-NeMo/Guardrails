# NeMo Guardrails with Docker

## Introduction

This guide provides step-by-step instructions for running NeMo Guardrails using Docker. Docker offers a seamless and rapid deployment method for getting started with NeMo Guardrails.

## Prerequisites

Ensure Docker is installed on your machine. If not, follow the [official Docker installation guide](https://docs.docker.com/get-docker/) for your respective platform.

## Build the Docker Images

### 1. Clone the repository

Start by cloning the NeMo Guardrails repository:

```bash
git clone https://github.com/NVIDIA/NeMo-Guardrails.git nemoguardrails
```

And change directory into the repository:

```bash
cd nemoguardrails
```

### 2. Build the Docker image

Build the `nemoguardrails` Docker image:

```bash
docker build -t nemoguardrails .
```

### 3. \[Optional] Build the AlignScore Server Image

If you want to use AlignScore-based fact-checking, you can also build a Docker image using the provided [Dockerfile](https://github.com/NVIDIA/NeMo-Guardrails/tree/develop/nemoguardrails/library/factchecking/align_score/Dockerfile).

```bash
cd nemoguardrails/library/factchecking/align_score
docker build -t alignscore-server .
```

NOTE: the provided Dockerfile downloads only the `base` AlignScore image. If you want support for the large model, uncomment the corresponding line in the Dockerfile.

### 4. \[Optional] Build the Jailbreak Detection Heuristics Server Image

If you want to use the jailbreak detection heuristics server, you can also build a Docker image using the provided [Dockerfile](https://github.com/NVIDIA/NeMo-Guardrails/tree/develop/nemoguardrails/library/jailbreak_detection/Dockerfile).

```bash
cd nemoguardrails/library/jailbreak_detection
docker build -t jailbreak_detection_heuristics .
```

## Running using Docker

To run the NeMo Guardrails server using the Docker image, run the following command:

```bash
docker run -p 8000:8000 -e OPENAI_API_KEY=$OPENAI_API_KEY nemoguardrails
```

This will start the NeMo Guardrails server with the example configurations. The Chat UI will be accessible at `http://localhost:8000`.

NOTE: Since the example configurations use OpenAI models (such as `gpt-3.5-turbo-instruct` and `gpt-4`), you need to provide an `OPENAI_API_KEY`.

To specify your own config folder for the server, you have to mount your local configuration into the `/config` path into the container:

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

## AlignScore Fact-checking

If one of your configurations uses the AlignScore fact-checking model, you can run the AlignScore server in a separate container:

```bash
docker run -p 5000:5000 alignscore-server
```

This will start the AlignScore server on port `5000`. You can then specify the AlignScore server URL in your configuration file:

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
