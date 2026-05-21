---
title:
  page: "PII Detection"
  nav: "PII Detection"
description: "Reference for PII detection guardrails that protect user privacy by detecting and masking sensitive data."
topics: ["Configuration", "AI Safety"]
tags: ["Rails", "PII", "Privacy", "YAML"]
content:
  type: "Reference"
  difficulty: "Intermediate"
  audience: ["Developer", "AI Engineer"]
---

# PII Detection

Personally Identifiable Information (PII) detection helps protect user privacy by detecting and masking sensitive data in user inputs, LLM outputs, and retrieved content.

## GLiNER-based PII Detection

The NeMo Guardrails library supports PII detection and masking using the [NVIDIA GLiNER-PII NIM](https://catalog.ngc.nvidia.com/orgs/nim/teams/nvidia/containers/gliner-pii). For a full step-by-step walkthrough that includes CLI usage, Python SDK usage, and local deployment, refer to the [GLiNER Integration](community/gliner.md) page. The examples below assume each configuration lives in its own subdirectory under `config/` (NeMo Guardrails merges every `.yml` / `.yaml` file it finds in a `--config` directory, so detection and masking rule sets need separate folders).

### NVIDIA-Hosted Endpoint

Use the NVIDIA-hosted NIM by setting `api_key_env_var` in both the `models` block and the `gliner` config block.

`nvidia/gliner-pii` does not appear in the configs below because it is the default value of `rails.config.gliner.model`. You only need to set that field explicitly if you want to use a different model:

```yaml
rails:
  config:
    gliner:
      model: nvidia/gliner-pii  # default — omit or change as needed
```

**PII detection** (save as `config/pii_detection/config.yml`) blocks input or output that contains PII:

```yaml
models:
  - type: main
    engine: nim
    model: meta/llama-3.1-8b-instruct
    api_key_env_var: NVIDIA_API_KEY

rails:
  config:
    gliner:
      server_endpoint: https://integrate.api.nvidia.com/v1/chat/completions
      api_key_env_var: NVIDIA_API_KEY
      threshold: 0.5  # Confidence threshold (0.0 to 1.0)
      input:
        entities:  # If no entity is specified, all default PII categories are detected
          - email
          - phone_number
          - ssn
          - first_name
          - last_name
      output:
        entities:
          - email
          - phone_number
          - credit_debit_card
  input:
    flows:
      - gliner detect pii on input
  output:
    flows:
      - gliner detect pii on output
```

**PII masking** (save as `config/pii_masking/config.yml`) replaces detected PII with label placeholders, such as changing `Hi John` to `Hi [FIRST_NAME]`:

```yaml
models:
  - type: main
    engine: nim
    model: meta/llama-3.1-8b-instruct
    api_key_env_var: NVIDIA_API_KEY

rails:
  config:
    gliner:
      server_endpoint: https://integrate.api.nvidia.com/v1/chat/completions
      api_key_env_var: NVIDIA_API_KEY
      input:
        entities:
          - email
          - first_name
          - last_name
      output:
        entities:
          - email
          - first_name
          - last_name
  input:
    flows:
      - gliner mask pii on input
  output:
    flows:
      - gliner mask pii on output
```

### Locally Hosted NIMs

To run both NIMs locally, pull the Docker containers and point each endpoint to localhost. No `api_key_env_var` is needed for local inference.

> **Note:** You still need an `NGC_API_KEY` (starting with `nvapi-`) to pull the Docker images and download model artifacts. You can generate one at [org.ngc.nvidia.com/setup/api-keys](https://org.ngc.nvidia.com/setup/api-keys) or [build.nvidia.com](https://build.nvidia.com). Legacy NGC keys (older format) will cause the container to fail during artifact download. See the [GLiNER Integration — Deploy NIMs Locally](community/gliner.md#start-the-containers) section for full `docker run` instructions.

**PII detection** (update `config/pii_detection/config.yml`):

```yaml
models:
  - type: main
    engine: nim
    model: meta/llama-3.1-8b-instruct
    parameters:
      base_url: http://localhost:8001/v1

rails:
  config:
    gliner:
      server_endpoint: http://localhost:8000/v1/chat/completions
      threshold: 0.5
      input:
        entities:
          - email
          - phone_number
          - ssn
          - first_name
          - last_name
      output:
        entities:
          - email
          - phone_number
          - credit_debit_card
  input:
    flows:
      - gliner detect pii on input
  output:
    flows:
      - gliner detect pii on output
```

**PII masking** (update `config/pii_masking/config.yml`):

```yaml
models:
  - type: main
    engine: nim
    model: meta/llama-3.1-8b-instruct
    parameters:
      base_url: http://localhost:8001/v1

rails:
  config:
    gliner:
      server_endpoint: http://localhost:8000/v1/chat/completions
      input:
        entities:
          - email
          - first_name
          - last_name
      output:
        entities:
          - email
          - first_name
          - last_name
  input:
    flows:
      - gliner mask pii on input
  output:
    flows:
      - gliner mask pii on output
```

See the [GLiNER Integration](community/gliner.md) page for Docker pull and run instructions.

## Presidio-based Sensitive Data Detection

The NeMo Guardrails library supports detecting sensitive data out-of-the-box using [Presidio](https://github.com/Microsoft/presidio), which provides fast identification and anonymization modules for private entities in text such as credit card numbers, names, locations, social security numbers, bitcoin wallets, US phone numbers, financial data and more. You can detect sensitive data on user input, bot output, or the relevant chunks retrieved from the knowledge base.

To activate a sensitive data detection input rail, you have to configure the entities that you want to detect:

```yaml
rails:
  config:
    sensitive_data_detection:
      input:
        entities:
          - PERSON
          - EMAIL_ADDRESS
          - ...
```

### Example usage

```yaml
rails:
  input:
    flows:
      - mask sensitive data on input
  output:
    flows:
      - mask sensitive data on output
  retrieval:
    flows:
      - mask sensitive data on retrieval
```

For more details, check out the [Presidio Integration](community/presidio.md) page.

## Private AI PII Detection

The NeMo Guardrails library supports using [Private AI API](https://docs.private-ai.com/?utm_medium=github&utm_campaign=nemo-guardrails) for PII detection and masking input, output and retrieval flows.

To activate the PII detection or masking, you need specify `server_endpoint`, and the entities that you want to detect or mask. You'll also need to set the `PAI_API_KEY` environment variable if you're using the Private AI cloud API.

```yaml
rails:
  config:
    privateai:
      server_endpoint: http://your-privateai-api-endpoint/process/text  # Replace this with your Private AI process text endpoint
      input:
        entities:  # If no entity is specified here, all supported entities will be detected by default.
          - NAME_FAMILY
          - EMAIL_ADDRESS
          ...
      output:
        entities:
          - NAME_FAMILY
          - EMAIL_ADDRESS
          ...
```

### Example usage

**PII detection**

```yaml
rails:
  input:
    flows:
      - detect pii on input
  output:
    flows:
      - detect pii on output
  retrieval:
    flows:
      - detect pii on retrieval
```

For more details, check out the [Private AI Integration](community/privateai.md) page.
