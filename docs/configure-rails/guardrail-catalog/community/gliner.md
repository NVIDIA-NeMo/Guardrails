# GLiNER Integration

[GLiNER](https://github.com/urchade/GLiNER) is a generalist and lightweight model for named entity recognition. [NVIDIA GLiNER-PII](https://huggingface.co/nvidia/gliner-PII) is an adaptation that detects a wide range of PII categories. This integration enables NeMo Guardrails to use GLiNER-PII for PII detection and masking in input, output, and retrieval flows.

## Prerequisites

To use the NVIDIA-hosted NIMs, set your NVIDIA API key:

```bash
export NVIDIA_API_KEY="nvapi-..."
```

You can obtain an API key at [build.nvidia.com](https://build.nvidia.com).

You will also need to [install](../../../getting-started/installation-guide.md) the NeMo Guardrails library.

## Configure Guardrails

Create a `config` directory and add a `config.yml` file. The examples below target the NVIDIA-hosted GLiNER-PII and Llama 3.1 8B NIM endpoints.

`nvidia/gliner-pii` does not appear in the configs below because it is the default value of `rails.config.gliner.model`. You only need to set that field explicitly if you want to use a different model.

### PII Detection

The detection flow blocks any input or output that contains PII. To implement this flow, save the config below as your `config.yml` file.

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
      threshold: 0.5
      input:
        entities:
          - first_name
          - last_name
          - email
          - phone_number
      output:
        entities:
          - first_name
          - last_name
          - email
          - phone_number
  input:
    flows:
      - gliner detect pii on input
  output:
    flows:
      - gliner detect pii on output
```

### PII Masking

The masking flow replaces detected PII with label placeholders before the LLM processes the text, rather than blocking the request outright. For example, `Hi, I am John. My email is john@example.com` becomes `Hi, I am [FIRST_NAME]. My email is [EMAIL]`. To implement this flow, save the config below as your `config.yml` file.

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
      threshold: 0.5
      input:
        entities:
          - first_name
          - last_name
          - email
          - phone_number
      output:
        entities:
          - first_name
          - last_name
          - email
          - phone_number
  input:
    flows:
      - gliner mask pii on input
  output:
    flows:
      - gliner mask pii on output
```

## Run the Guardrails Chat CLI

Start an interactive chat session using your config directory:

```bash
nemoguardrails chat --config ./config
```

With **PII detection** enabled, any message containing PII is blocked before reaching the LLM:

```text
> Hello! My name is John and my email is john@example.com.
I'm sorry, I can't respond to that.
```

With **PII masking** enabled, PII is replaced in-place before the LLM sees the message:

```text
> Hello! My name is John and my email is john@example.com.
Nice to meet you, [FIRST_NAME]! How can I help you today?
```

## Use the Python SDK

```python
import nest_asyncio
nest_asyncio.apply()

from nemoguardrails import LLMRails, RailsConfig

config = RailsConfig.from_path("./config")
rails = LLMRails(config)

response = rails.generate(
    messages=[{"role": "user", "content": "Hello! My name is John and my email is john@example.com."}]
)
print(response["content"])

# Inspect the guardrail execution trace
info = rails.explain()
print(info.colang_history)
```

## Deploy NIMs Locally

Running both NIMs locally eliminates network round-trips and removes the NVIDIA API key requirement for inference. You still need an NGC API key to pull the Docker images from [ngc.nvidia.com](https://ngc.nvidia.com).

### GPU Requirements

| NIM | Min GPUs | Min VRAM | Compatible GPUs |
|-----|----------|----------|-----------------|
| `nvidia/gliner-pii` | 1 | 4 GB | T4, L4, A10, A10G, A100, H100, L40S |
| `meta/llama-3.1-8b-instruct` | 1 | 16 GB | L4, A10G, A100 (40 GB or 80 GB), H100, L40S |

The Llama NIM auto-selects the optimal TensorRT-LLM profile (FP16 or INT8) based on available hardware. An A10G (24 GB) or L4 (24 GB) is the practical minimum for comfortable headroom; a T4 (16 GB) may work but is not officially supported.

> **Note:** `nvidia/gliner-pii` is pre-GA (`1.0.0-rc1`). The GPU requirements above are estimates based on the GLiNER encoder-only architecture because NVIDIA has not officially published requirements yet.

### Start the Containers

The **GLiNER-PII NIM** runs on port 8000:

```bash
# Authenticate with NGC (username: $oauthtoken, password: your NGC API key)
echo $NGC_API_KEY | docker login -u '$oauthtoken' --password-stdin nvcr.io

docker run --rm -it --gpus all \
  -e NGC_API_KEY \
  -p 8000:8000 \
  nvcr.io/nim/nvidia/gliner-pii:1.0.0-rc1
```

Map the **Llama 3.1 8B Instruct NIM** to port 8001 to avoid a conflict with GLiNER:

```bash
docker run --rm -it --gpus all \
  -e NGC_API_KEY \
  -p 8001:8000 \
  nvcr.io/nim/meta/llama-3.1-8b-instruct:latest
```

Wait until both containers log `Application startup complete` before proceeding.

### Update the Configuration

Replace the remote endpoints with local ones and remove both `api_key_env_var` fields:

**PII detection:**

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
          - first_name
          - last_name
          - email
          - phone_number
      output:
        entities:
          - first_name
          - last_name
          - email
          - phone_number
  input:
    flows:
      - gliner detect pii on input
  output:
    flows:
      - gliner detect pii on output
```

**PII masking:**

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
          - first_name
          - last_name
          - email
          - phone_number
      output:
        entities:
          - first_name
          - last_name
          - email
          - phone_number
  input:
    flows:
      - gliner mask pii on input
  output:
    flows:
      - gliner mask pii on output
```

### Repeat Steps 2 and 3

With the containers running and config updated, rerun the CLI and SDK commands from [Step 2](#2-run-the-guardrails-chat-cli) and [Step 3](#3-use-the-python-sdk). No other changes are required.

---

## API Specification

The GLiNER-PII NIM exposes an OpenAI-compatible chat completions endpoint.

### `POST /v1/chat/completions`

Extract entities from text.

**Request Body:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `model` | string | Yes | - | Must be `"nvidia/gliner-pii"` |
| `messages` | array | Yes | - | Array with a single user message containing the text to analyze |
| `labels` | array[string] | No | Server default | List of entity labels to detect |
| `threshold` | float | No | 0.5 | Confidence threshold (0.0 to 1.0) |
| `chunk_length` | int | No | 384 | Length of text chunks for processing |
| `overlap` | int | No | 128 | Overlap between chunks |
| `flat_ner` | bool | No | false | Whether to use flat NER mode |

**Example Request:**

```json
{
  "model": "nvidia/gliner-pii",
  "messages": [{"role": "user", "content": "Hello, my name is John and my email is john@example.com"}],
  "labels": ["email", "first_name"],
  "threshold": 0.5
}
```

**Response Body:**

The response follows the OpenAI chat completions format. The `choices[0].message.content` field contains a JSON string with the detected entities.

**Parsed Content Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `entities` | array[EntitySpan] | List of detected entities |
| `total_entities` | int | Total count of entities found |
| `tagged_text` | string | Text with entities tagged as `[value](label)` |

**EntitySpan Object:**

| Field | Type | Description |
|-------|------|-------------|
| `text` | string | The detected entity text |
| `label` | string | The entity label/type |
| `start` | int | Start character index (inclusive) |
| `end` | int | End character index (exclusive) |
| `score` | float | Confidence score |

**Example Parsed Content:**

```json
{
  "entities": [
    {
      "text": "John",
      "label": "first_name",
      "start": 18,
      "end": 22,
      "score": 0.95
    },
    {
      "text": "john@example.com",
      "label": "email",
      "start": 40,
      "end": 56,
      "score": 0.98
    }
  ],
  "total_entities": 2,
  "tagged_text": "Hello, my name is [John](first_name) and my email is [john@example.com](email)"
}
```

## Supported Entity Types

The NVIDIA GLiNER-PII NIM supports these PII categories:

| Category | Entity Types |
|----------|-------------|
| Personal Identifiers | `first_name`, `last_name`, `ssn`, `date_of_birth`, `age`, `gender` |
| Contact Information | `email`, `phone_number`, `fax_number`, `street_address`, `city`, `state`, `postcode`, `country`, `county` |
| Financial | `credit_debit_card`, `cvv`, `bank_routing_number`, `account_number`, `swift_bic`, `tax_id` |
| Technical | `ipv4`, `ipv6`, `mac_address`, `url`, `api_key`, `password`, `pin`, `http_cookie` |
| Identification | `national_id`, `license_plate`, `vehicle_identifier`, `employee_id`, `customer_id`, `unique_id`, `medical_record_number`, `health_plan_beneficiary_number` |
| Sensitive Attributes | `sexuality`, `political_view`, `race_ethnicity`, `religious_belief`, `blood_type` |

## Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `server_endpoint` | `http://localhost:8000/v1/chat/completions` | GLiNER-PII NIM endpoint |
| `api_key_env_var` | None | Environment variable containing the API key (required for hosted endpoint) |
| `threshold` | `0.5` | Confidence threshold for entity detection (0.0 to 1.0) |
| `chunk_length` | `384` | Length of text chunks for processing |
| `overlap` | `128` | Overlap between chunks |
| `flat_ner` | `false` | Whether to use flat NER mode |

## Testing

The GLiNER integration tests in `tests/test_gliner.py` use mocked API responses, so they don't require a running server.
To run them:

```bash
pytest tests/test_gliner.py -v
```

For a self-hosted alternative using the `nvidia/gliner-PII` model directly, the [`examples/deployment/gliner_server/`](https://github.com/NVIDIA-NeMo/Guardrails/tree/develop/examples/deployment/gliner_server/) directory provides a reference implementation. It exposes a `POST /v1/extract` endpoint — if you use it, set `server_endpoint` to `http://localhost:1235/v1/extract`. Refer to the [deployment README](https://github.com/NVIDIA-NeMo/Guardrails/tree/develop/examples/deployment/gliner_server/README.md) for setup instructions.

For more information on GLiNER, refer to the [GLiNER GitHub repository](https://github.com/urchade/GLiNER).
