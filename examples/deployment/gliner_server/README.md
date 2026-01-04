# GLiNER Server

A FastAPI server for PII detection and entity extraction using GLiNER.

## Overview

[GLiNER](https://github.com/urchade/GLiNER) is a Generalist and Lightweight Model for Named Entity Recognition. This package wraps GLiNER in a FastAPI application that exposes an API compatible with NeMo Guardrails' GLiNER integration.

## Installation

```bash
# Install with uv
uv pip install -e .

# Or install with dev dependencies
uv pip install -e ".[dev]"
```

## Quick Start

```bash
# Start the server with default settings (nvidia/gliner-PII model)
gliner-server --host 0.0.0.0 --port 1235

# Or run directly with Python
python -m gliner_server.server --host 0.0.0.0 --port 1235

# Or with custom model
gliner-server --model nvidia/gliner-PII --device auto --port 1235
```

## Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--host` | `0.0.0.0` | Host to bind to |
| `--port` | `1235` | Port to bind to |
| `--model` | `nvidia/gliner-PII` | GLiNER model to load |
| `--device` | `auto` | Device to use (`auto`, `cpu`, `cuda`, `mps`) |
| `--reload` | `false` | Enable auto-reload for development |

## Environment Variables

You can also configure the server using environment variables:

- `HOST` - Host to bind to
- `PORT` - Port to bind to
- `MODEL_NAME` - GLiNER model to load
- `DEVICE` - Device to use

## API Endpoints

### `POST /v1/extract`

Main endpoint for entity extraction. This is the endpoint used by NeMo Guardrails.

**Request:**
```json
{
  "text": "Hello, my name is John and my email is john@example.com",
  "labels": ["email", "first_name"],
  "threshold": 0.5,
  "chunk_length": 384,
  "overlap": 128,
  "flat_ner": false
}
```

**Response:**
```json
{
  "entities": [
    {
      "value": "John",
      "suggested_label": "first_name",
      "start_position": 18,
      "end_position": 22,
      "score": 0.95
    },
    {
      "value": "john@example.com",
      "suggested_label": "email",
      "start_position": 40,
      "end_position": 56,
      "score": 0.98
    }
  ],
  "total_entities": 2,
  "tagged_text": "Hello, my name is [John](first_name) and my email is [john@example.com](email)"
}
```

### `GET /v1/labels`

Get the default PII labels supported by the model.

### `GET /v1/models`

OpenAI-compatible models endpoint.

### `POST /v1/chat/completions`

OpenAI-compatible chat completions endpoint (returns entities as JSON in the response).

### `GET /health`

Health check endpoint with model status.

## Supported Entity Types

The default `nvidia/gliner-PII` model supports 56 PII categories:

| Category | Entity Types |
|----------|-------------|
| Personal Identifiers | `first_name`, `last_name`, `ssn`, `date_of_birth`, `age`, `gender` |
| Contact Information | `email`, `phone_number`, `fax_number`, `street_address`, `city`, `state`, `postcode`, `country`, `county` |
| Financial | `credit_debit_card`, `cvv`, `bank_routing_number`, `account_number`, `swift_bic`, `tax_id` |
| Technical | `ipv4`, `ipv6`, `mac_address`, `url`, `api_key`, `password`, `pin`, `http_cookie` |
| Identification | `national_id`, `license_plate`, `vehicle_identifier`, `employee_id`, `customer_id`, `unique_id`, `medical_record_number`, `health_plan_beneficiary_number` |
| Sensitive Attributes | `sexuality`, `political_view`, `race_ethnicity`, `religious_belief`, `blood_type` |

## Project Structure

```
gliner_server/
├── pyproject.toml          # Package configuration with uv
├── README.md
├── src/
│   └── gliner_server/
│       ├── __init__.py     # Package exports
│       ├── models.py       # Pydantic request/response models
│       ├── pii_utils.py    # PII detection utilities
│       └── server.py       # FastAPI server
└── tests/
    ├── __init__.py
    └── test_pii_utils.py   # Unit tests
```

## Testing

```bash
# Install dev dependencies
uv pip install -e ".[dev]"

# Run tests
pytest

# Run tests with verbose output
pytest -v
```

## Integration with NeMo Guardrails

Configure NeMo Guardrails to use this server:

```yaml
rails:
  config:
    gliner:
      server_endpoint: http://localhost:1235/v1/extract
      threshold: 0.5
      input:
        entities:
          - email
          - phone_number
          - first_name
  input:
    flows:
      - gliner detect pii on input
```

See the [GLiNER User Guide](../../../docs/user-guides/community/gliner.md) for more details.

## Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install uv
RUN pip install uv

# Copy package files
COPY pyproject.toml README.md ./
COPY src/ src/

# Install dependencies
RUN uv pip install --system .

EXPOSE 1235

CMD ["gliner-server", "--host", "0.0.0.0", "--port", "1235"]
```

Build and run:
```bash
docker build -t gliner-server .
docker run -p 1235:1235 gliner-server
```
