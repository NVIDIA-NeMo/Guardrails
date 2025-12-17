# GLiNER Server Deployment

This directory contains an example implementation of a GLiNER server that provides PII detection capabilities for NeMo Guardrails.

## Overview

[GLiNER](https://github.com/urchade/GLiNER) is a Generalist and Lightweight Model for Named Entity Recognition. This server wraps GLiNER in a FastAPI application that exposes an API compatible with NeMo Guardrails' GLiNER integration.

## Requirements

```bash
pip install gliner torch fastapi uvicorn pydantic aiohttp
```

## Quick Start

```bash
# Start the server with default settings (nvidia/gliner-PII model)
python gliner_server.py --host 0.0.0.0 --port 1235

# Or with custom model
python gliner_server.py --model nvidia/gliner-PII --device auto --port 1235
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

The server exposes the following endpoints:

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

## Testing

The server includes unit tests for the helper functions that don't require the GLiNER model or server to be running:

```bash
# Run from this directory
cd examples/deployment/gliner_server
pytest test_gliner_server.py -v
```

The tests cover:
- `create_tagged_text` - Creating tagged text from entities
- `remove_subset_entities` - Removing overlapping/subset entities
- `deduplicate_entities_by_score` - Keeping highest-scored entities
- `adjust_entity_positions` - Adjusting entity positions for chunking
- `process_raw_entities` - Full processing pipeline

## Docker Deployment

You can containerize the server for production:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

RUN pip install gliner torch fastapi uvicorn pydantic

COPY gliner_server.py .

EXPOSE 1235

CMD ["python", "gliner_server.py", "--host", "0.0.0.0", "--port", "1235"]
```

Build and run:
```bash
docker build -t gliner-server .
docker run -p 1235:1235 gliner-server
```
