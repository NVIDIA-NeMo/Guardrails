# GLiNER Integration

[GLiNER](https://github.com/urchade/GLiNER) is a Generalist and Lightweight Model for Named Entity Recognition that can detect a wide range of entity types, including comprehensive PII (Personally Identifiable Information) categories. This integration enables NeMo Guardrails to use a GLiNER-compatible server for PII detection and masking in input, output, and retrieval flows.

## Setup

1. Deploy a GLiNER-compatible server. An example implementation is provided at [`examples/deployment/gliner_server/`](../../../examples/deployment/gliner_server/):

```bash
cd examples/deployment/gliner_server

# Install with uv (recommended)
uv sync

# Start the server (uses nvidia/gliner-PII model by default)
uv run gliner-server --host 0.0.0.0 --port 1235

# Or install with pip
pip install -e .
gliner-server --host 0.0.0.0 --port 1235
```

2. Update your `config.yml` file to include the GLiNER settings:

**PII detection config**

```yaml
rails:
  config:
    gliner:
      server_endpoint: http://localhost:1235/v1/extract
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

The detection flow will block the input/output/retrieval text if PII is detected.

**PII masking config**

```yaml
rails:
  config:
    gliner:
      server_endpoint: http://localhost:1235/v1/extract
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

The masking flow will replace detected PII with labels. For example, `Hi John, my email is john@example.com` will be converted to `Hi [FIRST_NAME], my email is [EMAIL]`.

## API Specification

The GLiNER integration expects a server that implements the following API:

### `POST /v1/extract`

Extract entities from text.

**Request Body:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `text` | string | Yes | - | The text to analyze for entities |
| `labels` | array[string] | No | Server default | List of entity labels to detect |
| `threshold` | float | No | 0.5 | Confidence threshold (0.0 to 1.0) |
| `chunk_length` | int | No | 384 | Length of text chunks for processing |
| `overlap` | int | No | 128 | Overlap between chunks |
| `flat_ner` | bool | No | false | Whether to use flat NER mode |

**Example Request:**
```json
{
  "text": "Hello, my name is John and my email is john@example.com",
  "labels": ["email", "first_name"],
  "threshold": 0.5
}
```

**Response Body:**

| Field | Type | Description |
|-------|------|-------------|
| `entities` | array[EntitySpan] | List of detected entities |
| `total_entities` | int | Total count of entities found |
| `tagged_text` | string | Text with entities tagged as `[value](label)` |

**EntitySpan Object:**

| Field | Type | Description |
|-------|------|-------------|
| `value` | string | The detected entity text |
| `suggested_label` | string | The entity label/type |
| `start_position` | int | Start character index (inclusive) |
| `end_position` | int | End character index (exclusive) |
| `score` | float | Confidence score |

**Example Response:**
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

## Supported Entity Types

The example GLiNER server (using the `nvidia/gliner-PII` model) supports a comprehensive list of PII categories:

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
| `server_endpoint` | `http://localhost:1235/v1/extract` | GLiNER server endpoint |
| `threshold` | `0.5` | Confidence threshold for entity detection (0.0 to 1.0) |
| `chunk_length` | `384` | Length of text chunks for processing |
| `overlap` | `128` | Overlap between chunks |
| `flat_ner` | `false` | Whether to use flat NER mode |

## Usage

Once configured, the GLiNER integration can automatically:

1. Detect or mask PII in user inputs before they are processed by the LLM.
2. Detect or mask PII in LLM outputs before they are sent back to the user.
3. Detect or mask PII in retrieved chunks before they are sent to the LLM.

## Example Deployment

An example GLiNER server implementation is provided at [`examples/deployment/gliner_server/`](../../../examples/deployment/gliner_server/). This implementation:

- Uses the `nvidia/gliner-PII` model for comprehensive PII detection
- Supports GPU acceleration (CUDA, MPS on Apple Silicon)
- Implements text chunking with overlap for long documents
- Provides entity deduplication
- Structured as a proper Python package with `src/` layout
- CLI entry point (`gliner-server`) for easy startup
- Unit tests for PII utility functions (no server required)
- Integration test script for end-to-end validation

See the [deployment README](../../../examples/deployment/gliner_server/README.md) for detailed instructions.

## Testing

The NeMo Guardrails GLiNER integration tests (`tests/test_gliner.py`) use mocked API responses, so they don't require a running server. To run them:

```bash
pytest tests/test_gliner.py -v
```

The example server package also includes unit tests for the PII utility functions:

```bash
cd examples/deployment/gliner_server
uv run pytest tests/ -v
```

For integration testing with a running server, use the provided script:

```bash
cd examples/deployment/gliner_server
./test_integration.sh
```

## Notes

- Ensure a GLiNER-compatible server is running and accessible from your NeMo Guardrails environment.
- You can use the provided example server or implement your own server following the API specification above.
- For production deployments, consider containerizing the server.

For more information on GLiNER, see the [GLiNER GitHub repository](https://github.com/urchade/GLiNER).
