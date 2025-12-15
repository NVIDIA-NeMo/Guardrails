# GLiNER Integration

[GLiNER](https://github.com/urchade/GLiNER) is a Generalist and Lightweight Model for Named Entity Recognition that can detect a wide range of entity types, including comprehensive PII (Personally Identifiable Information) categories. This integration enables NeMo Guardrails to use a GLiNER server for PII detection and masking in input, output, and retrieval flows.

## Setup

1. Start the GLiNER server. The server code is available at `nemoguardrails/library/gliner/gliner_server.py`:

```bash
# Install dependencies
pip install gliner torch fastapi uvicorn

# Start the server (uses nvidia/gliner-PII model by default)
python nemoguardrails/library/gliner/gliner_server.py --host 0.0.0.0 --port 1235
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

## Supported Entity Types

The GLiNER server (using the `nvidia/gliner-PII` model) supports a comprehensive list of PII categories:

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

## Notes

- Ensure the GLiNER server is running and accessible from your NeMo Guardrails environment.
- The server uses GPU acceleration when available (CUDA or MPS on Apple Silicon).
- For production deployments, consider containerizing the GLiNER server.

For more information on GLiNER, see the [GLiNER GitHub repository](https://github.com/urchade/GLiNER).
