# Private AI Integration

[Private AI](https://docs.private-ai.com/?utm_medium=github&utm_campaign=nemo-guardrails) allows you to detect and mask Personally Identifiable Information (PII) in your data. This integration enables NeMo Guardrails to use Private AI for PII detection and masking in input, output, and retrieval flows.

## Setup

1. Ensure that you have access to Private AI API server running locally or in the cloud. To get started with the cloud version, you can use the [Private AI Portal](https://portal.private-ai.com/?utm_medium=github&utm_campaign=nemo-guardrails). For containerized deployments, check out this [Quickstart Guide](https://docs.private-ai.com/quickstart/?utm_medium=github&utm_campaign=nemo-guardrails).

2. Update your `config.yml` file to include the Private AI settings:

**PII detection config**

```yaml
rails:
  config:
    privateai:
      server_endpoint: http://your-privateai-api-endpoint/process/text  # Replace this with your Private AI process text endpoint
      input:
        entities:  # If no entity is specified here, all supported entities will be detected by default.
          - NAME_FAMILY
          - LOCATION_ADDRESS_STREET
          - EMAIL_ADDRESS
      output:
        entities:
          - NAME_FAMILY
          - LOCATION_ADDRESS_STREET
          - EMAIL_ADDRESS
  input:
    flows:
      - detect pii on input
  output:
    flows:
      - detect pii on output
```

The detection flow will not let the input/output/retrieval text pass if PII is detected.

**PII masking config**

```yaml
rails:
  config:
    privateai:
      server_endpoint: http://your-privateai-api-endpoint/process/text  # Replace this with your Private AI process text endpoint
      input:
        entities:  # If no entity is specified here, all supported entities will be detected by default.
          - NAME_FAMILY
          - LOCATION_ADDRESS_STREET
          - EMAIL_ADDRESS
      output:
        entities:
          - NAME_FAMILY
          - LOCATION_ADDRESS_STREET
          - EMAIL_ADDRESS
  input:
    flows:
      - mask pii on input
  output:
    flows:
      - mask pii on output
```

The masking flow will mask the PII in the input/output/retrieval text before they are sent to the LLM/user. For example, `Hi John Doe, my email is john.doe@example.com` will be converted to `Hi [NAME], my email is [EMAIL_ADDRESS]`.

Replace `http://your-privateai-api-endpoint/process/text` with your actual Private AI process text endpoint and set the `PAI_API_KEY` environment variable if you're using the Private AI cloud API.

3. You can customize the `entities` list under both `input` and `output` to include the PII types you want to detect. A full list of supported entities can be found [here](https://docs.private-ai.com/entities/?utm_medium=github&utm_campaign=nemo-guardrails).

## Usage

Once configured, the Private AI integration can automatically:

1. Detect or mask PII in user inputs before they are processed by the LLM.
2. Detect or mask PII in LLM outputs before they are sent back to the user.
3. Detect or mask PII in retrieved chunks before they are sent to the LLM.

The `detect_pii` and `mask_pii` actions in `nemoguardrails/library/privateai/actions.py` handle the PII detection and masking processes, respectively.

## Customization

You can customize the PII detection behavior by modifying the `entities` lists in the `config.yml` file. Refer to the Private AI documentation for a complete list of [supported entity types](https://docs.private-ai.com/entities/?utm_medium=github&utm_campaign=nemo-guardrails).

## Error Handling

If the Private AI detection API request fails, the system will assume PII is present as a precautionary measure.

## Notes

- Ensure that your Private AI process text endpoint is properly set up and accessible from your NeMo Guardrails environment.
- The integration currently supports PII detection and masking.

For more information on Private AI and its capabilities, please refer to the [Private AI documentation](https://docs.private-ai.com/?utm_medium=github&utm_campaign=nemo-guardrails).
