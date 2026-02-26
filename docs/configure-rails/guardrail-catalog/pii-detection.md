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

The NeMo Guardrails library supports various PII detection models.

To activate the PII detection, you need to set up the server endpoint of the PII detection model in your `config.yml` and specify the entities that you want to detect and mask. For example, the following configuration uses the [GLiNER](https://github.com/NVIDIA/GLiNER) PII detection model, where the GLiNER server endpoint is `http://localhost:1235/v1/extract`:

### PII detection config

The detection flow blocks the input, output, and retrieval text if it detects PII.

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

### PII masking config

The masking flow replaces detected PII with labels.
For example, `Hi John, my email is john@example.com` becomes `Hi [FIRST_NAME], my email is [EMAIL]`.

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

For a detailed example, please refer to the [GLiNER Integration](community/gliner.md) page.

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
