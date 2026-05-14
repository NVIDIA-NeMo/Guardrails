# Polygraf

[Polygraf](https://polygraf.ai/) offers a state-of-the-art PII Detection & Masking API designed to help identify and protect sensitive information in your text.

This integration enables the NeMo Guardrails library to detect or mask PII on user input, bot output, and/or retrieved documents.

## Setup

First, you need an API key from Polygraf. Once you have it, set it as an environment variable:
```bash
export POLYGRAF_API_KEY="your-polygraf-api-key"
```

Also, you'll need the endpoint where the Polygraf PII detection API is hosted.

- For **Polygraf cloud**, use `https://governance.api.polygraf.ai/gcp/pii/text-detect`.
- For **self-hosted** deployments, set this to your service endpoint (the default local example is typically `http://localhost:8000/v1/pii/text-detect`).

## Configuration

To activate the Polygraf integration, configure the `polygraf` block in `config.yml` and include the relevant flows.

To use it for PII detection, your `config.yml` should look like this:

```yaml
rails:
  config:
    polygraf:
      server_endpoint: "https://governance.api.polygraf.ai/gcp/pii/text-detect"
      input:
        entities:
          - Email
          - Person
          - Phone
      output:
        entities:
          - Email
          - Person
          - Phone

  input:
    flows:
      - polygraf detect pii on input

  output:
    flows:
      - polygraf detect pii on output
```

To use it for PII masking, use the `polygraf mask pii ...` flows instead:

```yaml
rails:
  config:
    polygraf:
      server_endpoint: "https://governance.api.polygraf.ai/gcp/pii/text-detect"
      input:
        entities:
          - Email
          - Person
          - Phone
      output:
        entities:
          - Email
          - Person
          - Phone

  input:
    flows:
      - polygraf mask pii on input

  output:
    flows:
      - polygraf mask pii on output
```

## Entity Types

You can configure which entities to detect or mask by specifying them in the `entities` list. If you omit the `entities` list for detection, any detected PII will trigger the detection flow. For masking, if you omit the `entities` list, it will mask all detected PII.

Common entities you might want to detect/mask:
- `Person`
- `Email`
- `Phone`
- `CREDIT_CARD_NUMBER`
- `PASSPORT_NUMBER`
- `SSN`
- `LOCATION`
- `ORGANIZATION`

For a complete list of supported entity types, please refer to the Polygraf documentation.
