# GuardrailsAI Integration

NeMo Guardrails provides out-of-the-box support for [GuardrailsAI](https://github.com/guardrails-ai/guardrails) validators, enabling comprehensive input and output validation using a rich ecosystem of community-built validators. GuardrailsAI offers validators for content safety, PII detection, toxic language filtering, jailbreak detection, topic restriction, and much more.

The integration provides access to both built-in validators and the entire [Guardrails Hub](https://hub.guardrailsai.com/) ecosystem, allowing you to dynamically load and configure validators for your specific use cases.

## Setup

To use GuardrailsAI validators, you need to install the `guardrails-ai` package:

```bash
pip install guardrails-ai
```

You may also need to install specific validators from the Guardrails Hub:

```bash
guardrails hub install guardrails/toxic_language
guardrails hub install guardrails/detect_jailbreak
guardrails hub install guardrails/guardrails_pii
```

## Usage

The GuardrailsAI integration uses a flexible configuration system that allows you to define validators with their parameters and metadata, then reference them in your input and output rails.

### Configuration Structure

Add GuardrailsAI validators to your `config.yml`:

```yaml
rails:
  config:
    guardrails_ai:
      validators:
        - name: toxic_language
          parameters:
            threshold: 0.5
            validation_method: "sentence"
          metadata: {}
        - name: guardrails_pii
          parameters:
            entities: ["phone_number", "email", "ssn"]
          metadata: {}
        - name: competitor_check
          parameters:
            competitors: ["Apple", "Google", "Microsoft"]
          metadata: {}
```

### Input Rails

To use GuardrailsAI validators for input validation:

```yaml
rails:
  input:
    flows:
      - guardrailsai check input $validator="guardrails_pii"
      - guardrailsai check input $validator="competitor_check"
```

### Output Rails

To use GuardrailsAI validators for output validation:

```yaml
rails:
  output:
    flows:
      - guardrailsai check output $validator="toxic_language"
      - guardrailsai check output $validator="restricttotopic"
```

## Built-in Validators

The integration includes support for the following validators that are pre-registered in the NeMo Guardrails validator registry. For detailed parameter specifications and usage examples, refer to the official [GuardrailsAI Hub](https://hub.guardrailsai.com/) documentation for each validator:

- `competitor_check` - `hub://guardrails/competitor_check`
- `detect_jailbreak` - `hub://guardrails/detect_jailbreak`
- `guardrails_pii` - `hub://guardrails/guardrails_pii`
- `one_line` - `hub://guardrails/one_line`
- `provenance_llm` - `hub://guardrails/provenance_llm`
- `regex_match` - `hub://guardrails/regex_match`
- `restricttotopic` - `hub://tryolabs/restricttotopic`
- `toxic_language` - `hub://guardrails/toxic_language`
- `valid_json` - `hub://guardrails/valid_json`
- `valid_length` - `hub://guardrails/valid_length`

## Complete Example

Here's a comprehensive example configuration:

```yaml
models:
  - type: main
    engine: openai
    model: gpt-4

rails:
  config:
    guardrails_ai:
      validators:
        - name: toxic_language
          parameters:
            threshold: 0.5
            validation_method: "sentence"
          metadata: {}
        - name: guardrails_pii
          parameters:
            entities: ["phone_number", "email", "ssn", "credit_card"]
          metadata: {}
        - name: competitor_check
          parameters:
            competitors: ["Apple", "Google", "Microsoft", "Amazon"]
          metadata: {}
        - name: restricttotopic
          parameters:
            valid_topics: ["technology", "science", "education"]
          metadata: {}
        - name: valid_length
          parameters:
            min: 10
            max: 500
          metadata: {}

  input:
    flows:
      - guardrailsai check input $validator="guardrails_pii"
      - guardrailsai check input $validator="competitor_check"

  output:
    flows:
      - guardrailsai check output $validator="toxic_language"
      - guardrailsai check output $validator="restricttotopic"
      - guardrailsai check output $validator="valid_length"
```

## Custom Validators from Guardrails Hub

You can use any validator from the [Guardrails Hub](https://hub.guardrailsai.com/) by specifying its hub path:

```yaml
rails:
  config:
    guardrails_ai:
      validators:
        - name: custom_validator_name
          parameters:
            # Custom parameters specific to the validator
          metadata: {}
```

The integration will automatically fetch validator information from the hub if it's not in the built-in registry.

## Performance Considerations

- Validators are cached to improve performance on repeated use
- Guard instances are reused when the same validator is called with identical parameters
- Consider the latency impact when chaining multiple validators

For a complete working example, see the [GuardrailsAI example configuration](https://github.com/NVIDIA/NeMo-Guardrails/tree/develop/examples/configs/guardrails_ai/).
