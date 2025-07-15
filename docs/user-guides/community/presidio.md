# Presidio Integration

NeMo Guardrails supports detecting sensitive data out-of-the-box using [Presidio](https://github.com/Microsoft/presidio), which provides fast identification and anonymization modules for private entities in text such as credit card numbers, names, locations, social security numbers, bitcoin wallets, US phone numbers, financial data and more. You can detect sensitive data on user input, bot output, or the relevant chunks retrieved from the knowledge base.

## Setup

To use the built-in sensitive data detection rails, you must install Presidio and download the `en_core_web_lg` model for `spacy`.

```bash
pip install presidio-analyzer presidio-anonymizer spacy
python -m spacy download en_core_web_lg
```

As an alternative, you can also use the `sdd` extra.

```bash
pip install nemoguardrails[sdd]
python -m spacy download en_core_web_lg
```

## Usage

You can activate sensitive data detection in three ways: input rail, output rail, and retrieval rail.

### Input Rail

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

For the complete list of supported entities, please refer to [Presidio - Supported Entities](https://microsoft.github.io/presidio/supported_entities/) page.

Also, you have to add the `detect sensitive data on input` or `mask sensitive data on input` flows to the list of input rails:

```yaml
rails:
  input:
    flows:
      - ...
      - mask sensitive data on input     # or 'detect sensitive data on input'
      - ...
```

When using `detect sensitive data on input`, if sensitive data is detected, the bot will refuse to respond to the user's input. When using `mask sensitive data on input` the bot will mask the sensitive parts in the user's input and continue the processing.

### Output Rail

The configuration for the output rail is very similar to the input rail:

```yaml
rails:
  config:
    sensitive_data_detection:
      output:
        entities:
          - PERSON
          - EMAIL_ADDRESS
          - ...

  output:
    flows:
      - ...
      - mask sensitive data on output     # or 'detect sensitive data on output'
      - ...
```

### Retrieval Rail

The configuration for the retrieval rail is very similar to the input/output rail:

```yaml
rails:
  config:
    sensitive_data_detection:
      retrieval:
        entities:
          - PERSON
          - EMAIL_ADDRESS
          - ...

  retrieval:
    flows:
      - ...
      - mask sensitive data on retrieval     # or 'detect sensitive data on retrieval'
      - ...
```

## Score Threshold

To avoid false positives when detecting sensitive data entities, you can adjust the `score_threshold` parameter. This threshold defines the minimum confidence value required for a detected entity to be returned. By setting a higher `score_threshold`, you can reduce false positives by requiring a higher confidence level for detected entities. The default value for this parameter is 0.2.

The `score_threshold` parameter can be configured for any of the above sources (input, output and retrieval) to filter out sensitive data entities with a score above the defined threshold. Below is an example configuration that adjusts the `score_threshold` for a specific case:

```yaml

rails:
  config:
    sensitive_data_detection:
      input:
        score_threshold: 0.6
        entities:
          - PERSON
          - EMAIL_ADDRESS
          - ...
      output:
        score_threshold: 0.6
        entities:
          - PERSON
          - EMAIL_ADDRESS
          - ...

```

For additional guidance on handling undetected PII entities and minimizing false negatives, refer to the [Presidio FAQ](https://microsoft.github.io/presidio/faq/#what-can-i-do-if-presidio-does-not-detect-some-of-the-pii-entities-in-my-data-false-negatives).

## Custom Recognizers

If you have custom entities that you want to detect, you can define custom *recognizers*.
For more details, check out this [tutorial](https://microsoft.github.io/presidio/tutorial/08_no_code/) and this [example](https://github.com/microsoft/presidio/blob/main/presidio-analyzer/presidio_analyzer/conf/example_recognizers.yaml).

Below is an example of configuring a `TITLE` entity and detecting it inside the input rail.

```yaml
rails:
  config:
    sensitive_data_detection:
      recognizers:
        - name: "Titles recognizer"
          supported_language: "en"
          supported_entity: "TITLE"
          deny_list:
            - Mr.
            - Mrs.
            - Ms.
            - Miss
            - Dr.
            - Prof.
      input:
        entities:
          - PERSON
          - TITLE
```

## Custom Detection

If you want to implement a completely different sensitive data detection mechanism, you can override the default actions [`detect_sensitive_data`](https://github.com/NVIDIA/NeMo-Guardrails/tree/develop/nemoguardrails/library/sensitive_data_detection/actions.py) and [`mask_sensitive_data`](https://github.com/NVIDIA/NeMo-Guardrails/tree/develop/nemoguardrails/library/sensitive_data_detection/actions.py).
