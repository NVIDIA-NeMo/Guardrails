# XGB Detectors Integration

XGB Detectors utilizes [XGBoost machine learning models](https://xgboost.readthedocs.io/en/stable/tutorials/model.html) to detect harmful content in data. Currently, only
the spam text detector, trained by the [Red Hat TrustyAI team](https://github.com/trustyai-explainability), is available for guardrailing use.

## Setup

Update your `config.yaml` file to include XGB detectors:

**Spam detection config**
```
rails:
  config:
    xgb:
      input:
        detectors:
          - SPAM
      output:
        detectors:
          - SPAM
  input:
    flows:
      - xgb detect on input
  output:
    flows:
      - xgb detect on output
```
The detection flow will not let the input and output text pass if spam is detected.

## Usage

Once configured, the XGB Guardrails integration will automatically:

1. Detect spam in inputs to the LLM
3. Detect spam in outputs from the LLM

## Error Handling

If the inference request to the XGB spam model fails, the system will assume spam is present as a precautionary measure.

## Notes

For more information on TrustyAI and its projects, please visit the TrustyAI [documentation](https://trustyai.org/docs/main/main).
