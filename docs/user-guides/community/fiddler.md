# Fiddler Guardrails Integration

Fiddler Guardrails utilizes [Fiddler Trust Models](https://docs.fiddler.ai/product-guide/llm-monitoring/llm-based-metrics#fiddler-fast-trust-metrics) in a specialized low-latency, high-throughput configuration. Guardrails can be used to guard Large Language Model (LLM) applications against user threats, such as prompt injection or harmful and inappropriate content, and LLM hallucinations.

Currently, only Fiddler Trust Models ([Faithfulness](https://docs.fiddler.ai/product-guide/llm-monitoring/enrichments-private-preview#fast-faithfulness-private-preview) and [Safety](https://docs.fiddler.ai/product-guide/llm-monitoring/enrichments-private-preview#fast-safety-private-preview)) - Fiddler's in-house, purpose-built SLMs - are available for guardrail use. Future model releases and model updates/improvements will also be available for guardrail use.

## Setup

1. Ensure that you have access to a valid Fiddler environment. To obtain one, please [contact us](https://www.fiddler.ai/contact-sales).

2. Create a new [Fiddler environment key](https://docs.fiddler.ai/ui-guide/administration-ui/settings#credentials) and set the `FIDDLER_API_KEY` environment variable to this key to authenticate into the Fiddler service.

Update your `config.yml` file to include the following settings:

```yaml
rails:
    config:
        fiddler:
            fiddler_endpoint: https://testfiddler.ai # Replace this with your fiddler environment
            safety_threshold: .2 # Any value greater than this threshold will trigger a violation
            faithfulness_threshold: .3 # Any value less than this threshold will trigger a violation
    input:
        flows:
            - fiddler user safety
    output:
        flows:
            - fiddler bot safety
            - fiddler bot faithfulness
```

## Usage

Once configured, the Fiddler Guardrails integration will automatically:

1. Detect unsafe, offensive, harmful, or jailbreaking inputs into the LLM
2. Detect unsafe, offensive, or harmful outputs from the LLM
3. Detect potential hallucinated outputs from the LLM

## Customization

You can configure the thresholds for both the safety and hallucination detection using the `safety_threshold` and the `faithfulness_threshold` parameters

## Error Handling

Fiddler Guardrails will not block inputs in the event of any API failure.

## Notes

For more information about Fiddler Guardrails, please visit the Fiddler Guardrails [documentation](https://docs.fiddler.ai/product-guide/llm-monitoring/guardrails).
