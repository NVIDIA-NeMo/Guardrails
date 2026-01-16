# Pangea AI Guard integration

The Pangea guardrail uses configurable detection policies (called *recipes*) from the [AI Guard service](https://pangea.cloud/docs/ai-guard/) to identify and mitigate risks in AI application traffic, including:

- Prompt injection attacks (with over 99% efficacy)
- 50+ types of PII and sensitive content, with support for custom patterns
- Toxicity, violence, self-harm, and other unwanted content
- Malicious links, IPs, and domains
- 100 spoken languages, with allowlist and denylist controls

All detections are logged in an audit trail for analysis, attribution, and incident response.
You can also configure webhooks to trigger alerts for specific detection types.

The following environment variable is required to use the Pangea AI Guard integration:

- `PANGEA_API_TOKEN`: Pangea API token with access to the AI Guard service.

You can also optionally set:

- `PANGEA_BASE_URL_TEMPLATE`: Template for constructing the base URL for API requests. The `{SERVICE_NAME}` placeholder will be replaced with the service name slug.
  Defaults to `https://ai-guard.aws.us.pangea.cloud` for Pangea's hosted (SaaS) deployment.

## Setup

Colang v1:

```yaml
# config.yml

rails:
  config:
    pangea:
      input:
        recipe: pangea_prompt_guard
      output:
        recipe: pangea_llm_response_guard

  input:
    flows:
      - pangea ai guard input

  output:
    flows:
      - pangea ai guard output
```

Colang v2:

```yaml
# config.yml

colang_version: "2.x"

rails:
  config:
    pangea:
      input:
        recipe: pangea_prompt_guard
      output:
        recipe: pangea_llm_response_guard
```

```
# rails.co

import guardrails
import nemoguardrails.library.pangea

flow input rails $input_text
    pangea ai guard input

flow output rails $output_text
    pangea ai guard output
```

## Next steps

- Explore example configurations for integrating Pangea AI Guard with your preferred Colang version:
  - [Pangea AI Guard for NeMo Guardrails v1](https://github.com/NVIDIA-NeMo/Guardrails/tree/develop/examples/configs/pangea)
  - [Pangea AI Guard for NeMo Guardrails v2](https://github.com/NVIDIA-NeMo/Guardrails/tree/develop/examples/configs/pangea_v2)
  - [Pangea AI Guard without LLM (guardrails only)](https://github.com/NVIDIA-NeMo/Guardrails/tree/develop/examples/configs/pangea_v2_no_llm) – Use this setup to evaluate AI Guard’s detection and response capabilities independently.
- Adjust your detection policies to fit your application’s risk profile. See the [AI Guard Recipes](https://pangea.cloud/docs/ai-guard/recipes) documentation for configuration details.
- Enable [AI Guard webhooks](https://pangea.cloud/docs/ai-guard/recipes#add-webhooks-to-detectors) to receive real-time alerts for detections in your NeMo Guardrails-powered application.
- Monitor and analyze detection activity in the [AI Guard Activity Log](https://pangea.cloud/docs/ai-guard/activity-log) for auditing and attribution.
- Learn more about [AI Guard Deployment Options](https://pangea.cloud/docs/deployment-models/) to understand how and where AI Guard can run to protect your AI applications.
