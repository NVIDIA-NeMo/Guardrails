# Trend Micro Vision One AI Application Security

Trend Micro Vision One [AI Application Security's](https://docs.trendmicro.com/en-us/documentation/article/trend-vision-one-ai-scanner-ai-guard) AI Guard feature uses a configurable policy to identify risks in AI Applications, such as:

- Prompt injection attacks
- Toxicity, violent, and other harmful content
- Sensitive Data


## Setup

1. Create a new [Vision One API Key](https://docs.trendmicro.com/en-us/documentation/article/trend-vision-one-platform-api-keys) with permissions to Call Detection API
2. See the [AI Guard Integration Guide](https://docs.trendmicro.com/en-us/documentation/article/trend-vision-one-platform-api-keys) for details around creating your policy

[Colang v1](../../../examples/configs/trend_micro/):

```yaml
# config.yml

rails:
  config:
    trend_micro:
      v1_url: "https://api.xdr.trendmicro.com/beta/aiSecurity/guard" # Replace this with your AI Guard URL
      api_key_env_var: "V1_API_KEY"
  input:
    flows:
      - trend ai guard input

  output:
    flows:
      - trend ai guard output
```
[Colang v2](../../../examples/configs/trend_micro_v2/):
```yaml
# config.yml
colang_version: "2.x"
rails:
  config:
    trend_micro:
      v1_url: "https://api.xdr.trendmicro.com/beta/aiSecurity/guard" # Replace this with your AI Guard URL
      api_key_env_var: "V1_API_KEY"
```
```
# rails.co

import guardrails
import nemoguardrails.library.trend_micro

flow input rails $input_text
    trend ai guard $input_text

flow output rails $output_text
    trend ai guard $output_text
```
