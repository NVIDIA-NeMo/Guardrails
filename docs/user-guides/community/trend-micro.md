# Trend Micro Vision One AI Application Security

Trend Micro Vision One [AI Application Security's](https://docs.trendmicro.com/en-us/documentation/article/trend-vision-one-ai-scanner-ai-guard) AI Guard feature uses a configurable policy to identify risks in AI Applications, such as:

- Prompt injection attacks
- Toxicity, violent, and other harmful content
- Sensitive Data


The following environment variable is required to use the integration:

- `V1_API_KEY`: A Vision One API Token with AI Guard Permissions

You can optionally set:

- `V1_URL`: The URL for which instances of AI Guard should be invoked
  Defaults to `https://api.xdr.trendmicro.com/beta/aiSecurity/guard` for Vision One's hosted US SaaS deployment

## Setup

[Colang v1](../../../examples/configs/trend_micro/):

```yaml
# config.yml

rails:
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
