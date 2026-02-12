# CrowdStrike AIDR integration

The CrowdStrike AIDR guardrail uses configurable detection policies to identify and mitigate risks in AI application
traffic, including:

- **Prompt injection and jailbreak attempts** - Adversarial prompts designed to manipulate AI behavior or bypass security controls
- **Sensitive data exposure** - PII, credentials, financial data, and confidential information in prompts and responses via built-in patterns, natural language processing, and custom definitions
- **Malicious entities** - Known malicious URLs, IP addresses, and domains in AI outputs using integrated threat intelligence
- **Toxic and harmful content** - Violent, abusive, or harmful content in AI inputs and outputs
- **Language** - Language detection with optional use of an allowlist or denylist
- **Topic violations** - Configurable content category restrictions

All detections are logged in an audit trail for analysis, attribution, and incident response. Note that this guardrail
operates in a fail-open mode.

The following environment variable is required to use the CrowdStrike AIDR integration:

- `CS_AIDR_TOKEN`: CrowdStrike AIDR API token.

You can also optionally set:

- `CS_AIDR_BASE_URL_TEMPLATE`: Template for constructing the base URL for API requests. The `{SERVICE_NAME}` placeholder will be replaced with the service name slug.
  Defaults to `https://api.crowdstrike.com/aidr/{SERVICE_NAME}`.

## Setup

Colang v1:

```yaml
# config.yml

rails:
  config:
    crowdstrike_aidr:
      timeout: 30.0  # Optional request timeout in seconds. Defaults to 30 seconds.

  input:
    flows:
      - crowdstrike aidr guard input

  output:
    flows:
      - crowdstrike aidr guard output
```

Colang v2:

```yaml
# config.yml

colang_version: "2.x"

rails:
  config:
    crowdstrike_aidr:
      timeout: 30.0  # Optional request timeout in seconds. Defaults to 30 seconds.
```

```
# rails.co

import guardrails
import nemoguardrails.library.crowdstrike_aidr

flow input rails $input_text
    crowdstrike aidr guard input

flow output rails $output_text
    crowdstrike aidr guard output
```
