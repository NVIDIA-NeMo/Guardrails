# Zscaler AI Guard Integration

The Zscaler AI Guard guardrail uses the [Zscaler AI Guard](https://help.zscaler.com/ai-guard) API to scan prompts and LLM responses for security threats, including:

- **Credentials and secrets** - API keys, tokens, passwords, and cloud credentials
- **PII (Personally Identifiable Information)** - Names, emails, phone numbers, SSNs, and other personal data
- **Toxicity** - Violent, abusive, hateful, or harmful content
- **Prompt injection** - Adversarial prompts designed to manipulate AI behavior or bypass security controls
- **Malicious URLs** - Known malicious URLs and domains
- **Policy violations** - Custom content policy violations configured in the AI Guard console

The integration uses the [zscaler-sdk-python](https://pypi.org/project/zscaler-sdk-python/) SDK and implements **fail-closed** semantics: if the API call fails or returns an unexpected result, the integration blocks the content by default.

The following environment variables are required:

- `AIGUARD_API_KEY`: Zscaler AI Guard API key (Bearer token). To obtain this key, go to **AI Guard Console** > **Private AI Apps** > **Applications** > **API Keys**.
- `AIGUARD_CLOUD`: Cloud region. Options: `us1` (default), `us2`, `eu1`, `eu2`.

To use a specific policy instead of automatic resolution, set the following optional variable:

- `AIGUARD_POLICY_ID`: Integer policy ID. When set, the integration calls `execute-policy` with the specified ID instead of `resolve-and-execute-policy`.

## Setup

### Colang v1

```yaml
# config.yml

# Optional: show detailed block messages (severity, policy name, detectors)
enable_rails_exceptions: true

rails:
  input:
    flows:
      - zscaler aiguard moderation on input

  output:
    flows:
      - zscaler aiguard moderation on output
```

### Colang v2

```yaml
# config.yml

colang_version: "2.x"
```

```text
# rails.co

import guardrails
import nemoguardrails.library.zscaler_aiguard

flow input rails $input_text
    zscaler aiguard moderation on input

flow output rails $output_text
    zscaler aiguard moderation on output
```

## How It Works

1. **Input scanning**: Before user prompts reach the LLM, the `zscaler aiguard moderation on input` flow sends the prompt to the AI Guard API with `direction="IN"`.
2. **Output scanning**: After the LLM generates a response, the `zscaler aiguard moderation on output` flow sends the response to the AI Guard API with `direction="OUT"`.
3. **Policy selection**: By default, the integration calls `resolve-and-execute-policy`, which automatically selects the appropriate policy. If `AIGUARD_POLICY_ID` is set, the integration calls `execute-policy` with the specified policy ID instead.
4. **Verdict handling**: If the API returns an `action` of `BLOCK`, the flow aborts and the bot refuses to respond. If the API returns `ALLOW` or `DETECT`, the content passes through normally.
5. **Rails exceptions**: Setting `enable_rails_exceptions: true` at the top level of the config causes blocked requests to emit a `ZscalerAiguardInputRailException` or `ZscalerAiguardOutputRailException`. The exception message contains the severity, policy name, blocking detectors, and transaction ID.
6. **Fail-closed**: If the API call fails (network error, timeout, or authentication failure), the action returns `action: BLOCK` by default to prevent potentially unsafe content from passing through.

## Detectors

The AI Guard policy engine evaluates all configured detectors and returns a single `ALLOW` / `BLOCK` / `DETECT` verdict. Available detectors include:

- `toxicity` - Toxic or harmful language
- `pii` - Personally identifiable information
- `personalData` - Personal data patterns
- `piiDeepscan` - Deep PII scanning
- `secrets` - Credentials, API keys, tokens
- `promptInjection` - Prompt injection attempts
- `maliciousUrl` - Malicious URL detection

Configure detectors in the [Zscaler AI Guard console](https://help.zscaler.com/aiguard), not in the NeMo Guardrails configuration.

## Dependencies

Install the required SDK:

```bash
pip install zscaler-sdk-python
```
