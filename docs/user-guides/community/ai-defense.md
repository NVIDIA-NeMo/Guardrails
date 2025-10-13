# Cisco AI Defense Integration

[Cisco AI Defense](https://www.cisco.com/site/us/en/products/security/ai-defense/index.html?utm_medium=github&utm_campaign=nemo-guardrails) allows you to protect LLM interactions. This integration enables NeMo Guardrails to use Cisco AI Defense to protect input and output flows.

## Overview

The diagram below shows how Cisco AI Defense integrates with the NeMo Guardrails flow to provide comprehensive protection at both input and output stages:

```{image} ../../_static/images/guardrails_flow_ai_defense.png
:alt: "High-level flow through programmable guardrails including AI Defense integration, showing how Cisco AI Defense provides privacy, safety, and security inspection for both input and output rails"
:align: center
```

You'll need to set the following environment variables to work with  Cisco AI Defense:

1. AI_DEFENSE_API_ENDPOINT - This is the URL for the Cisco AI Defense inspection API endpoint. This will look like https://[REGION].api.inspect.aidefense.security.cisco.com/api/v1/inspect/chat where REGION is us, ap, eu, etc.
2. AI_DEFENSE_API_KEY - This is the API key for Cisco AI Defense. It is used to authenticate the API request. It can be generated from the [Cisco Security Cloud Control UI](https://security.cisco.com)

## Setup

1. Ensure that you have access to the [Cisco AI Defense endpoints](https://developer.cisco.com/docs/ai-defense/) (SaaS or in your private deployment)
2. Set the required environment variables: `AI_DEFENSE_API_ENDPOINT` and `AI_DEFENSE_API_KEY`

### For Colang 1.0

Enable Cisco AI Defense flows in your `config.yml` file:

```yaml
rails:
  config:
    ai_defense:
      timeout: 30.0
      fail_open: false

  input:
    flows:
      - ai defense inspect prompt

  output:
    flows:
      - ai defense inspect response
```

### For Colang 2.x

You can set configuration options in your `config.yml`:

```yaml
# config.yml
colang_version: "2.x"

rails:
  config:
    ai_defense:
      timeout: 30.0
      fail_open: false
```

Example `rails.co` file:

```colang
import guardrails
import nemoguardrails.library.ai_defense

flow input rails $input_text
  """Check user utterances before they get further processed."""
  ai defense inspect prompt $input_text

flow output rails $output_text
  """Check bot responses before sending them to the user."""
  ai defense inspect response $output_text
```

### Configuration Options

The AI Defense integration supports the following configuration options under `rails.config.ai_defense`:

- **`timeout`** (float, default: 30.0): Timeout in seconds for API requests to the AI Defense service.
- **`fail_open`** (boolean, default: false): Determines the behavior when AI Defense API calls fail:
  - `false` (fail closed): Block content when API calls fail or return malformed responses
  - `true` (fail open): Allow content when API calls fail or return malformed responses

**Note**: Configuration validation failures (missing API key or endpoint) will always block content regardless of the `fail_open` setting.

## Usage

Once configured, the Cisco AI Defense integration will automatically:

1. Protect prompts before they are processed by the LLM.
2. Protect LLM outputs before they are sent back to the user.

The `ai_defense_inspect` action in `nemoguardrails/library/ai_defense/actions.py` handles the protection process.

## Error Handling

The AI Defense integration provides configurable error handling through the `fail_open` setting:

- **Fail Closed (default)**: When `fail_open: false`, API failures and malformed responses will block the content (conservative approach)
- **Fail Open**: When `fail_open: true`, API failures and malformed responses will allow the content to proceed

This allows you to choose between security (fail closed) and availability (fail open) based on your requirements.

### Error Scenarios

1. **API Failures** (network errors, timeouts, HTTP errors): Behavior determined by `fail_open` setting
2. **Malformed Responses** (missing required fields): Behavior determined by `fail_open` setting
3. **Configuration Errors** (missing API key/endpoint): Always fail closed regardless of `fail_open` setting

## Notes

For more information on Cisco AI Defense capabilities and configuration, please refer to the [Cisco AI Defense documentation](https://securitydocs.cisco.com/docs/scc/admin/108321.dita?utm_medium=github&utm_campaign=nemo-guardrails).
