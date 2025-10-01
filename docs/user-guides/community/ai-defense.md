# Cisco AI Defense Integration

[Cisco AI Defense](https://www.cisco.com/site/us/en/products/security/ai-defense/index.html?utm_medium=github&utm_campaign=nemo-guardrails) allows you to protect LLM interactions. This integration enables NeMo Guardrails to use Cisco AI Defense to protect input and output flows.

You'll need to set the following env variables to work with  Cisco AI Defense:

1. AI_DEFENSE_API_ENDPOINT - This is the URL for the Cisco AI Defense inspection API endpoint. This will look like https://[REGION].api.inspect.aidefense.security.cisco.com/api/v1/inspect/chat where REGION is us, ap, eu, etc.
2. AI_DEFENSE_API_KEY - This is the API key for Cisco AI Defense. It is used to authenticate the API request. It can be generated from the Cisco Security Cloud Control UI at https://security.cisco.com

## Setup

1. Ensure that you have access to the Cisco AI Defense endpoints (SaaS or in your private deployment)
2. Enable Cisco AI Defense flows in your `config.yml` file:

```yaml
rails:
  input:
    flows:
      - ai defense inspect prompt

  output:
    flows:
      - ai defense inspect response
```

Don't forget to set the `AI_DEFENSE_API_ENDPOINT` and `AI_DEFENSE_API_KEY` environment variables.

## Usage

Once configured, the Cisco AI Defense integration will automatically:

1. Protect prompts before they are processed by the LLM.
2. Protect LLM outputs before they are sent back to the user.

The `ai_defense_inspect` action in `nemoguardrails/library/ai_defense/actions.py` handles the protection process.

## Error Handling

If the Cisco AI Defense API request fails, it will operate in a fail-open mode (not blocking the prompt/response).

## Notes

For more information on Cisco AI Defense capabilities and configuration, please refer to the [Cisco AI Defense documentation](https://securitydocs.cisco.com/docs/scc/admin/108321.dita?utm_medium=github&utm_campaign=nemo-guardrails).
