# F5 Guardrails Integration

NeMo Guardrails supports using [F5 AI Guardrails API](https://www.f5.com/products/ai-guardrails) as an input and output rail.

```yaml
rails:
  input:
    flows:
      - f5 guardrails scan input
  output:
    flows:
      - f5 guardrails scan output
```

The F5 Guardrails API scans the text for various violations and returns an outcome. If the outcome is not cleared, the rail will trigger and the bot will refuse to respond.


## Configuration

The following environment variables can be used to configure the integration:

- `F5_GUARDRAILS_API_KEY`: The API key for the F5 Guardrails API.
- `F5_GUARDRAILS_API_URL`: The base URL for the F5 Guardrails API (defaults to https://us1.calypsoai.app).


## Customization

To customize the behavior, you can overwrite the default flows in your configuration. For example, to provide a custom refusal message:

```text
define subflow f5 guardrails scan input
  $result = execute f5_guardrails_scan(text=$user_message)

  if $result.result.outcome != "cleared"
    bot say "I cannot process this request due to safety policies."
    stop
```
