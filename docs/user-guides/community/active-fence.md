# ActiveFence Integration

NeMo Guardrails supports using the [ActiveFence ActiveScore API](https://docs.activefence.com/index.html) as an input rail out-of-the-box (you need to have the `ACTIVEFENCE_API_KEY` environment variable set).

```yaml
rails:
  input:
    flows:
      # The simplified version
      - activefence moderation

      # The detailed version with individual risk scores
      # - activefence moderation detailed
```

The `activefence moderation` flow uses the maximum risk score with an 0.85 threshold to decide if the input should be allowed or not (i.e., if the risk score is above the threshold, it is considered a violation). The `activefence moderation detailed` has individual scores per category of violation.

To customize the scores, you have to overwrite the [default flows](https://github.com/NVIDIA/NeMo-Guardrails/tree/develop/nemoguardrails/library/activefence/flows.co) in your config. For example, to change the threshold for `activefence moderation` you can add the following flow to your config:

```colang
define subflow activefence moderation
  """Guardrail based on the maximum risk score."""
  $result = execute call activefence api

  if $result.max_risk_score > 0.85
    bot inform cannot answer
    stop
```

ActiveFence’s ActiveScore API gives flexibility in controlling the behavior of various supported violations individually. To leverage that, you can use the violations dictionary (`violations_dict`), one of the outputs from the API, to set different thresholds for different violations. Below is an example of one such input moderation flow:

```colang
define flow activefence input moderation detailed
  $result = execute call activefence api

  if $result.violations.get("abusive_or_harmful.hate_speech", 0) > 0.8
    bot inform cannot engage in abusive or harmful behavior
    stop

define bot inform cannot engage in abusive or harmful behavior
  "I will not engage in any abusive or harmful behavior."
```
