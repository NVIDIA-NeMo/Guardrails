# GCP Text Moderation Integration

NeMo Guardrails supports using the [GCP Text Modeation API](https://cloud.google.com/natural-language/docs/moderating-text) as an input rail out-of-the-box. There are many ways you can authentication on GCP, refer to this [link](https://cloud.google.com/docs/authentication/application-default-credentials) for more details .

```yaml
rails:
  input:
    flows:
      # The simplified version
      - gcpnlp moderation

      # The detailed version with individual risk scores
      # - gcpnlp moderation detailed
```

The `gcpnlp moderation` flow uses the maximum risk score with an 0.80 threshold to decide if the input should be allowed or not (i.e., if the risk score is above the threshold, it is considered a violation). The `gcpnlp moderation detailed` has individual scores per category of violation.

To customize the scores, you have to overwrite the [default flows](https://github.com/NVIDIA/NeMo-Guardrails/tree/develop/nemoguardrails/library/gcp_moderate_text/flows.co) in your config. For example, to change the threshold for `gcpnlp moderation` you can add the following flow to your config:

```colang
define subflow gcpnlp moderation
  """Guardrail based on the maximum risk score."""
  $result = execute call gcpnlp api

  if $result.max_risk_score > 0.8
    bot inform cannot answer
    stop
```

Using GCP Text Moderation user can control various violations individually. Post text analyzing, GCP returns "moderation_categories" protobuff object containing name of the voilation categories in which text falls in & probablity of the prediction. For accessibility it in "violations_dict" in currently implemented action. Below is an example of one such input moderation flow:

```colang
define flow gcpnlp input moderation detailed
  $result = execute call gcpnlp api(text=$user_message)

  if $result.violations.get("Toxic", 0) > 0.8
    bot inform cannot engage in abusive or harmful behavior
    stop

define bot inform cannot engage in abusive or harmful behavior
  "I will not engage in any abusive or harmful behavior."
```
