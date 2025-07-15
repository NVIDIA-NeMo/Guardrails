# Patronus Evaluate API Integration

NeMo Guardrails supports using [Patronus AI](www.patronus.ai)'s Evaluate API as an output rail. The Evaluate API gives you access to Patronus' powerful suite of fully-managed in-house evaluation models, including [Lynx](patronus-lynx.md), Judge (a hosted LLM-as-a-Judge model), Toxicity, PII, and PHI models, and a suite of specialized RAG evaluators with
industry-leading performance on metrics like Answer Relevance, Context Relevance, Context Sufficiency, and Hallucination.

Patronus also has Managed configurations of the Judge evaluator, which you can use to detect AI failures like prompt injection and brand misalignment in order to prevent problematic bot responses from being returned to users.

## Setup

1. Sign up for an account on [app.patronus.ai](https://app.patronus.ai).
2. You can follow the Quick Start guide [here](https://docs.patronus.ai/docs/quickstart-guide) to get onboarded.
3. Create an API Key and save it somewhere safe.

## Usage

Here's how to use the Patronus Evaluate API as an output rail:

1. Get a Patronus API key and set it to the PATRONUS_API_KEY variable in your environment.

2. Add the guardrail `patronus api check output` to your output rails in `config.yml`:

```yaml
rails:
  output:
    flows:
      - patronus api check output
```

3. Add a rails config for Patronus in `config.yml`:

```yaml
rails:
  config:
    patronus:
      output:
        evaluate_config:
          success_strategy: "all_pass"
          params:
            {
              evaluators:
                [
                  { "evaluator": "lynx" },
                  {
                    "evaluator": "answer-relevance",
                    "explain_strategy": "on-fail",
                  },
                ],
              tags: { "retrieval_configuration": "ast-123" },
            }
```

The `evaluate_config` has two top-level arguments: `success_strategy` and `params`.

In `params` you can pass the relevant arguments to the Patronus Evaluate API. The schema is the same as the API documentation [here](https://docs.patronus.ai/reference/evaluate_v1_evaluate_post), so as new API parameters are added and new values are supported, you can readily add them to your NeMo Guardrails configuration.

Note that you can pass in multiple evaluators to the Patronus Evaluate API. By setting `success_strategy` to "all_pass",
every single evaluator called in the Evaluate API must pass for the rail to pass successfully. If you set it to "any_pass", then only one evaluator needs to pass.

## Additional Information

For now, the Evaluate API Integration only looks at whether the evaluators return Pass or Fail in the API response. However, most evaluators return a score between 0 and 1, where by default a score below 0.5 indicates a Fail and score above 0.5 indicates a Pass. But you can use the score directly to adjust how sensitive your pass/fail threshold should be. The API response can also include explanations of why the rail passed or failed that could be surfaced to a user (set `explain_strategy` in the evaluator object). Some evaluators even include spans of problematic keywords or sentences where issues like hallucinations are present, so you can scrub them out before returning the bot response.

Here's the `patronus api check output` flow, showing how the action is executed:

```colang
define bot inform answer unknown
  "I don't know the answer to that."

define flow patronus api check output
  $patronus_response = execute PatronusApiCheckOutputAction
  $evaluation_passed = $patronus_response["pass"]

  if not $evaluation_passed
    bot inform answer unknown
```
