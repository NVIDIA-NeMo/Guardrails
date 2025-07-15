# Guardrail Evaluation

The Guardrail Evaluation UI enables you to review your evaluation results.

- Go to <a href="/Config" target="_self">Config</a> to review details about the evaluation configuration.
- Go to <a href="/Review" target="_self">Review</a> to review your evaluation results individually.
- Go to <a href="/Summary_-_Short" target="_self">Summary - Short</a> to view a short summary of the evaluation results.
- Go to <a href="/Summary_-_Detailed" target="_self">Summary - Detailed</a> to view a detailed summary of the evaluation results.

## Usage

Below is a getting started guide for the `nemoguardrails eval` CLI.

## Run Evaluations

To run a new evaluation with a guardrail configuration:

```bash
nemoguardrails eval run -g <GUARDRAIL_CONFIG_PATH> -o <OUTPUT_PATH>
```

## Check Compliance

To check the compliance with the policies, you can use the LLM-as-a-judge method.

```bash
nemoguardrails eval check-compliance --llm-judge=<LLM_MODEL_NAME> -o <OUTPUT_PATH>
```

You can use any LLM supported by NeMo Guardrails.

```yaml
models:
  - type: llm-judge
    engine: openai
    model: gpt-4

  - type: llm-judge
    engine: nvidia_ai_endpoints
    model: meta/llama3-8b-instruct
```

## Review and Analyze

To review and analyze the results, launch the NeMo Guardrails Eval UI:

```bash
nemoguardrails eval ui
```
