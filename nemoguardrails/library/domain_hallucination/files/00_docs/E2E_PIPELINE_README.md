# E2E Domain Hallucination Pipeline

This pipeline evaluates guards on natural LLM-generated answers.

It does not modify core `domain_hallucination` code and does not affect the A/B/C ablation runner.

## Files

- `e2e_eval_pipeline.py`: runnable E2E pipeline
- `question_pool_v2.json`: external question pool, 15 categories and 265 questions

## Guard Modes

| Mode | Meaning |
|---|---|
| `domain-s1` | Non-expert S1: full verification, TLS on, WHOIS on |
| `domain-s2` | Non-expert S2: full verification, skip secondary checks after DNS failure, TLS on, WHOIS on |
| `domain-s3` | Non-expert S3: full verification, skip secondary checks after DNS failure, TLS on, WHOIS off |
| `domain-s4` | Non-expert S4: HTTP-level verification, skip secondary checks after DNS failure, TLS/WHOIS off |
| `domain-expert-s1` | Expert S1 |
| `domain-expert-s2` | Expert S2 |
| `domain-expert-s3` | Expert S3 |
| `domain-expert-s4` | Expert S4 |
| `nemo-hallucination` | NeMo built-in `library/hallucination` baseline |
| `none` | Only generate answers and collect ground truth |

## Smoke Test

```powershell
cd E:\123\Guardrails\nemoguardrails\library\domain_hallucination\files

$env:DEEPSEEK_API_KEY="your key"
$env:HTTP_PROXY="http://127.0.0.1:7897"
$env:HTTPS_PROXY="http://127.0.0.1:7897"

python .\e2e_eval_pipeline.py `
  --llm-provider deepseek `
  --model deepseek-chat `
  --questions question_pool_v2.json `
  --questions-per-category 1 `
  --guard-modes domain-s1 domain-s2 domain-s3 domain-s4 nemo-hallucination `
  --output e2e_smoke_s1_s4_vs_nemo.json
```

The run writes:

```text
e2e_smoke_s1_s4_vs_nemo.json
e2e_smoke_s1_s4_vs_nemo.json.partial.json
```

## Expert S2 Run

```powershell
python .\e2e_eval_pipeline.py `
  --llm-provider deepseek `
  --model deepseek-chat `
  --questions question_pool_v2.json `
  --questions-per-category 1 `
  --guard-modes domain-expert-s1 domain-expert-s2 domain-expert-s3 domain-expert-s4 nemo-hallucination `
  --output e2e_smoke_expert_s1_s4_vs_nemo.json
```
