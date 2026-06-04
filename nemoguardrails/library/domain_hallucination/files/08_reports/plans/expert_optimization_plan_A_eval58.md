# Expert Review Experiment A Sweep on eval_dataset.json

This file records the expert-review variants of S1-S4. These outputs are separate from the previous non-expert files and must not overwrite them.

All runs use:

- Dataset: `eval_dataset.json`
- Cases: 58
- Experiment: A
- Baseline: skipped during sweep
- Expert review: enabled
- Expert trigger threshold: default `L2`
- Incremental save: enabled by default

## Output Files

| Strategy | Output |
|---|---|
| Expert S1 | `expert_S1_cached_full_eval58.json` |
| Expert S2 | `expert_S2_cached_full_skip_dnsfail_eval58.json` |
| Expert S3 | `expert_S3_cached_full_skip_dnsfail_no_whois_eval58.json` |
| Expert S4 | `expert_S4_http_skip_dnsfail_eval58.json` |

Each output also has:

```text
<output>.partial.json
<output>.backup_<timestamp>
<output>.partial.json.backup_<timestamp>
```

## Run Command

```powershell
cd E:\123\Guardrails\nemoguardrails\library\domain_hallucination\files
.\run_A_eval58_expert_sweep.ps1
```
