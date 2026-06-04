# Domain Hallucination Experiment Files

This directory contains datasets, runners, result JSON files, and summary
reports for the domain hallucination guardrail experiments.

The main detailed report is:

```text
README_experiment.md
```

The previous report has been backed up as:

```text
README_experiment_previous.md
readme_previous.md
```

## Current Main Claims

1. `domain_hallucination` is a domain-specific output rail for fabricated URLs,
   domains, and GitHub repositories.
2. The NeMo baseline used here is the official
   `nemoguardrails.library.hallucination.actions.self_check_hallucination`
   action, invoked through the same experiment runner.
3. DeepSeek S2 learned the main fixed calibration boundary:

```text
score_source = recalibrated
expert_policy = none
warn = 25
refine = 45
block = 75
```

4. Cross-model transfer of this fixed boundary is the main generalization
   experiment.
5. Model-specific calibration is an upper-bound analysis, not the primary
   transfer proof.
6. Expert review should be advisory and upgrade-only. It must not lower risk
   against hard DNS/GitHub/HTTP/TLS/WHOIS evidence.

## New 2026-06-04 Outputs

| File | Purpose |
|---|---|
| `e2e_nemo_hallucination_baseline_full_20260604.json` | E2E official NeMo hallucination baseline |
| `e2e_domain_vs_nemo_summary_20260604.json` | E2E aggregate comparison |
| `exp_A_danger_s2_expert_vs_nemo_20260604.json` | Danger-set static Experiment A |
| `exp_A_safe_s2_expert_vs_nemo_20260604.json` | Safe-set static Experiment A |
| `safe_danger_deepseek_thresholds_vs_nemo_20260604.json` | Safe/danger threshold-transfer comparison |

See `README_experiment.md` for the complete protocol and interpretation.
