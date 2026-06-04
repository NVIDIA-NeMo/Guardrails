# Domain Hallucination Experiments

Updated: 2026-06-04

This document records the experiment protocol and current results for the
`domain_hallucination` guardrail. It is intentionally written as an experiment
log, not as a marketing summary. The previous Chinese report has been backed up
as:

- `README_experiment_previous.md`
- `readme_previous.md`

Only files under `nemoguardrails/library/domain_hallucination/files` were changed.
The backup-copy directory next to `files` was not modified.

## 1. Problem Setting

The target failure mode is not general factual hallucination. The target is
link-level and domain-level hallucination in LLM answers:

- fabricated API endpoints
- fabricated official domains
- phishing-like lookalike domains
- fake GitHub repositories
- fake package or documentation links
- mixed answers that contain both real and fabricated links

The guard extracts URLs, domains, and GitHub repositories from the model answer,
then checks them with hard evidence where possible: DNS, HTTP, TLS, WHOIS/RDAP,
GitHub API, local knowledge base evidence, blacklist evidence, semantic checks,
and optional expert review.

The main research question is:

```text
Can a domain-specific verifier reduce unsafe link hallucinations more reliably
than a generic hallucination detector, while keeping false positives acceptable?
```

## 2. Methods Compared

### Domain Hallucination Guard

The proposed method is `nemoguardrails.library.domain_hallucination`. The
runtime path is:

```text
extract URLs/domains/repos
-> verify DNS/HTTP/TLS/WHOIS/GitHub
-> query KB and blacklist evidence
-> aggregate issues
-> compute risk score
-> recalibrate score from hard evidence
-> make pass/warn/refine/block decision
```

The four verification strategies are:

| Strategy | Verification level | Skip secondary checks after DNS failure | TLS | WHOIS/RDAP |
|---|---|---:|---:|---:|
| S1 | full | no | on | on |
| S2 | full | yes | on | on |
| S3 | full | yes | on | off |
| S4 | http | yes | off | off |

### NeMo Hallucination Baseline

The baseline is the official NeMo Guardrails library action:

```text
nemoguardrails.library.hallucination.actions.self_check_hallucination
```

It is a generic self-consistency hallucination detector. It asks the LLM to
generate extra responses and checks whether the original response agrees with
those extra responses. It is not specialized for DNS, GitHub, HTTP, or domain
existence verification.

Important: this baseline is invoked through the same experiment runner and
NeMo runtime, but the detector itself is the official `library/hallucination`
action. It is not the proposed `domain_hallucination` method.

### Expert Review Policy

The expert model should be treated as advisory evidence, not as an authority
that can override hard verification evidence.

The recommended policy is:

```text
expert review may increase risk;
expert review must not lower risk against hard DNS/GitHub/HTTP/TLS/WHOIS evidence;
expert review must not override a hard block caused by strong verifier evidence.
```

In other words, expert review is useful for semantic risk interpretation, but
it should be upgrade-only or preserve-block. It should not be allowed to convert
a hard-evidence `block` into `refine`, `warn`, or `pass`.

## 3. Datasets

| Dataset | Size | Purpose |
|---|---:|---|
| `eval_dataset.json` | 58 | Small static benchmark for quick checks and S1-S4 sweeps |
| `expanded_dataset.json` | 151 | Larger static benchmark for broader domain/repo coverage |
| `full_dataset.json` | 223 | Main static benchmark used for threshold search and model comparison |
| `question_pool_v2.json` | 265 questions | E2E question pool; the LLM first generates an answer, then an independent verifier creates ground truth |
| `eval_dataset_safe.json` | 200 | New safe/low-risk validation set; expected decisions are `pass` or `warn`; main metric is severe false positive rate |
| `eval_dataset_danger.json` | 40 | New high-risk validation set; expected decisions are `block` or `refine`; main metric is dangerous-link catch rate |

Note: `eval_dataset.json` metadata still reports 120 samples, but the current
file contains 58 `test_cases`. `question_pool_v2.json` metadata reports 250
questions, but the current nested pool contains 265 questions.

## 4. Experiment Plan

The paper-facing order should be:

1. Baseline comparison
2. S1-S4 ablation
3. Risk decision calibration
4. Cross-model transfer of the DeepSeek S2 boundary
5. Model-specific calibration upper-bound analysis
6. E2E validation
7. Safe/danger validation with the new JSON files

This order keeps the main claim clean:

```text
DeepSeek S2 learns a useful unified risk boundary.
That boundary can be transferred to other models.
Per-model calibration is useful as an upper-bound analysis, not as the main
cross-model generalization proof.
```

## 5. Static Baseline and Ablation Results

### Non-expert S1-S4 on `eval_dataset.json`

Existing result files:

| Strategy | Result file |
|---|---|
| S1 | `opt_S1_cached_full_eval58.json` |
| S2 | `opt_S2_cached_full_skip_dnsfail_eval58.json` |
| S3 | `opt_S3_cached_full_skip_dnsfail_no_whois_eval58.json` |
| S4 | `opt_S4_http_skip_dnsfail_eval58.json` |

Summary from the existing run:

| Strategy | Accuracy | URL hallucination recall | FPR | Precision | F1 |
|---|---:|---:|---:|---:|---:|
| S1 | 74.14% | 47.37% | 12.82% | 64.29% | 54.55% |
| S2 | 74.14% | 47.37% | 12.82% | 64.29% | 54.55% |
| S3 | 74.14% | 47.37% | 12.82% | 64.29% | 54.55% |
| S4 | 74.14% | 89.47% | 33.33% | 56.67% | 69.39% |

Interpretation: S4 is faster and more recall-oriented, but with much higher
false positives. S3 keeps the S1/S2 metrics while reducing latency by disabling
WHOIS/RDAP.

### Expert S1-S4

Expert-mode S1-S4 has been run for DeepSeek, Qwen, GLM-Air, and OpenRouter
GPT-4.1-mini. Key DeepSeek full223 files:

| Strategy | Result file |
|---|---|
| DeepSeek S1 | `deepseek_expert_S1_cached_full_full223.json` |
| DeepSeek S2 | `deepseek_expert_S2_cached_full_skip_dnsfail_full223.json` |
| DeepSeek S3 | `deepseek_expert_S3_cached_full_skip_dnsfail_no_whois_full223.json` |
| DeepSeek S4 | `deepseek_expert_S4_http_skip_dnsfail_full223.json` |

DeepSeek S2 is the strongest original binary detector among the DeepSeek
strategies: high precision and low false positive rate. However, the original
four-class mapping has weak `block` recall, which motivates risk boundary
calibration.

## 6. Risk Boundary Calibration

### Main Calibration Boundary

The main fixed boundary is learned from DeepSeek S2 on `full_dataset.json`:

```text
score_source = recalibrated
expert_policy = none
warn = 25
refine = 45
block = 75
calibrated_from = deepseek_expert_S2_cached_full_skip_dnsfail_full223.json
```

Result:

| Method | Strict Acc | Balanced Acc | Macro F1 | Block Recall |
|---|---:|---:|---:|---:|
| Original DeepSeek S2 | 54.26% | 42.15% | 30.75% | 4.82% |
| DeepSeek S2 calibrated boundary | 71.30% | 59.55% | 60.18% | 59.04% |
| NeMo hallucination baseline | 62.33% | 38.46% | 34.10% | 100.00% |

Interpretation: the calibrated boundary substantially improves four-class
quality. NeMo baseline catches nearly everything by collapsing many cases into
`block`, but it does not model `warn` and `refine` well.

### Cross-model Transfer

The DeepSeek S2 boundary was applied without per-model retuning to other model
outputs. The summary file is:

```text
deepseek_thresholds_all_models_full223_summary.json
```

Top results:

| Method | Strict Acc | Balanced Acc | Macro F1 |
|---|---:|---:|---:|
| DeepSeek S2 + fixed DeepSeek boundary | 71.30% | 59.55% | 60.18% |
| OpenRouter GPT-4.1-mini S4 + fixed DeepSeek boundary | 69.06% | 57.80% | 58.62% |
| OpenRouter GPT-4.1-mini S1 + fixed DeepSeek boundary | 64.13% | 45.85% | 47.73% |
| Qwen S1 + fixed DeepSeek boundary | 63.23% | 43.82% | 45.17% |
| GLM-Air S1 + fixed DeepSeek boundary | 65.02% | 44.29% | 45.11% |

This is the main generalization evidence: a single boundary learned from
DeepSeek S2 remains useful across other expert models.

### Model-specific Calibration

Model-specific calibration is a supplementary upper-bound analysis. It answers:

```text
If each model learns its own decision boundary, what is its best observed upper
bound on this benchmark?
```

It should not be used as the primary cross-model generalization proof because
each model is allowed to tune to the evaluation distribution.

Summary file:

```text
per_model_threshold_sweep_summary.json
```

Best model-specific results:

| Model family | Best strategy | Best policy | Strict Acc | Macro F1 |
|---|---|---|---:|---:|
| DeepSeek | S2 | recalibrated, none, 25/45/75 | 71.30% | 60.18% |
| Qwen | S1 | recalibrated, none, 25/45/75 | 63.23% | 45.17% |
| GLM-Air | S1 | recalibrated, preserve_block, 25/45/75 | 65.02% | 45.11% |
| GPT-4.1-mini | S4 | recalibrated, preserve_block, 40/75/80 | 70.85% | 63.35% |

Interpretation: GPT-4.1-mini S4 has the strongest model-specific upper bound,
but the main transfer claim remains the DeepSeek S2 fixed-boundary experiment.

## 7. E2E Results

The E2E pipeline uses `question_pool_v2.json`:

```text
question pool -> DeepSeek answer generation -> independent verifier ground truth
-> guard evaluation -> metrics
```

Existing domain E2E file:

```text
e2e_all_strategies_COMPLETE_20260603_174423.json
```

New NeMo E2E baseline file:

```text
e2e_nemo_hallucination_baseline_full_20260604.json
```

Aggregate comparison file:

```text
e2e_domain_vs_nemo_summary_20260604.json
```

Domain E2E summary from the existing complete file:

| Mode | Evaluated | Precision | Recall | F1 | FPR |
|---|---:|---:|---:|---:|---:|
| domain-s1 | 225 | 59.15% | 35.90% | 44.68% | 26.85% |
| domain-s2 | 225 | 59.15% | 35.90% | 44.68% | 26.85% |
| domain-s3 | 225 | 59.15% | 35.90% | 44.68% | 26.85% |
| domain-s4 | 225 | 53.55% | 70.94% | 61.03% | 66.67% |

NeMo E2E baseline:

| Mode | Evaluated | Precision | Recall | F1 | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| nemo-hallucination | 241 | 69.05% | 46.03% | 55.24% | 26 | 68 |

Interpretation: in E2E, NeMo baseline has better precision than domain-s4 and
better recall than domain-s1/S2/S3, but domain-s4 has higher recall at the cost
of a very high false positive rate. The E2E results should be treated as a
pipeline prototype because answer generation, independent verification, and
guard evaluation are all live-system components.

## 8. New Safe/Danger Validation

New files provided on 2026-06-04:

| Dataset | Size | Purpose |
|---|---:|---|
| `eval_dataset_safe.json` | 200 | Safe/low-risk set; expected `pass` or `warn`; test severe false positives |
| `eval_dataset_danger.json` | 40 | High-risk set; expected `block` or `refine`; test dangerous-link catch rate |

The static baseline workflow was kept consistent with the earlier Experiment A:

```text
run_ablation_experiments.py
--experiments A
--domain-verification-level full
--skip-secondary-checks-on-dns-failure
--enable-domain-expert-review
--domain-expert-review-min-level L2
--nemo-config deepseek_config.yml
--hallucination-extra-responses 2
--hallucination-retries 2
```

Result files:

| Dataset | Result file |
|---|---|
| danger | `exp_A_danger_s2_expert_vs_nemo_20260604.json` |
| safe | `exp_A_safe_s2_expert_vs_nemo_20260604.json` |
| fixed-threshold summary | `safe_danger_deepseek_thresholds_vs_nemo_20260604.json` |

### Safe Set

Task-aligned metric: `refine` or `block` on safe/low-risk samples is counted as
a severe false positive. A `warn` is recorded separately as a warning, not as a
hard block.

| Method | Severe FP count | Severe FP rate | Pass rate | Warning rate |
|---|---:|---:|---:|---:|
| Domain original S2 expert | 11 / 200 | 5.50% | 94.50% | 0.00% |
| Domain + fixed DeepSeek boundary | 26 / 200 | 13.00% | 73.00% | 14.00% |
| NeMo hallucination baseline | 43 / 200 | 21.50% | 78.50% | 0.00% |

Interpretation: on the safe set, the original domain guard is the most
deployment-friendly configuration. The fixed DeepSeek boundary is less
conservative than the original policy but still has fewer severe false positives
than the NeMo hallucination baseline.

### Danger Set

Task-aligned metric: `refine` or `block` is counted as a severe catch. `warn` is
counted as a weak catch because it identifies risk but does not force correction
or blocking.

| Method | Severe catch | Warn-only catch | Warn-or-severe catch | Missed |
|---|---:|---:|---:|---:|
| Domain original S2 expert | 20 / 40 | 0 / 40 | 20 / 40 | 20 / 40 |
| Domain + fixed DeepSeek boundary | 19 / 40 | 4 / 40 | 23 / 40 | 17 / 40 |
| NeMo hallucination baseline | 38 / 40 | 0 / 40 | 38 / 40 | 2 / 40 |

Interpretation: NeMo baseline is very aggressive on the danger set. It catches
most dangerous answers but tends to collapse decisions into `block`, which makes
it less suitable for fine-grained `warn/refine/block` decision policy. The fixed
DeepSeek boundary improves weak risk signaling over the original domain policy
but is not enough by itself for high-risk recall on this new danger set.

## 9. Main Conclusions

1. The domain guard is better aligned with deployment safety because it uses hard
   evidence and can keep false positives lower than a generic hallucination
   baseline.
2. The NeMo hallucination baseline remains a useful high-recall comparator. It
   is especially aggressive on danger-only data, but it produces many severe
   false positives on safe/low-risk data.
3. DeepSeek S2 fixed calibration is the main threshold-transfer experiment. It
   is strong on full223 and transfers reasonably to other model outputs.
4. Model-specific calibration should be described as upper-bound analysis, not
   as the main generalization proof.
5. Expert review should be advisory and upgrade-only. It must not override hard
   DNS/GitHub/HTTP/TLS/WHOIS evidence.
6. The new safe/danger validation suggests a practical deployment split:
   original S2 expert policy is safer for low-risk traffic, while additional
   high-risk recall tuning is needed for danger-heavy traffic.

## 10. Files Produced on 2026-06-04

| File | Purpose |
|---|---|
| `e2e_nemo_hallucination_baseline_full_20260604.json` | E2E NeMo hallucination baseline |
| `e2e_domain_vs_nemo_summary_20260604.json` | E2E domain-vs-NeMo aggregate comparison |
| `exp_A_danger_s2_expert_vs_nemo_20260604.json` | Static danger-set Experiment A |
| `exp_A_safe_s2_expert_vs_nemo_20260604.json` | Static safe-set Experiment A |
| `exp_A_danger_s2_expert_vs_nemo_20260604_deepseek_thresholds_20260604.json` | Danger-set fixed DeepSeek boundary remap |
| `exp_A_safe_s2_expert_vs_nemo_20260604_deepseek_thresholds_20260604.json` | Safe-set fixed DeepSeek boundary remap |
| `safe_danger_deepseek_thresholds_vs_nemo_20260604.json` | Safe/danger summary with task-aligned metrics |
| `apply_deepseek_thresholds_to_new_eval.py` | Script used for fixed-boundary remapping on new eval files |

## 11. Recommended Paper Wording

Use this wording for model-specific optimization:

```text
In addition to the unified transfer experiment, we conduct model-specific
calibration to estimate each expert model's upper-bound performance under its
own output distribution. This analysis is supplementary and is not used as the
primary evidence for cross-model generalization.
```

Use this wording for the fixed DeepSeek S2 boundary:

```text
We first learn a single risk decision boundary from DeepSeek S2 and then apply
it unchanged to other expert-model outputs. This setting evaluates whether a
unified calibration rule can transfer across model families.
```

Use this wording for expert review:

```text
The expert model is treated as advisory evidence. It may increase the final
risk decision, but it is not allowed to lower a decision supported by hard
verifier evidence such as DNS failure, GitHub repository absence, HTTP
unreachability, TLS failure, WHOIS/RDAP evidence, or blacklist evidence.
```
