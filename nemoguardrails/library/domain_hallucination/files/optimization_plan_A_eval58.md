# Experiment A Optimization Plan on eval_dataset.json

Objective: reduce `domain_hallucination` full-mode latency on the 58-case native dataset while avoiding a large increase in false positives or false negatives.

Baseline reference already available:

- `exp_A_full_hallucination_vs_domain.json`
- backup: `exp_A_full_hallucination_vs_domain.backup_20260602_203701.json`

All new runs write incremental progress to:

```text
<output>.partial.json
```

This means every completed case is persisted even if the run is interrupted.

## Metric Notes

`url_hallucination_recall` means: among all cases that should be flagged as URL/domain/GitHub hallucination, the fraction successfully flagged by the detector.

Higher recall is better because it means fewer hallucinated links are missed. It must be interpreted with `false_positive_rate` and `precision`: a detector can get high recall by blocking too much.

## Strategy Set

### S1: Cached Full

Purpose: measure the effect of cross-case caching while keeping full verification enabled.

```powershell
python run_ablation_experiments.py `
  --experiments A `
  --datasets eval_dataset.json `
  --domain-verification-level full `
  --skip-baseline `
  --output opt_S1_cached_full_eval58.json
```

### S2: Cached Full + Skip Secondary Checks After DNS Failure

Purpose: avoid HTTP/TLS/WHOIS work when DNS already proves the domain is non-resolvable.

```powershell
python run_ablation_experiments.py `
  --experiments A `
  --datasets eval_dataset.json `
  --domain-verification-level full `
  --skip-secondary-checks-on-dns-failure `
  --skip-baseline `
  --output opt_S2_cached_full_skip_dnsfail_eval58.json
```

### S3: Cached Full + Skip DNS-Failed Secondary Checks + No WHOIS

Purpose: test whether WHOIS/RDAP is the main latency source and whether removing it hurts detection.

```powershell
python run_ablation_experiments.py `
  --experiments A `
  --datasets eval_dataset.json `
  --domain-verification-level full `
  --skip-secondary-checks-on-dns-failure `
  --disable-domain-whois `
  --skip-baseline `
  --output opt_S3_cached_full_skip_dnsfail_no_whois_eval58.json
```

### S4: HTTP-Level Verification + Skip DNS-Failed Secondary Checks

Purpose: keep DNS, HTTP, GitHub, KB, semantic and advanced heuristics, but avoid TLS/WHOIS.

```powershell
python run_ablation_experiments.py `
  --experiments A `
  --datasets eval_dataset.json `
  --domain-verification-level http `
  --skip-secondary-checks-on-dns-failure `
  --skip-baseline `
  --output opt_S4_http_skip_dnsfail_eval58.json
```

## Final Confirmation Run

After choosing the best strategy, rerun that strategy without `--skip-baseline` to produce the final A comparison against NeMo native hallucination.

Example for S2:

```powershell
python run_ablation_experiments.py `
  --experiments A `
  --datasets eval_dataset.json `
  --domain-verification-level full `
  --skip-secondary-checks-on-dns-failure `
  --output final_A_S2_full_skip_dnsfail_vs_nemo_eval58.json `
  --hallucination-retries 3
```

