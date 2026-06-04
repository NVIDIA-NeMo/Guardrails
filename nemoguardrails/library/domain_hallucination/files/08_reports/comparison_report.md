# Domain Hallucination Guard — Evaluation Report

**Date:** 2026-06-02T14:49:33  
**Mode:** dry_run  
**Total Test Cases:** 151

## Executive Summary

The Domain Hallucination Guard was evaluated on **151** test cases spanning 7 categories. Overall accuracy was **100.0%**, with a binary hallucination detection F1 of **1.0000** (Precision: 1.0000, Recall: 1.0000).

Compared to the baseline, this represents an accuracy improvement of **+0.0%** and a binary F1 improvement of **+0.0000**.

## Overall Metrics

| Metric | Value |
|--------|-------|
| Accuracy | 1.0000 (100.0%) |
| Weighted F1 | 1.0000 |
| Macro F1 | 1.0000 |
| Macro Precision | 1.0000 |
| Macro Recall | 1.0000 |

## Binary Hallucination Detection

Mapping: `block`/`refine` → **flagged**, `warn`/`pass` → **not flagged**

| Metric | Value |
|--------|-------|
| Precision | 1.0000 |
| Recall | 1.0000 |
| F1 Score | 1.0000 |
| True Positives | 75 |
| False Positives | 0 |
| False Negatives | 0 |
| True Negatives | 76 |

## Per-Decision Metrics

| Decision | Precision | Recall | F1 | Support |
|----------|-----------|--------|-----|---------|
| block | 1.0000 | 1.0000 | 1.0000 | 58 |
| pass | 1.0000 | 1.0000 | 1.0000 | 64 |
| refine | 1.0000 | 1.0000 | 1.0000 | 17 |
| warn | 1.0000 | 1.0000 | 1.0000 | 12 |

## Confusion Matrix

| Actual \ Predicted | block | pass | refine | warn |
|---|---|---|---|---|
| **block** | **58** | 0 | 0 | 0 |
| **pass** | 0 | **64** | 0 | 0 |
| **refine** | 0 | 0 | **17** | 0 |
| **warn** | 0 | 0 | 0 | **12** |

## Per-Category Performance

| Category | Accuracy | Samples | Notes |
|----------|----------|---------|-------|
| blacklisted | ██████████ 100.0% | 3 | |
| edge_cases | ██████████ 100.0% | 15 | |
| hallucinated_links | ██████████ 100.0% | 43 | |
| mixed_links | ██████████ 100.0% | 19 | |
| no_links | ██████████ 100.0% | 14 | |
| real_links | ██████████ 100.0% | 38 | |
| typosquatting | ██████████ 100.0% | 19 | |

## Latency Performance

| Statistic | Value (ms) |
|-----------|-----------|
| Mean | 0.1 |
| Median (P50) | 0.1 |
| P95 | 0.1 |
| P99 | 0.1 |
| Min | 0.1 |
| Max | 0.1 |

## Comparison: Baseline vs Current

| Metric | Baseline | Current | Delta | Change |
|--------|----------|---------|-------|--------|
| Accuracy | 1.0000 | 1.0000 | +0.0000 | = 0.0% |
| Weighted F1 | 1.0000 | 1.0000 | +0.0000 | = 0.0% |
| Macro F1 | 1.0000 | 1.0000 | +0.0000 | = 0.0% |
| Binary Precision | 1.0000 | 1.0000 | +0.0000 | = 0.0% |
| Binary Recall | 1.0000 | 1.0000 | +0.0000 | = 0.0% |
| Binary F1 | 1.0000 | 1.0000 | +0.0000 | = 0.0% |
| Latency (mean ms) | 0.1000 | 0.1000 | +0.0000 | = 0.0% |

### Per-Category Accuracy Comparison

| Category | Baseline | Current | Delta |
|----------|----------|---------|-------|
| blacklisted | 100.0% | 100.0% | = 0.0% |
| edge_cases | 100.0% | 100.0% | = 0.0% |
| hallucinated_links | 100.0% | 100.0% | = 0.0% |
| mixed_links | 100.0% | 100.0% | = 0.0% |
| no_links | 100.0% | 100.0% | = 0.0% |
| real_links | 100.0% | 100.0% | = 0.0% |
| typosquatting | 100.0% | 100.0% | = 0.0% |

## Methodology

### Dataset Composition

The evaluation dataset contains the following categories:

| Category | Description | Expected Behavior |
|----------|-------------|-------------------|
| real_links | LLM outputs with verified real URLs/repos | pass |
| hallucinated_links | LLM outputs with fabricated URLs/repos | block/refine |
| mixed_links | Mix of real and fake links | refine/warn |
| no_links | Pure text without URLs | fast pass |
| typosquatting | Domains mimicking real ones | block/warn |
| blacklisted | Known malicious domains | block |
| edge_cases | IP addresses, encoded URLs, redirects, etc. | varies |

### Metrics Definition

- **Accuracy**: Fraction of test cases where the predicted decision matches expected
- **Binary Detection**: Maps block/refine → 'flagged', warn/pass → 'not flagged'
- **Precision**: Of all flagged outputs, how many were truly hallucinated
- **Recall**: Of all truly hallucinated outputs, how many were flagged
- **F1**: Harmonic mean of precision and recall
