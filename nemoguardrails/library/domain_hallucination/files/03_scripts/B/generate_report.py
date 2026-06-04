#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Domain Hallucination Guard - Evaluation Report Generator

Produces a publishable Markdown report from evaluation results JSON,
including comparison tables, charts data, and analysis.

Usage:
    python generate_report.py --results eval_results.json
    python generate_report.py --results current.json --baseline baseline.json
"""

import argparse
import json


def generate_report(results: dict, baseline: dict = None) -> str:
    """Generate a comprehensive Markdown evaluation report."""

    overall = results["overall"]
    by_cat = results.get("by_category", {})
    run_info = results.get("run_info", {})

    lines = []
    L = lines.append

    L("# Domain Hallucination Guard — Evaluation Report\n")
    L(f"**Date:** {run_info.get('timestamp', 'N/A')}  ")
    L(f"**Mode:** {run_info.get('mode', 'N/A')}  ")
    L(f"**Total Test Cases:** {overall['total_samples']}\n")

    # ── Executive Summary
    L("## Executive Summary\n")

    bd = overall["binary_detection"]
    L(
        f"The Domain Hallucination Guard was evaluated on **{overall['total_samples']}** "
        f"test cases spanning {len(by_cat)} categories. "
        f"Overall accuracy was **{overall['accuracy']:.1%}**, with a binary "
        f"hallucination detection F1 of **{bd['f1']:.4f}** "
        f"(Precision: {bd['precision']:.4f}, Recall: {bd['recall']:.4f}).\n"
    )

    if baseline:
        b_overall = baseline.get("overall", {})
        b_bd = b_overall.get("binary_detection", {})
        acc_delta = overall["accuracy"] - b_overall.get("accuracy", 0)
        f1_delta = bd["f1"] - b_bd.get("f1", 0)
        L(
            f"Compared to the baseline, this represents an accuracy improvement "
            f"of **{acc_delta:+.1%}** and a binary F1 improvement of **{f1_delta:+.4f}**.\n"
        )

    # ── Overall Metrics Table
    L("## Overall Metrics\n")
    L("| Metric | Value |")
    L("|--------|-------|")
    L(f"| Accuracy | {overall['accuracy']:.4f} ({overall['accuracy']:.1%}) |")
    L(f"| Weighted F1 | {overall['weighted_f1']:.4f} |")
    L(f"| Macro F1 | {overall['macro_avg']['f1']:.4f} |")
    L(f"| Macro Precision | {overall['macro_avg']['precision']:.4f} |")
    L(f"| Macro Recall | {overall['macro_avg']['recall']:.4f} |")
    L("")

    # ── Binary Detection
    L("## Binary Hallucination Detection\n")
    L("Mapping: `block`/`refine` → **flagged**, `warn`/`pass` → **not flagged**\n")
    L("| Metric | Value |")
    L("|--------|-------|")
    L(f"| Precision | {bd['precision']:.4f} |")
    L(f"| Recall | {bd['recall']:.4f} |")
    L(f"| F1 Score | {bd['f1']:.4f} |")
    L(f"| True Positives | {bd['true_positives']} |")
    L(f"| False Positives | {bd['false_positives']} |")
    L(f"| False Negatives | {bd['false_negatives']} |")
    L(f"| True Negatives | {bd['true_negatives']} |")
    L("")

    # ── Per-Decision Breakdown
    L("## Per-Decision Metrics\n")
    L("| Decision | Precision | Recall | F1 | Support |")
    L("|----------|-----------|--------|-----|---------|")
    for label, m in sorted(overall.get("per_decision", {}).items()):
        L(f"| {label} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | {m['support']} |")
    L("")

    # ── Confusion Matrix
    cm = overall.get("confusion_matrix", {})
    if cm:
        all_labels = sorted(set(list(cm.keys()) + [lbl for row in cm.values() for lbl in row.keys()]))
        L("## Confusion Matrix\n")
        header = "| Actual \\ Predicted | " + " | ".join(all_labels) + " |"
        sep = "|" + "---|" * (len(all_labels) + 1)
        L(header)
        L(sep)
        for true_label in all_labels:
            row = f"| **{true_label}** |"
            for pred_label in all_labels:
                val = cm.get(true_label, {}).get(pred_label, 0)
                if true_label == pred_label and val > 0:
                    row += f" **{val}** |"
                else:
                    row += f" {val} |"
            L(row)
        L("")

    # ── Per-Category Performance
    L("## Per-Category Performance\n")
    L("| Category | Accuracy | Samples | Notes |")
    L("|----------|----------|---------|-------|")
    for cat, m in sorted(by_cat.items()):
        n = m.get("total_samples", 0)
        acc = m.get("accuracy", 0)
        # Visual bar
        bar_len = int(acc * 10)
        bar = "█" * bar_len + "░" * (10 - bar_len)
        L(f"| {cat} | {bar} {acc:.1%} | {n} | |")
    L("")

    # ── Latency
    lat = overall.get("latency_ms", {})
    L("## Latency Performance\n")
    L("| Statistic | Value (ms) |")
    L("|-----------|-----------|")
    L(f"| Mean | {lat.get('mean', 0):.1f} |")
    L(f"| Median (P50) | {lat.get('p50', 0):.1f} |")
    L(f"| P95 | {lat.get('p95', 0):.1f} |")
    L(f"| P99 | {lat.get('p99', 0):.1f} |")
    L(f"| Min | {lat.get('min', 0):.1f} |")
    L(f"| Max | {lat.get('max', 0):.1f} |")
    L("")

    # ── Comparison with Baseline
    if baseline:
        b_overall = baseline.get("overall", {})
        b_bd = b_overall.get("binary_detection", {})
        b_lat = b_overall.get("latency_ms", {})

        L("## Comparison: Baseline vs Current\n")
        L("| Metric | Baseline | Current | Delta | Change |")
        L("|--------|----------|---------|-------|--------|")

        comparisons = [
            ("Accuracy", b_overall.get("accuracy", 0), overall["accuracy"]),
            ("Weighted F1", b_overall.get("weighted_f1", 0), overall["weighted_f1"]),
            ("Macro F1", b_overall.get("macro_avg", {}).get("f1", 0), overall["macro_avg"]["f1"]),
            ("Binary Precision", b_bd.get("precision", 0), bd["precision"]),
            ("Binary Recall", b_bd.get("recall", 0), bd["recall"]),
            ("Binary F1", b_bd.get("f1", 0), bd["f1"]),
            ("Latency (mean ms)", b_lat.get("mean", 0), lat.get("mean", 0)),
        ]

        for name, base_val, curr_val in comparisons:
            delta = curr_val - base_val
            if name.startswith("Latency"):
                arrow = "↓" if delta < 0 else "↑" if delta > 0 else "="
                # For latency, lower is better
                pct = f"{abs(delta) / base_val * 100:.1f}%" if base_val else "N/A"
                change = f"{arrow} {pct}"
            else:
                arrow = "↑" if delta > 0 else "↓" if delta < 0 else "="
                pct = f"{abs(delta) / base_val * 100:.1f}%" if base_val else "N/A"
                change = f"{arrow} {pct}"

            L(f"| {name} | {base_val:.4f} | {curr_val:.4f} | {delta:+.4f} | {change} |")
        L("")

        # Per-category comparison
        b_cat = baseline.get("by_category", {})
        if b_cat:
            L("### Per-Category Accuracy Comparison\n")
            L("| Category | Baseline | Current | Delta |")
            L("|----------|----------|---------|-------|")
            for cat in sorted(set(list(by_cat.keys()) + list(b_cat.keys()))):
                b_acc = b_cat.get(cat, {}).get("accuracy", 0)
                c_acc = by_cat.get(cat, {}).get("accuracy", 0)
                delta = c_acc - b_acc
                arrow = "↑" if delta > 0 else "↓" if delta < 0 else "="
                L(f"| {cat} | {b_acc:.1%} | {c_acc:.1%} | {arrow} {abs(delta):.1%} |")
            L("")

    # ── Error Analysis
    errors = overall.get("errors", [])
    if errors:
        L("## Error Analysis\n")
        L(f"**{len(errors)}** test cases encountered errors during evaluation:\n")
        for err in errors[:10]:
            L(f"- `{err['id']}`: {err['error'][:100]}")
        if len(errors) > 10:
            L(f"- ... and {len(errors) - 10} more")
        L("")

    # ── Failure Analysis
    details = results.get("per_case_details", [])
    failures = [d for d in details if not d.get("correct", True)]
    if failures:
        L("## Misclassification Analysis\n")
        L(f"**{len(failures)}** cases were incorrectly classified:\n")
        L("| ID | Expected | Predicted | Notes |")
        L("|----|----------|-----------|-------|")
        for f in failures[:20]:
            L(f"| {f['id']} | {f['expected']} | {f['predicted']} | |")
        if len(failures) > 20:
            L(f"\n... and {len(failures) - 20} more misclassifications.")
        L("")

    # ── Methodology
    L("## Methodology\n")
    L("### Dataset Composition\n")
    L("The evaluation dataset contains the following categories:\n")
    L("| Category | Description | Expected Behavior |")
    L("|----------|-------------|-------------------|")
    L("| real_links | LLM outputs with verified real URLs/repos | pass |")
    L("| hallucinated_links | LLM outputs with fabricated URLs/repos | block/refine |")
    L("| mixed_links | Mix of real and fake links | refine/warn |")
    L("| no_links | Pure text without URLs | fast pass |")
    L("| typosquatting | Domains mimicking real ones | block/warn |")
    L("| blacklisted | Known malicious domains | block |")
    L("| edge_cases | IP addresses, encoded URLs, redirects, etc. | varies |")
    L("")

    L("### Metrics Definition\n")
    L("- **Accuracy**: Fraction of test cases where the predicted decision matches expected")
    L("- **Binary Detection**: Maps block/refine → 'flagged', warn/pass → 'not flagged'")
    L("- **Precision**: Of all flagged outputs, how many were truly hallucinated")
    L("- **Recall**: Of all truly hallucinated outputs, how many were flagged")
    L("- **F1**: Harmonic mean of precision and recall")
    L("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate evaluation report from results JSON")
    parser.add_argument("--results", required=True, help="Path to evaluation results JSON")
    parser.add_argument("--baseline", default=None, help="Path to baseline results JSON for comparison")
    parser.add_argument("--output", default="eval_report.md", help="Output path for Markdown report")

    args = parser.parse_args()

    with open(args.results) as f:
        results = json.load(f)

    baseline = None
    if args.baseline:
        with open(args.baseline) as f:
            baseline = json.load(f)

    report = generate_report(results, baseline)

    with open(args.output, "w") as f:
        f.write(report)

    print(f"Report generated: {args.output}")
    print(f"({len(report)} characters, {report.count(chr(10))} lines)")


if __name__ == "__main__":
    main()
