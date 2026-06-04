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

"""Evaluate safe/danger validation results with 4-dimension metrics.

Reads API results from 07_safe_danger_validation and produces comparison tables:
  - Safe set (200 cases): strict_accuracy, balanced_accuracy, macro_f1, block_recall
  - Danger set (40 cases): same 4 metrics

Methods compared: original DeepSeek, calibrated DeepSeek (old), and NeMo baseline

Purely offline — no API calls.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT.parent
RESULTS_DIR = DATA_ROOT / "07_safe_danger_validation"

LABELS = ["pass", "warn", "refine", "block"]

# ============================================================================
# Helpers (adapted from calculate_strict_multiclass_metrics.py)
# ============================================================================


def normalize_label(value: Any) -> str:
    label = str(value or "pass").strip().lower()
    return label if label in LABELS else "pass"


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def f1_score(precision: float, recall: float) -> float:
    return safe_div(2 * precision * recall, precision + recall)


def confusion_matrix(details: List[Dict[str, Any]], labels: List[str]) -> Dict[str, Dict[str, int]]:
    matrix = {label: {pred: 0 for pred in labels} for label in labels}
    for item in details:
        expected = normalize_label(item.get("expected_decision"))
        predicted = normalize_label(item.get("predicted"))
        matrix[expected][predicted] += 1
    return matrix


def multiclass_metrics(details: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(details)
    if total == 0:
        return {
            "total": 0,
            "strict_accuracy": 0.0,
            "balanced_accuracy": 0.0,
            "macro_f1": 0.0,
            "per_class": {label: {"recall": 0.0} for label in LABELS},
        }

    correct = sum(
        1
        for item in details
        if normalize_label(item.get("expected_decision")) == normalize_label(item.get("predicted"))
    )
    matrix = confusion_matrix(details, LABELS)

    per_class: Dict[str, Dict[str, float]] = {}
    recalls = []
    for label in LABELS:
        tp = matrix[label][label]
        fp = sum(matrix[other][label] for other in LABELS if other != label)
        fn = sum(matrix[label][other] for other in LABELS if other != label)
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        recalls.append(recall)
        per_class[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1_score(precision, recall), 4),
        }

    macro_precision = sum(v["precision"] for v in per_class.values()) / len(LABELS)
    macro_recall = sum(v["recall"] for v in per_class.values()) / len(LABELS)
    macro_f1 = sum(v["f1"] for v in per_class.values()) / len(LABELS)

    return {
        "total": total,
        "strict_accuracy": round(safe_div(correct, total), 4),
        "balanced_accuracy": round(sum(recalls) / len(recalls), 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class,
    }


# ============================================================================
# Main
# ============================================================================


def pct(v: float) -> str:
    return f"{v * 100:.2f}%"


def evaluate_files(original_file: str, calibrated_file: str, label: str) -> Dict[str, Dict[str, Any]]:
    """Evaluate three methods: original DeepSeek, calibrated DeepSeek, and NeMo baseline."""
    original_path = RESULTS_DIR / original_file
    calibrated_path = RESULTS_DIR / calibrated_file

    if not original_path.exists() or not calibrated_path.exists():
        return {}

    # Load original DeepSeek domain_hallucination
    orig_data = json.loads(original_path.read_text(encoding="utf-8"))
    orig_result = orig_data.get("results", [{}])[0]
    orig_dh_details = orig_result.get("domain_hallucination", {}).get("details", [])

    # Load calibrated DeepSeek domain_hallucination
    cal_data = json.loads(calibrated_path.read_text(encoding="utf-8"))
    cal_result = cal_data.get("results", [{}])[0]
    cal_dh_details = cal_result.get("domain_hallucination", {}).get("details", [])

    # Load NeMo baseline (from either file, should be same)
    bl_details = orig_result.get("baseline", {}).get("details", [])

    return {
        label: {
            "原始 DeepSeek S2": multiclass_metrics(orig_dh_details),
            "DeepSeek S2 + 校准边界": multiclass_metrics(cal_dh_details),
            "NeMo 官方 baseline": multiclass_metrics(bl_details),
        }
    }


def print_table(title: str, methods: List[str], rows: Dict[str, Dict[str, Any]]) -> None:
    """Print comparison table."""
    print(f"\n{title}")
    print(f"  {'方法':<32} {'严格准确率':>10} {'均衡准确率':>10} {'Macro F1':>10} {'Block 召回':>10}")
    print("  " + "-" * 74)

    for method in methods:
        if method not in rows:
            continue
        m = rows[method]
        block_recall = m.get("per_class", {}).get("block", {}).get("recall", 0.0)
        print(
            f"  {method:<32} "
            f"{pct(m.get('strict_accuracy', 0)):<10} "
            f"{pct(m.get('balanced_accuracy', 0)):<10} "
            f"{pct(m.get('macro_f1', 0)):<10} "
            f"{pct(block_recall):<10}"
        )


def main() -> None:
    # Evaluate safe set (200 cases)
    safe_eval = evaluate_files(
        "exp_A_safe_s2_expert_vs_nemo_20260604.json",
        "exp_A_safe_s2_expert_vs_nemo_20260604_deepseek_thresholds_20260604.json",
        "safe",
    )

    # Evaluate danger set (40 cases)
    danger_eval = evaluate_files(
        "exp_A_danger_s2_expert_vs_nemo_20260604.json",
        "exp_A_danger_s2_expert_vs_nemo_20260604_deepseek_thresholds_20260604.json",
        "danger",
    )

    if not safe_eval or not danger_eval:
        print("[ERROR] Could not load result files")
        return

    safe_rows = safe_eval["safe"]
    danger_rows = danger_eval["danger"]
    methods = ["原始 DeepSeek S2", "DeepSeek S2 + 校准边界", "NeMo 官方 baseline"]

    # Merge safe + danger for overall evaluation
    merged_rows = {}
    for method in methods:
        safe_path = RESULTS_DIR / "exp_A_safe_s2_expert_vs_nemo_20260604.json"
        danger_path = RESULTS_DIR / "exp_A_danger_s2_expert_vs_nemo_20260604.json"
        safe_cal_path = RESULTS_DIR / "exp_A_safe_s2_expert_vs_nemo_20260604_deepseek_thresholds_20260604.json"
        danger_cal_path = RESULTS_DIR / "exp_A_danger_s2_expert_vs_nemo_20260604_deepseek_thresholds_20260604.json"

        if method == "原始 DeepSeek S2":
            safe_file, danger_file = safe_path, danger_path
            method_key = "domain_hallucination"
        elif method == "DeepSeek S2 + 校准边界":
            safe_file, danger_file = safe_cal_path, danger_cal_path
            method_key = "domain_hallucination"
        else:  # NeMo baseline
            safe_file, danger_file = safe_path, danger_path
            method_key = "baseline"

        # Load safe set details
        with open(safe_file, encoding="utf-8") as f:
            safe_data = json.load(f)
        safe_details = safe_data["results"][0][method_key]["details"]

        # Load danger set details
        with open(danger_file, encoding="utf-8") as f:
            danger_data = json.load(f)
        danger_details = danger_data["results"][0][method_key]["details"]

        # Merge
        merged_details = safe_details + danger_details
        merged_rows[method] = multiclass_metrics(merged_details)

    # Print tables
    print("\n" + "=" * 80)
    print("  ② 安全集验证（200条低风险样本，expected∈{pass,warn}）")
    print_table("", methods, safe_rows)

    print("\n" + "=" * 80)
    print("  ③ 危险集验证（40条高风险样本，expected∈{block,refine}）")
    print_table("", methods, danger_rows)

    print("\n" + "=" * 80)
    print("  ④ 综合验证（240条总数据集，safe + danger）")
    print_table("", methods, merged_rows)

    # Save results
    output = {
        "safe_set": {
            "count": 200,
            **{m: safe_rows[m] for m in methods},
        },
        "danger_set": {
            "count": 40,
            **{m: danger_rows[m] for m in methods},
        },
        "merged_set": {
            "count": 240,
            **{m: merged_rows[m] for m in methods},
        },
    }

    out_path = RESULTS_DIR / "safe_danger_validation_report.json"
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n\nFull results saved to {out_path.name}")


if __name__ == "__main__":
    main()
