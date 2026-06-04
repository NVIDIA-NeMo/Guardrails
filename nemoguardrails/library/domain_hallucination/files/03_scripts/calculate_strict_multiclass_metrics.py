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

"""Calculate strict multi-class metrics from saved experiment JSON files.

This is an offline evaluator. It does not call any detector, LLM, DNS, HTTP,
GitHub, or WHOIS service.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

ROOT = Path(__file__).resolve().parent
FILES_ROOT = ROOT.parent
RAW_FULL223_ROOT = FILES_ROOT / "05_calibration" / "full223" / "raw"
STATIC_EVAL_ROOT = FILES_ROOT / "04_static_eval"
LABELS = ["pass", "warn", "refine", "block"]
FLAGGED = {"warn", "refine", "block"}


def normalize_label(value: Any) -> str:
    label = str(value or "pass").strip().lower()
    return label if label in LABELS else "pass"


def binary_label(value: str) -> str:
    return "flagged" if normalize_label(value) in FLAGGED else "not_flagged"


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def f1_score(precision: float, recall: float) -> float:
    return safe_div(2 * precision * recall, precision + recall)


def collect_details(data: Dict[str, Any], method_key: str) -> List[Dict[str, Any]]:
    details: List[Dict[str, Any]] = []
    for result in data.get("results", []):
        method = result.get(method_key) or {}
        for item in method.get("details", []) or []:
            if isinstance(item, dict) and "expected_decision" in item and "predicted" in item:
                details.append(item)
    return details


def confusion_matrix(details: Iterable[Dict[str, Any]], labels: List[str]) -> Dict[str, Dict[str, int]]:
    matrix = {label: {pred: 0 for pred in labels} for label in labels}
    for item in details:
        expected = normalize_label(item.get("expected_decision"))
        predicted = normalize_label(item.get("predicted"))
        matrix[expected][predicted] += 1
    return matrix


def multiclass_metrics(details: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(details)
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
            "support": sum(matrix[label].values()),
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
        "confusion_matrix": matrix,
    }


def binary_metrics(details: List[Dict[str, Any]]) -> Dict[str, Any]:
    counts = Counter()
    for item in details:
        expected = binary_label(item.get("expected_decision"))
        predicted = binary_label(item.get("predicted"))
        if expected == "flagged" and predicted == "flagged":
            counts["tp"] += 1
        elif expected == "not_flagged" and predicted == "flagged":
            counts["fp"] += 1
        elif expected == "flagged" and predicted == "not_flagged":
            counts["fn"] += 1
        else:
            counts["tn"] += 1

    tp, fp, fn, tn = counts["tp"], counts["fp"], counts["fn"], counts["tn"]
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    return {
        "accuracy": round(safe_div(tp + tn, tp + fp + fn + tn), 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1_score(precision, recall), 4),
        "false_positive_rate": round(safe_div(fp, fp + tn), 4),
        "specificity": round(specificity, 4),
        "balanced_accuracy": round((recall + specificity) / 2, 4),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


def summarize_file(path: Path, method_key: str) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    details = collect_details(data, method_key)
    return {
        "file": path.name,
        "method_key": method_key,
        "binary": binary_metrics(details),
        "strict_multiclass": multiclass_metrics(details),
    }


def resolve_input_path(filename: str) -> Path:
    path = Path(filename)
    if path.is_absolute():
        return path

    candidates = [
        path,
        ROOT / filename,
        FILES_ROOT / filename,
        RAW_FULL223_ROOT / filename,
        STATIC_EVAL_ROOT / filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    return FILES_ROOT / filename


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def print_summary(rows: List[Dict[str, Any]]) -> None:
    print(
        "| file | method | binary_acc | binary_recall | binary_fpr | binary_precision | binary_f1 | strict_acc | strict_bal_acc | macro_f1 |"
    )
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        binary = row["binary"]
        strict = row["strict_multiclass"]
        print(
            "| "
            + " | ".join(
                [
                    row["file"],
                    row["method_key"],
                    pct(binary["accuracy"]),
                    pct(binary["recall"]),
                    pct(binary["false_positive_rate"]),
                    pct(binary["precision"]),
                    pct(binary["f1"]),
                    pct(strict["strict_accuracy"]),
                    pct(strict["balanced_accuracy"]),
                    pct(strict["macro_f1"]),
                ]
            )
            + " |"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate strict multi-class metrics from saved results.")
    parser.add_argument(
        "items",
        nargs="*",
        help="Pairs in the form result.json:domain_hallucination or result.json:baseline.",
    )
    parser.add_argument(
        "--output",
        default="strict_multiclass_metrics.json",
        help="Output JSON file under files/.",
    )
    args = parser.parse_args()

    items = args.items or [
        str(RAW_FULL223_ROOT / "deepseek_expert_S2_cached_full_skip_dnsfail_full223.json") + ":domain_hallucination",
        str(STATIC_EVAL_ROOT / "baseline_deepseek_nemo_hallucination_full223.json") + ":baseline",
    ]

    rows = []
    for item in items:
        if ":" in item:
            filename, method_key = item.rsplit(":", 1)
        else:
            filename, method_key = item, "domain_hallucination"
        rows.append(summarize_file(resolve_input_path(filename), method_key))

    output = STATIC_EVAL_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print_summary(rows)
    print(f"\nSaved strict metrics to {output}")


if __name__ == "__main__":
    main()
