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

"""Apply the DeepSeek-calibrated strict four-class thresholds to all model files.

This is an offline evaluator. It does not call DNS, HTTP, GitHub, WHOIS, or any
LLM. It reuses saved scores from previous experiment JSON files and remaps
predictions with the fixed DeepSeek S2 thresholds:

    score_source = recalibrated
    expert_policy = none
    warn = 25
    refine = 45
    block = 75
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "A"))
from threshold_sweep_strict_multiclass import (
    collect_domain_details,
    multiclass_metrics,
    replace_domain_details,
    simulate_details,
)

ROOT = Path(__file__).resolve().parent
FILES_ROOT = ROOT.parent.parent
CAL_ROOT = FILES_ROOT / "05_calibration"
RAW_ROOT = CAL_ROOT / "full223" / "raw"
CALIBRATED_ROOT = CAL_ROOT / "full223" / "calibrated"
ANALYSIS_ROOT = CAL_ROOT / "full223" / "analysis"
THRESHOLD_CONFIG = {
    "score_source": "recalibrated",
    "expert_policy": "none",
    "warn_threshold": 25.0,
    "refine_threshold": 45.0,
    "block_threshold": 75.0,
    "calibrated_from": "deepseek_expert_S2_cached_full_skip_dnsfail_full223.json",
}

MODEL_FILES = [
    "deepseek_expert_S1_cached_full_full223.json",
    "deepseek_expert_S2_cached_full_skip_dnsfail_full223.json",
    "deepseek_expert_S3_cached_full_skip_dnsfail_no_whois_full223.json",
    "deepseek_expert_S4_http_skip_dnsfail_full223.json",
    "qwen_expert_S1_cached_full_full223.json",
    "qwen_expert_S2_cached_full_skip_dnsfail_full223.json",
    "qwen_expert_S3_cached_full_skip_dnsfail_no_whois_full223.json",
    "qwen_expert_S4_http_skip_dnsfail_full223.json",
    "glm_air_expert_S1_cached_full_full223.json",
    "glm_air_expert_S2_cached_full_skip_dnsfail_full223.json",
    "glm_air_expert_S3_cached_full_skip_dnsfail_no_whois_full223_rerun.json",
    "glm_air_expert_S4_http_skip_dnsfail_full223_rerun.json",
    "openrouter_gpt41mini_expert_S1_cached_full_full223.json",
    "openrouter_gpt41mini_expert_S2_cached_full_skip_dnsfail_full223.json",
    "openrouter_gpt41mini_expert_S3_cached_full_skip_dnsfail_no_whois_full223.json",
    "openrouter_gpt41mini_expert_S4_http_skip_dnsfail_full223.json",
]


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def method_label(filename: str) -> str:
    if filename.startswith("deepseek"):
        model = "DeepSeek"
    elif filename.startswith("qwen"):
        model = "Qwen"
    elif filename.startswith("glm_air"):
        model = "GLM-Air"
    elif filename.startswith("openrouter_gpt41mini"):
        model = "OpenRouter GPT-4.1-mini"
    else:
        model = "Unknown"
    strategy = "S?"
    for candidate in ("S1", "S2", "S3", "S4"):
        if f"_{candidate}_" in filename:
            strategy = candidate
            break
    return f"{model} {strategy}"


def binary_metrics(details: List[Dict[str, Any]]) -> Dict[str, Any]:
    flagged = {"warn", "refine", "block"}
    tp = fp = fn = tn = 0
    for item in details:
        expected = str(item.get("expected_decision") or "pass").lower()
        predicted = str(item.get("predicted") or "pass").lower()
        expected_flagged = expected in flagged
        predicted_flagged = predicted in flagged
        if expected_flagged and predicted_flagged:
            tp += 1
        elif not expected_flagged and predicted_flagged:
            fp += 1
        elif expected_flagged and not predicted_flagged:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    total = tp + fp + fn + tn
    return {
        "accuracy": round((tp + tn) / total, 4) if total else 0.0,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "false_positive_rate": round(fpr, 4),
        "f1": round(f1, 4),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


def process_file(filename: str) -> Dict[str, Any]:
    source_path = RAW_ROOT / filename
    data = json.loads(source_path.read_text(encoding="utf-8"))
    details = collect_domain_details(data)
    remapped = simulate_details(
        details,
        warn=THRESHOLD_CONFIG["warn_threshold"],
        refine=THRESHOLD_CONFIG["refine_threshold"],
        block=THRESHOLD_CONFIG["block_threshold"],
        score_source=THRESHOLD_CONFIG["score_source"],
        expert_policy=THRESHOLD_CONFIG["expert_policy"],
    )
    strict = multiclass_metrics(remapped)
    binary = binary_metrics(remapped)

    output_data = replace_domain_details(data, remapped, strict)
    output_data["fixed_threshold_remap"] = {
        "source_file": filename,
        **THRESHOLD_CONFIG,
        "binary_metrics": binary,
        "strict_multiclass_metrics": strict,
    }

    stem = source_path.stem
    output_name = f"{stem}_deepseek_thresholds.json"
    output_path = CALIBRATED_ROOT / output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output_data, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "method": method_label(filename),
        "source_file": filename,
        "output_file": output_name,
        "binary": binary,
        "strict_multiclass": strict,
    }


def print_summary(rows: List[Dict[str, Any]]) -> None:
    print("| method | output | binary_f1 | precision | recall | fpr | strict_acc | balanced_acc | macro_f1 |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        b = row["binary"]
        s = row["strict_multiclass"]
        print(
            "| "
            + " | ".join(
                [
                    row["method"],
                    row["output_file"],
                    pct(b["f1"]),
                    pct(b["precision"]),
                    pct(b["recall"]),
                    pct(b["false_positive_rate"]),
                    pct(s["strict_accuracy"]),
                    pct(s["balanced_accuracy"]),
                    pct(s["macro_f1"]),
                ]
            )
            + " |"
        )


def main() -> None:
    rows = []
    errors = []
    for filename in MODEL_FILES:
        try:
            rows.append(process_file(filename))
        except Exception as exc:
            errors.append({"file": filename, "error": f"{type(exc).__name__}: {exc}"})

    rows.sort(
        key=lambda row: (
            row["strict_multiclass"]["macro_f1"],
            row["strict_multiclass"]["balanced_accuracy"],
            row["strict_multiclass"]["strict_accuracy"],
        ),
        reverse=True,
    )
    summary = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "threshold_config": THRESHOLD_CONFIG,
        "selection_note": "Rows are sorted by strict multiclass macro_f1.",
        "results": rows,
        "errors": errors,
    }
    summary_path = ANALYSIS_ROOT / "deepseek_thresholds_all_models_full223_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print_summary(rows)
    if rows:
        best = rows[0]
        print(
            "\nBest macro-F1 after fixed DeepSeek thresholds: "
            f"{best['method']} -> {pct(best['strict_multiclass']['macro_f1'])}"
        )
    if errors:
        print("\nErrors:")
        for item in errors:
            print(f"- {item['file']}: {item['error']}")
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
