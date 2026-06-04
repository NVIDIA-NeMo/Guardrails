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

"""Per-model safe/dangerous subset comparison with 4-dimension metrics.

Produces three tables per model (mirroring the existing report format):
  - Overall (223 cases): strict_accuracy, balanced_accuracy, macro_f1, block_recall
  - Safe set   (~183 cases, expected in {pass, warn}):  same 4 metrics
  - Dangerous set (~40 cases, expected in {block, refine}): same 4 metrics

Three configurations are compared per model:
  1. Original predictions (as-saved in raw result files)
  2. Per-model calibrated thresholds (from per_model_threshold_sweep_summary.json)
  3. NeMo baseline (from baseline_deepseek_nemo_hallucination_full223.json)

No API calls, no DNS/HTTP/GitHub — purely offline post-processing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent
FILES_ROOT = ROOT.parent.parent
sys.path.insert(0, str(FILES_ROOT / "03_scripts" / "A"))

# Reuse metric helpers from existing scripts (no modification needed)
CAL_ROOT = FILES_ROOT / "05_calibration"
RAW_ROOT = CAL_ROOT / "full223" / "raw"
ANALYSIS_ROOT = CAL_ROOT / "full223" / "analysis"

from threshold_sweep_strict_multiclass import (  # noqa: E402
    collect_domain_details,
    multiclass_metrics,
    simulate_details,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SAFE_LABELS = {"pass", "warn"}
DANGEROUS_LABELS = {"block", "refine"}

MODEL_FILES: Dict[str, Dict[str, str]] = {
    "deepseek": {
        "S1": "deepseek_expert_S1_cached_full_full223.json",
        "S2": "deepseek_expert_S2_cached_full_skip_dnsfail_full223.json",
        "S3": "deepseek_expert_S3_cached_full_skip_dnsfail_no_whois_full223.json",
        "S4": "deepseek_expert_S4_http_skip_dnsfail_full223.json",
    },
    "qwen": {
        "S1": "qwen_expert_S1_cached_full_full223.json",
        "S2": "qwen_expert_S2_cached_full_skip_dnsfail_full223.json",
        "S3": "qwen_expert_S3_cached_full_skip_dnsfail_no_whois_full223.json",
        "S4": "qwen_expert_S4_http_skip_dnsfail_full223.json",
    },
    "glm": {
        "S1": "glm_air_expert_S1_cached_full_full223.json",
        "S2": "glm_air_expert_S2_cached_full_skip_dnsfail_full223.json",
        "S3": "glm_air_expert_S3_cached_full_skip_dnsfail_no_whois_full223_rerun.json",
        "S4": "glm_air_expert_S4_http_skip_dnsfail_full223_rerun.json",
    },
    "gpt41mini": {
        "S1": "openrouter_gpt41mini_expert_S1_cached_full_full223.json",
        "S2": "openrouter_gpt41mini_expert_S2_cached_full_skip_dnsfail_full223.json",
        "S3": "openrouter_gpt41mini_expert_S3_cached_full_skip_dnsfail_no_whois_full223.json",
        "S4": "openrouter_gpt41mini_expert_S4_http_skip_dnsfail_full223.json",
    },
}

NEMO_BASELINE_FILE = "baseline_deepseek_nemo_hallucination_full223.json"
PER_MODEL_SWEEP_SUMMARY = "per_model_threshold_sweep_summary.json"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def normalize(v: Any) -> str:
    s = str(v or "pass").strip().lower()
    return s if s in {"pass", "warn", "refine", "block"} else "pass"


def split_by_subset(details: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    return {
        "overall": details,
        "safe": [d for d in details if normalize(d.get("expected_decision")) in SAFE_LABELS],
        "dangerous": [d for d in details if normalize(d.get("expected_decision")) in DANGEROUS_LABELS],
    }


def extract_4d(metrics: Dict[str, Any]) -> Dict[str, float]:
    return {
        "strict_accuracy": metrics["strict_accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "macro_f1": metrics["macro_f1"],
        "block_recall": metrics["per_class"]["block"]["recall"],
    }


def collect_nemo_baseline_details(path: Path) -> List[Dict[str, Any]]:
    """NeMo baseline uses 'baseline' key instead of 'domain_hallucination'."""
    data = json.loads(path.read_text(encoding="utf-8"))
    details = []
    for result in data.get("results", []):
        method = result.get("baseline") or {}
        for item in method.get("details", []) or []:
            if isinstance(item, dict) and "expected_decision" in item and "predicted" in item:
                details.append(item)
    return details


def compute_row(details: List[Dict[str, Any]], label: str) -> Dict[str, Any]:
    subsets = split_by_subset(details)
    return {
        "label": label,
        "overall": extract_4d(multiclass_metrics(subsets["overall"])),
        "safe": extract_4d(multiclass_metrics(subsets["safe"])) if subsets["safe"] else {},
        "dangerous": extract_4d(multiclass_metrics(subsets["dangerous"])) if subsets["dangerous"] else {},
        "n_overall": len(subsets["overall"]),
        "n_safe": len(subsets["safe"]),
        "n_dangerous": len(subsets["dangerous"]),
    }


def pct(v: Optional[float]) -> str:
    if v is None:
        return "N/A"
    return f"{v * 100:.2f}%"


def print_table(rows: List[Dict[str, Any]], subset: str, title: str) -> None:
    print(f"\n  {title}")
    print(f"  {'方法':<36} {'严格准确率':>10} {'均衡准确率':>10} {'Macro F1':>10} {'Block 召回':>10}")
    print("  " + "-" * 80)
    for row in rows:
        m = row.get(subset, {})
        print(
            f"  {row['label']:<36} "
            f"{pct(m.get('strict_accuracy')):>10} "
            f"{pct(m.get('balanced_accuracy')):>10} "
            f"{pct(m.get('macro_f1')):>10} "
            f"{pct(m.get('block_recall')):>10}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    # Load per-model optimal thresholds
    sweep_path = ANALYSIS_ROOT / PER_MODEL_SWEEP_SUMMARY
    sweep_summary = json.loads(sweep_path.read_text(encoding="utf-8"))
    calibrated_thresholds = sweep_summary.get("comparison", {})

    # Load NeMo baseline
    nemo_path = FILES_ROOT / "04_static_eval" / NEMO_BASELINE_FILE
    nemo_details = collect_nemo_baseline_details(nemo_path) if nemo_path.exists() else []

    all_output: Dict[str, Any] = {}

    for model, strategies in MODEL_FILES.items():
        # Find the model's best strategy file
        cal = calibrated_thresholds.get(model, {})
        best_strategy = cal.get("best_strategy", "S2")
        best_file = strategies.get(best_strategy)

        if not best_file:
            print(f"\n[SKIP] {model}: no file for best strategy {best_strategy}")
            continue

        raw_path = RAW_ROOT / best_file
        if not raw_path.exists():
            print(f"\n[SKIP] {model}: file not found: {best_file}")
            continue

        data = json.loads(raw_path.read_text(encoding="utf-8"))
        original_details = collect_domain_details(data)

        if not original_details:
            print(f"\n[SKIP] {model}: no details in {best_file}")
            continue

        # Apply calibrated thresholds
        thr = cal.get("best_threshold", {})
        calibrated_details = simulate_details(
            original_details,
            warn=float(thr.get("warn", 20)),
            refine=float(thr.get("refine", 40)),
            block=float(thr.get("block", 60)),
            score_source=thr.get("score_source", "recalibrated"),
            expert_policy=thr.get("expert_policy", "none"),
        )

        model_label = model.upper()
        rows = [
            compute_row(original_details, f"{model_label} {best_strategy} 原始策略"),
            compute_row(calibrated_details, f"{model_label} {best_strategy} + 自身最优阈值"),
        ]
        if nemo_details:
            rows.append(compute_row(nemo_details, "NeMo 官方 baseline"))

        n_safe = rows[0]["n_safe"]
        n_dangerous = rows[0]["n_dangerous"]
        n_total = rows[0]["n_overall"]

        print(f"\n{'=' * 80}")
        print(
            f"  模型: {model_label}  最佳策略: {best_strategy}  [总计={n_total}  安全集={n_safe}  危险集={n_dangerous}]"
        )
        print(
            f"  最优阈值: warn={thr.get('warn')}  refine={thr.get('refine')}  "
            f"block={thr.get('block')}  score={thr.get('score_source')}  "
            f"policy={thr.get('expert_policy')}"
        )

        print_table(rows, "overall", f"① 整体 ({n_total}条)")
        print_table(rows, "safe", f"② 安全集 ({n_safe}条，expected∈{{pass,warn}})")
        print_table(rows, "dangerous", f"③ 危险集 ({n_dangerous}条，expected∈{{block,refine}})")

        all_output[model] = {
            "best_strategy": best_strategy,
            "best_threshold": thr,
            "n_overall": n_total,
            "n_safe": n_safe,
            "n_dangerous": n_dangerous,
            "rows": rows,
        }

    # Save JSON output
    out_path = ANALYSIS_ROOT / "per_model_subset_comparison_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n\nFull results saved to {out_path.name}")


if __name__ == "__main__":
    main()
