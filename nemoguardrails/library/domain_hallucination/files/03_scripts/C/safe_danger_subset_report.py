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

"""Safe/dangerous subset comparison with 4-dimension metrics.

Produces two tables (safe set + dangerous set) for three methods:
  1. 原始 DeepSeek S2
  2. DeepSeek S2 + 校准边界 (warn=25, refine=45, block=75, recalibrated, none)
  3. NeMo 官方 baseline

Purely offline — no API, DNS, HTTP, or LLM calls.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent
FILES_ROOT = ROOT.parent.parent
sys.path.insert(0, str(FILES_ROOT / "03_scripts" / "A"))

from threshold_sweep_strict_multiclass import (  # noqa: E402
    collect_domain_details,
    multiclass_metrics,
    simulate_details,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DEEPSEEK_S2_FILE = (
    FILES_ROOT / "05_calibration" / "full223" / "raw" / "deepseek_expert_S2_cached_full_skip_dnsfail_full223.json"
)
NEMO_BASELINE_FILE = FILES_ROOT / "04_static_eval" / "baseline_deepseek_nemo_hallucination_full223.json"

# Calibrated thresholds (DeepSeek S2 optimal)
CALIBRATED = dict(warn=25.0, refine=45.0, block=75.0, score_source="recalibrated", expert_policy="none")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
SAFE_LABELS = {"pass", "warn"}
DANGEROUS_LABELS = {"block", "refine"}


def normalize(v: Any) -> str:
    s = str(v or "pass").strip().lower()
    return s if s in {"pass", "warn", "refine", "block"} else "pass"


def split_subsets(details: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    return {
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


def collect_nemo_details(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    details = []
    for result in data.get("results", []):
        for key in ("baseline", "domain_hallucination"):
            method = result.get(key) or {}
            items = [
                item
                for item in (method.get("details") or [])
                if isinstance(item, dict) and "expected_decision" in item and "predicted" in item
            ]
            if items:
                details.extend(items)
                break
    return details


def pct(v: float) -> str:
    return f"{v * 100:.2f}%"


def print_table(title: str, rows: List[Dict[str, Any]], subset_key: str) -> None:
    print(f"\n{title}")
    header = f"  {'方法':<36} {'严格准确率':>10} {'均衡准确率':>10} {'Macro F1':>10} {'Block 召回':>10}"
    print(header)
    print("  " + "-" * 78)
    for row in rows:
        m = row[subset_key]
        print(
            f"  {row['label']:<36} "
            f"{pct(m['strict_accuracy']):>10} "
            f"{pct(m['balanced_accuracy']):>10} "
            f"{pct(m['macro_f1']):>10} "
            f"{pct(m['block_recall']):>10}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    # Load DeepSeek S2 raw results
    raw_details = collect_domain_details(json.loads(DEEPSEEK_S2_FILE.read_text(encoding="utf-8")))

    # Apply calibrated thresholds
    cal_details = simulate_details(
        raw_details,
        warn=CALIBRATED["warn"],
        refine=CALIBRATED["refine"],
        block=CALIBRATED["block"],
        score_source=CALIBRATED["score_source"],
        expert_policy=CALIBRATED["expert_policy"],
    )

    # Load NeMo baseline
    nemo_details = collect_nemo_details(NEMO_BASELINE_FILE)

    # Build rows
    rows = []
    for label, details in [
        ("原始 DeepSeek S2", raw_details),
        ("DeepSeek S2 + 校准边界", cal_details),
        ("NeMo 官方 baseline", nemo_details),
    ]:
        subsets = split_subsets(details)
        rows.append(
            {
                "label": label,
                "safe": extract_4d(multiclass_metrics(subsets["safe"])),
                "dangerous": extract_4d(multiclass_metrics(subsets["dangerous"])),
                "n_safe": len(subsets["safe"]),
                "n_dangerous": len(subsets["dangerous"]),
            }
        )

    n_safe = rows[0]["n_safe"]
    n_dangerous = rows[0]["n_dangerous"]

    print_table(f"② 安全集验证（{n_safe}条低风险样本，expected∈{{pass,warn}}）", rows, "safe")
    print_table(f"③ 危险集验证（{n_dangerous}条高风险样本，expected∈{{block,refine}}）", rows, "dangerous")

    # Save JSON
    out = FILES_ROOT / "05_calibration" / "safe_danger" / "analysis" / "safe_danger_subset_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved to {out.relative_to(FILES_ROOT)}")


if __name__ == "__main__":
    main()
