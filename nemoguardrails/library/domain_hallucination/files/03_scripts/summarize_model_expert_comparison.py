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

"""Build a model-by-strategy comparison table and chart for expert sweeps."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT.parent / "05_calibration" / "eval58" / "raw"


FILES = {
    "DeepSeek": {
        "S1": "expert_S1_cached_full_eval58.json",
        "S2": "expert_S2_cached_full_skip_dnsfail_eval58_resume.json",
        "S3": "expert_S3_cached_full_skip_dnsfail_no_whois_eval58_resume.json",
        "S4": "expert_S4_http_skip_dnsfail_eval58_resume.json",
    },
    "Qwen": {
        "S1": "qwen_expert_S1_cached_full_eval58_resume.json",
        "S2": "qwen_expert_S2_cached_full_skip_dnsfail_eval58_resume.json",
        "S3": "qwen_expert_S3_cached_full_skip_dnsfail_no_whois_eval58_resume.json",
        "S4": "qwen_expert_S4_http_skip_dnsfail_eval58_resume.json",
    },
    "GLM": {
        "S1": "glm_expert_S1_cached_full_eval58.json",
        "S2": "glm_expert_S2_cached_full_skip_dnsfail_eval58.json",
        "S3": "glm_expert_S3_cached_full_skip_dnsfail_no_whois_eval58.json",
        "S4": "glm_expert_S4_http_skip_dnsfail_eval58.json",
    },
}


def pct(value: Any) -> str:
    return f"{float(value) * 100:.2f}%"


def seconds(value: Any) -> str:
    return f"{float(value) / 1000:.2f}s"


def read_row(model: str, strategy: str, filename: str) -> Dict[str, Any] | None:
    path = DATA_ROOT / filename
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    result = data["results"][0]
    metrics = result["domain_hallucination"]["metrics"]
    latency = metrics["latency_ms"]
    details = result["domain_hallucination"]["details"]
    expert_calls = 0
    for item in details:
        review = (item.get("raw") or {}).get("expert_review")
        if isinstance(review, dict) and review.get("enabled") is True:
            expert_calls += 1
    return {
        "model": model,
        "strategy": strategy,
        "file": filename,
        "accuracy": metrics["accuracy"],
        "recall": metrics["url_hallucination_recall"],
        "fpr": metrics["false_positive_rate"],
        "precision": metrics["precision"],
        "f1": metrics["f1"],
        "mean_ms": latency["mean"],
        "p95_ms": latency["p95"],
        "expert_calls": expert_calls,
    }


def collect_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for model, strategies in FILES.items():
        for strategy, filename in strategies.items():
            row = read_row(model, strategy, filename)
            if row is not None:
                rows.append(row)
    return rows


def print_table(rows: List[Dict[str, Any]]) -> None:
    headers = ["model", "strategy", "accuracy", "recall", "fpr", "precision", "f1", "mean", "p95", "expert_calls"]
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        values = [
            row["model"],
            row["strategy"],
            pct(row["accuracy"]),
            pct(row["recall"]),
            pct(row["fpr"]),
            pct(row["precision"]),
            pct(row["f1"]),
            seconds(row["mean_ms"]),
            seconds(row["p95_ms"]),
            str(row["expert_calls"]),
        ]
        print("| " + " | ".join(values) + " |")


def save_chart(rows: List[Dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Chart skipped: matplotlib unavailable: {exc}")
        return

    labels = [f"{row['model']}-{row['strategy']}" for row in rows]
    f1 = [row["f1"] * 100 for row in rows]
    latency = [row["mean_ms"] / 1000 for row in rows]

    fig, ax1 = plt.subplots(figsize=(14, 6))
    xs = list(range(len(rows)))
    bars = ax1.bar(xs, f1, color="#2f6fbb", alpha=0.82, label="F1 (%)")
    ax1.set_ylabel("F1 (%)")
    ax1.set_ylim(0, max(f1 + [1]) * 1.25)
    ax1.set_xticks(xs)
    ax1.set_xticklabels(labels, rotation=35, ha="right")

    ax2 = ax1.twinx()
    ax2.plot(xs, latency, color="#c43c35", marker="o", linewidth=2, label="Mean latency (s)")
    ax2.set_ylabel("Mean latency (s)")
    ax2.set_ylim(0, max(latency + [1]) * 1.25)

    for bar, value in zip(bars, f1):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.suptitle("Expert Model Comparison on Experiment A eval_dataset.json")
    fig.tight_layout()
    output = ROOT.parent / "05_calibration" / "eval58" / "analysis" / "model_expert_comparison_A_eval58.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    print(f"Saved chart to {output}")


def main() -> None:
    rows = collect_rows()
    print_table(rows)
    save_chart(rows)


if __name__ == "__main__":
    main()
