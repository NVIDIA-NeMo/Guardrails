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

"""Summarize full_dataset.json expert model sweeps."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT.parent / "05_calibration" / "full223" / "raw"
MODELS = ["deepseek", "qwen", "glm_air", "openrouter_gpt41mini"]
STRATEGIES = {
    "S1": "cached_full",
    "S2": "cached_full_skip_dnsfail",
    "S3": "cached_full_skip_dnsfail_no_whois",
    "S4": "http_skip_dnsfail",
}
OVERRIDES = {
    ("glm_air", "S3"): "glm_air_expert_S3_cached_full_skip_dnsfail_no_whois_full223_rerun.json",
    ("glm_air", "S4"): "glm_air_expert_S4_http_skip_dnsfail_full223_rerun.json",
}


def pct(value: Any) -> str:
    return f"{float(value) * 100:.2f}%"


def seconds(value: Any) -> str:
    return f"{float(value) / 1000:.2f}s"


def read_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for model in MODELS:
        for strategy, suffix in STRATEGIES.items():
            filename = OVERRIDES.get((model, strategy), f"{model}_expert_{strategy}_{suffix}_full223.json")
            path = DATA_ROOT / filename
            if not path.exists():
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            result = data["results"][0]
            metrics = result["domain_hallucination"]["metrics"]
            latency = metrics["latency_ms"]
            details = result["domain_hallucination"]["details"]
            calls = 0
            for detail in details:
                review = (detail.get("raw") or {}).get("expert_review")
                if isinstance(review, dict) and review.get("enabled") is True:
                    calls += 1
            rows.append(
                {
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
                    "expert_calls": calls,
                    "evaluated": metrics["evaluated_samples"],
                }
            )
    return rows


def main() -> None:
    rows = read_rows()
    headers = [
        "model",
        "strategy",
        "evaluated",
        "accuracy",
        "recall",
        "fpr",
        "precision",
        "f1",
        "mean",
        "p95",
        "expert_calls",
    ]
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        values = [
            row["model"],
            row["strategy"],
            str(row["evaluated"]),
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


if __name__ == "__main__":
    main()
