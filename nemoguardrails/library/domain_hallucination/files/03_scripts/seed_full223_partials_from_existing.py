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

"""Seed full_dataset partial files from existing eval/full-compatible results.

This lets full_dataset.json runs reuse case-level results that were already
computed for the same model and strategy on eval_dataset.json.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

ROOT = Path(__file__).resolve().parent
FILES_ROOT = ROOT.parent
EVAL_RAW_ROOT = FILES_ROOT / "05_calibration" / "eval58" / "raw"
FULL_PARTIAL_ROOT = FILES_ROOT / "05_calibration" / "full223" / "partials"


EXPERIMENT_ID = "A"
EXPERIMENT_NAME = "nemo_hallucination_vs_domain"
DATASET_NAME = "full_dataset.json"
METHOD = "ours"


SEEDS = {
    "deepseek_expert_S1_cached_full_full223.json": [
        "expert_S1_cached_full_eval58.json",
    ],
    "deepseek_expert_S2_cached_full_skip_dnsfail_full223.json": [
        "expert_S2_cached_full_skip_dnsfail_eval58_resume.json",
    ],
    "deepseek_expert_S3_cached_full_skip_dnsfail_no_whois_full223.json": [
        "expert_S3_cached_full_skip_dnsfail_no_whois_eval58_resume.json",
    ],
    "deepseek_expert_S4_http_skip_dnsfail_full223.json": [
        "expert_S4_http_skip_dnsfail_eval58_resume.json",
    ],
    "qwen_expert_S1_cached_full_full223.json": [
        "qwen_expert_S1_cached_full_eval58_resume.json",
    ],
    "qwen_expert_S2_cached_full_skip_dnsfail_full223.json": [
        "qwen_expert_S2_cached_full_skip_dnsfail_eval58_resume.json",
    ],
    "qwen_expert_S3_cached_full_skip_dnsfail_no_whois_full223.json": [
        "qwen_expert_S3_cached_full_skip_dnsfail_no_whois_eval58_resume.json",
    ],
    "qwen_expert_S4_http_skip_dnsfail_full223.json": [
        "qwen_expert_S4_http_skip_dnsfail_eval58_resume.json",
    ],
    "openrouter_gpt41mini_expert_S1_cached_full_full223.json": [
        "openrouter_gpt41mini_expert_S1_cached_full_eval58.json.partial.json",
    ],
}


def load_full_cases() -> Dict[str, Dict[str, Any]]:
    data = json.loads((FILES_ROOT / "01_datasets" / "full_dataset.json").read_text(encoding="utf-8"))
    return {str(case.get("id") or ""): case for case in data.get("test_cases", [])}


def events_from_result(path: Path, full_cases: Dict[str, Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "events" in data:
        for event in data.get("events", []):
            if event.get("method") != METHOD or event.get("id") not in full_cases:
                continue
            seeded = dict(event)
            seeded["dataset"] = DATASET_NAME
            seeded["recorded_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            seeded["seeded_from"] = path.name
            yield seeded
        return

    for result in data.get("results", []):
        details = (result.get("domain_hallucination") or {}).get("details", [])
        for detail in details:
            case_id = str(detail.get("id") or "")
            case = full_cases.get(case_id)
            if case is None:
                continue
            yield {
                "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "experiment_id": EXPERIMENT_ID,
                "experiment_name": EXPERIMENT_NAME,
                "dataset": DATASET_NAME,
                "method": METHOD,
                "id": case_id,
                "category": detail.get("category") or case.get("category"),
                "expected_decision": detail.get("expected_decision") or case.get("expected_decision"),
                "predicted": detail.get("predicted"),
                "latency_ms": detail.get("latency_ms"),
                "mode": detail.get("mode"),
                "skipped": False,
                "error": None,
                "raw": detail.get("raw") if isinstance(detail.get("raw"), dict) else {},
                "seeded_from": path.name,
            }


def merge_events(existing: List[Dict[str, Any]], new_events: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: Dict[tuple[str, str, str, str, str], Dict[str, Any]] = {}
    order: List[tuple[str, str, str, str, str]] = []
    for event in list(existing) + list(new_events):
        key = (
            str(event.get("experiment_id") or ""),
            str(event.get("experiment_name") or ""),
            str(event.get("dataset") or ""),
            str(event.get("method") or ""),
            str(event.get("id") or ""),
        )
        if key not in merged:
            order.append(key)
        merged[key] = event
    return [merged[key] for key in order]


def main() -> None:
    full_cases = load_full_cases()
    for target, sources in SEEDS.items():
        partial_path = FULL_PARTIAL_ROOT / f"{target}.partial.json"
        existing: List[Dict[str, Any]] = []
        started_at = time.strftime("%Y-%m-%dT%H:%M:%S")
        if partial_path.exists():
            try:
                payload = json.loads(partial_path.read_text(encoding="utf-8"))
                existing = list(payload.get("events", []))
                started_at = str(payload.get("started_at") or started_at)
            except Exception:
                existing = []

        seeded: List[Dict[str, Any]] = []
        for source in sources:
            source_path = EVAL_RAW_ROOT / source
            if source_path.exists():
                seeded.extend(events_from_result(source_path, full_cases))

        events = merge_events(existing, seeded)
        payload = {
            "started_at": started_at,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "repo_root": str(ROOT.parents[4]),
            "args": {
                "seeded_for": target,
                "dataset": DATASET_NAME,
                "sources": sources,
            },
            "events": events,
        }
        partial_path.parent.mkdir(parents=True, exist_ok=True)
        partial_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"{target}: seeded {len(events)} partial events")


if __name__ == "__main__":
    main()
