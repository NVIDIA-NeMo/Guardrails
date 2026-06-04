#!/usr/bin/env python3
"""Summarize optimization trial JSON files for Experiment A."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


ROOT = Path(__file__).resolve().parent


def pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:.2f}%"
    except Exception:
        return "n/a"


def ms_to_s(value: Any) -> str:
    try:
        return f"{float(value) / 1000:.2f}s"
    except Exception:
        return "n/a"


def load_rows(paths: Iterable[Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        system = data.get("system_under_test", {})
        for result in data.get("results", []):
            ours = result.get("domain_hallucination", {})
            metrics = ours.get("metrics", {})
            latency = metrics.get("latency_ms", {})
            confusion = metrics.get("confusion", {})
            rows.append(
                {
                    "file": path.name,
                    "dataset": result.get("dataset", ""),
                    "verification": system.get("verification_level", ""),
                    "skip_dnsfail": system.get("skip_secondary_checks_on_dns_failure", False),
                    "tls": system.get("enable_tls_verification", True),
                    "whois": system.get("enable_whois_verification", True),
                    "baseline_skipped": system.get("baseline_skipped", False),
                    "accuracy": metrics.get("accuracy"),
                    "recall": metrics.get("url_hallucination_recall"),
                    "fpr": metrics.get("false_positive_rate"),
                    "precision": metrics.get("precision"),
                    "f1": metrics.get("f1"),
                    "mean_ms": latency.get("mean"),
                    "p95_ms": latency.get("p95"),
                    "max_ms": latency.get("max"),
                    "tp": confusion.get("tp"),
                    "fp": confusion.get("fp"),
                    "fn": confusion.get("fn"),
                    "tn": confusion.get("tn"),
                }
            )
    return rows


def print_markdown(rows: List[Dict[str, Any]]) -> None:
    headers = [
        "file",
        "dataset",
        "verification",
        "skip_dnsfail",
        "tls",
        "whois",
        "accuracy",
        "recall",
        "fpr",
        "precision",
        "f1",
        "mean",
        "p95",
        "confusion",
    ]
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        values = [
            row["file"],
            row["dataset"],
            str(row["verification"]),
            str(row["skip_dnsfail"]),
            str(row["tls"]),
            str(row["whois"]),
            pct(row["accuracy"]),
            pct(row["recall"]),
            pct(row["fpr"]),
            pct(row["precision"]),
            pct(row["f1"]),
            ms_to_s(row["mean_ms"]),
            ms_to_s(row["p95_ms"]),
            f"{row['tp']}/{row['fp']}/{row['fn']}/{row['tn']}",
        ]
        print("| " + " | ".join(values) + " |")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize optimization JSON results.")
    parser.add_argument("files", nargs="*", help="Result JSON files under files/.")
    args = parser.parse_args()

    paths = [ROOT / name for name in args.files]
    if not paths:
        paths = sorted(ROOT.glob("opt_S*_eval58.json"))
    rows = load_rows(paths)
    print_markdown(rows)


if __name__ == "__main__":
    main()

