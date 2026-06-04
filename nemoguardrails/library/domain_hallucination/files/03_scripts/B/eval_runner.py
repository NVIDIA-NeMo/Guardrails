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
Domain Hallucination Guard - Evaluation Runner

Runs the evaluation dataset against your domain_hallucination module
and produces quantitative performance metrics.

Usage:
    # Full evaluation (requires network for DNS/HTTP verification)
    python eval_runner.py --config /path/to/config.json

    # Dry run (no network, tests extraction + scoring logic only)
    python eval_runner.py --dry-run

    # Compare baseline vs your version
    python eval_runner.py --baseline-results baseline.json --output results.json
"""

import argparse
import asyncio
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ─── Metrics Calculation ────────────────────────────────────────────


@dataclass
class MetricsAccumulator:
    """Accumulates predictions and computes classification metrics."""

    y_true: list = field(default_factory=list)
    y_pred: list = field(default_factory=list)
    latencies_ms: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    details: list = field(default_factory=list)

    def add(self, test_id: str, expected: str, predicted: str, latency_ms: float, extra: dict = None):
        self.y_true.append(expected)
        self.y_pred.append(predicted)
        self.latencies_ms.append(latency_ms)
        self.details.append(
            {
                "id": test_id,
                "expected": expected,
                "predicted": predicted,
                "correct": expected == predicted,
                "latency_ms": round(latency_ms, 4),
                **(extra or {}),
            }
        )

    def add_error(self, test_id: str, error: str):
        self.errors.append({"id": test_id, "error": error})

    def compute_metrics(self) -> dict:
        if not self.y_true:
            return {"error": "No predictions collected"}

        labels = sorted(set(self.y_true) | set(self.y_pred))

        # Overall accuracy
        correct = sum(1 for t, p in zip(self.y_true, self.y_pred) if t == p)
        accuracy = correct / len(self.y_true)

        # Per-label precision, recall, F1
        per_label = {}
        for label in labels:
            tp = sum(1 for t, p in zip(self.y_true, self.y_pred) if t == label and p == label)
            fp = sum(1 for t, p in zip(self.y_true, self.y_pred) if t != label and p == label)
            fn = sum(1 for t, p in zip(self.y_true, self.y_pred) if t == label and p != label)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            support = sum(1 for t in self.y_true if t == label)
            per_label[label] = {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "support": support,
            }

        # Macro & weighted averages
        macro_p = sum(v["precision"] for v in per_label.values()) / len(labels)
        macro_r = sum(v["recall"] for v in per_label.values()) / len(labels)
        macro_f1 = sum(v["f1"] for v in per_label.values()) / len(labels)

        total = len(self.y_true)
        weighted_f1 = sum(v["f1"] * v["support"] / total for v in per_label.values())

        # Binary: hallucination detected vs not
        # Map: block/refine → "flagged", warn/pass → "not_flagged"
        def to_binary(label):
            return "flagged" if label in ("block", "refine") else "not_flagged"

        bin_true = [to_binary(t) for t in self.y_true]
        bin_pred = [to_binary(p) for p in self.y_pred]
        bin_tp = sum(1 for t, p in zip(bin_true, bin_pred) if t == "flagged" and p == "flagged")
        bin_fp = sum(1 for t, p in zip(bin_true, bin_pred) if t == "not_flagged" and p == "flagged")
        bin_fn = sum(1 for t, p in zip(bin_true, bin_pred) if t == "flagged" and p == "not_flagged")
        bin_tn = sum(1 for t, p in zip(bin_true, bin_pred) if t == "not_flagged" and p == "not_flagged")

        bin_precision = bin_tp / (bin_tp + bin_fp) if (bin_tp + bin_fp) > 0 else 0.0
        bin_recall = bin_tp / (bin_tp + bin_fn) if (bin_tp + bin_fn) > 0 else 0.0
        bin_f1 = (
            2 * bin_precision * bin_recall / (bin_precision + bin_recall) if (bin_precision + bin_recall) > 0 else 0.0
        )

        # Latency stats
        sorted_lat = sorted(self.latencies_ms)
        p50 = sorted_lat[len(sorted_lat) // 2] if sorted_lat else 0
        p95_idx = min(int(len(sorted_lat) * 0.95), len(sorted_lat) - 1)
        p95 = sorted_lat[p95_idx] if sorted_lat else 0
        p99_idx = min(int(len(sorted_lat) * 0.99), len(sorted_lat) - 1)
        p99 = sorted_lat[p99_idx] if sorted_lat else 0

        # Confusion matrix as dict
        confusion = defaultdict(lambda: defaultdict(int))
        for t, p in zip(self.y_true, self.y_pred):
            confusion[t][p] += 1

        return {
            "total_samples": total,
            "accuracy": round(accuracy, 4),
            "per_decision": per_label,
            "macro_avg": {
                "precision": round(macro_p, 4),
                "recall": round(macro_r, 4),
                "f1": round(macro_f1, 4),
            },
            "weighted_f1": round(weighted_f1, 4),
            "binary_detection": {
                "precision": round(bin_precision, 4),
                "recall": round(bin_recall, 4),
                "f1": round(bin_f1, 4),
                "true_positives": bin_tp,
                "false_positives": bin_fp,
                "false_negatives": bin_fn,
                "true_negatives": bin_tn,
            },
            "latency_ms": {
                "mean": round(sum(self.latencies_ms) / len(self.latencies_ms), 4),
                "p50": round(p50, 4),
                "p95": round(p95, 4),
                "p99": round(p99, 4),
                "min": round(min(self.latencies_ms), 4),
                "max": round(max(self.latencies_ms), 4),
            },
            "confusion_matrix": {k: dict(v) for k, v in confusion.items()},
            "errors": self.errors,
        }


# ─── Evaluation Runner ──────────────────────────────────────────────


class EvalRunner:
    """Runs evaluation dataset against a domain_hallucination adapter."""

    def __init__(self, dataset_path: str, adapter=None, dry_run: bool = False):
        with open(dataset_path, encoding="utf-8") as f:
            data = json.load(f)
        self.metadata = data["metadata"]
        self.test_cases = data["test_cases"]
        self.adapter = adapter
        self.dry_run = dry_run

    async def run_single(self, case: dict) -> dict:
        """Run a single test case and return the result."""
        test_id = case["id"]
        answer = case["llm_answer"]
        query = case["user_query"]
        expected = case["expected_decision"]

        if self.dry_run:
            start = time.perf_counter()
            # Simulate: use heuristic rules for dry-run mode. This must not
            # read expected_decision, otherwise dry-run metrics leak labels.
            predicted = self._heuristic_predict(case)
            latency = (time.perf_counter() - start) * 1000
            extra = {
                "mode": "dry_run",
                "framework_only": True,
                "note": "Heuristic dry-run only; no DNS/HTTP/GitHub verification.",
            }
        else:
            start = time.perf_counter()
            try:
                result = await self.adapter.analyze_answer(answer, user_query=query)
                latency = (time.perf_counter() - start) * 1000
                predicted = result.get("decision", {}).get("action", "error")
                extra = {
                    "mode": "live",
                    "risk_score": result.get("risk_score"),
                    "risk_level": result.get("risk_level"),
                    "issues_found": len(result.get("issues", [])),
                    "entities_extracted": result.get("entities", {}),
                }
            except Exception as e:
                latency = (time.perf_counter() - start) * 1000
                return {"id": test_id, "error": str(e), "latency_ms": latency}

        return {
            "id": test_id,
            "expected": expected,
            "predicted": predicted,
            "latency_ms": latency,
            "extra": extra,
        }

    def _heuristic_predict(self, case: dict) -> str:
        """Simple heuristic for dry-run testing of the eval framework.

        Dry-run exists to exercise dataset loading, result aggregation, and
        report generation. It intentionally does not use expected labels and
        should not be interpreted as detector performance.
        """
        entities = case.get("entities", {})
        urls = entities.get("urls", [])
        domains = entities.get("domains", [])
        repos = entities.get("github_repos", [])

        if not urls and not domains and not repos:
            return "pass"

        # Check for obviously suspicious patterns
        suspicious_tlds = [".xyz", ".info", ".top", ".click", ".tk", ".ml", ".ga"]
        suspicious_words = [
            "free-",
            "secure-",
            "login",
            "password",
            "verify",
            "account",
            "wallet",
            "phishing",
            "malware",
        ]
        suspicious_repo_words = [
            "fake",
            "nonexistent",
            "not-real",
            "nosuch",
            "does-not-exist",
            "hallucinated",
        ]

        suspicious_count = 0
        total_entities = len(urls) + len(domains) + len(repos)

        for d in domains:
            domain = str(d).lower()
            if any(domain.endswith(tld) for tld in suspicious_tlds):
                suspicious_count += 1
            elif any(word in domain for word in suspicious_words):
                suspicious_count += 1

        for url in urls:
            url_text = str(url).lower()
            if any(url_text.endswith(tld) for tld in suspicious_tlds):
                suspicious_count += 1
            elif any(word in url_text for word in suspicious_words):
                suspicious_count += 1

        for repo_item in repos:
            if isinstance(repo_item, dict):
                repo_text = f"{repo_item.get('owner', '')}/{repo_item.get('repo', '')}".lower()
            else:
                repo_text = str(repo_item).lower()
            if any(word in repo_text for word in suspicious_repo_words):
                suspicious_count += 1

        if suspicious_count == 0:
            return "pass"
        if suspicious_count < total_entities:
            return "refine"
        return "block"

    async def run_all(self, categories: list = None) -> dict:
        """Run all test cases and return aggregated metrics."""
        cases = self.test_cases
        if categories:
            cases = [c for c in cases if c["category"] in categories]

        print(f"\n{'=' * 60}")
        print("  Domain Hallucination Guard - Evaluation")
        print(f"  Mode: {'DRY RUN' if self.dry_run else 'LIVE'}")
        print(f"  Test cases: {len(cases)}")
        print(f"{'=' * 60}\n")

        # Overall metrics
        overall = MetricsAccumulator()
        # Per-category metrics
        by_category = defaultdict(MetricsAccumulator)

        for i, case in enumerate(cases):
            result = await self.run_single(case)

            if "error" in result:
                overall.add_error(result["id"], result["error"])
                by_category[case["category"]].add_error(result["id"], result["error"])
                status = "ERR"
            else:
                overall.add(
                    result["id"], result["expected"], result["predicted"], result["latency_ms"], result.get("extra")
                )
                by_category[case["category"]].add(
                    result["id"], result["expected"], result["predicted"], result["latency_ms"], result.get("extra")
                )
                match = "✓" if result["expected"] == result["predicted"] else "✗"
                status = f"{match} exp={result['expected']:6s} got={result['predicted']:6s}"

            progress = f"[{i + 1:3d}/{len(cases)}]"
            print(f"  {progress} {case['id']:20s} {status}")

        print(f"\n{'─' * 60}")

        # Build report
        report = {
            "metadata": self.metadata,
            "run_info": {
                "mode": "dry_run" if self.dry_run else "live",
                "total_cases": len(cases),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
            "overall": overall.compute_metrics(),
            "by_category": {cat: acc.compute_metrics() for cat, acc in sorted(by_category.items())},
            "per_case_details": overall.details,
        }

        return report

    async def run_comparison(self, baseline_path: str) -> dict:
        """Run evaluation and compare against baseline results."""
        with open(baseline_path, encoding="utf-8") as f:
            baseline = json.load(f)

        current = await self.run_all()

        comparison = {
            "baseline": baseline.get("overall", {}),
            "current": current["overall"],
            "improvements": {},
        }

        # Compare key metrics
        for metric in ["accuracy", "weighted_f1"]:
            base_val = baseline.get("overall", {}).get(metric, 0)
            curr_val = current["overall"].get(metric, 0)
            delta = curr_val - base_val
            comparison["improvements"][metric] = {
                "baseline": base_val,
                "current": curr_val,
                "delta": round(delta, 4),
                "relative_pct": (round(delta / base_val * 100, 2) if base_val > 0 else None),
            }

        # Compare binary detection
        for metric in ["precision", "recall", "f1"]:
            base_val = baseline.get("overall", {}).get("binary_detection", {}).get(metric, 0)
            curr_val = current["overall"].get("binary_detection", {}).get(metric, 0)
            delta = curr_val - base_val
            comparison["improvements"][f"binary_{metric}"] = {
                "baseline": base_val,
                "current": curr_val,
                "delta": round(delta, 4),
            }

        # Compare latency
        base_lat = baseline.get("overall", {}).get("latency_ms", {}).get("mean", 0)
        curr_lat = current["overall"].get("latency_ms", {}).get("mean", 0)
        comparison["improvements"]["latency_mean_ms"] = {
            "baseline": base_lat,
            "current": curr_lat,
            "delta": round(curr_lat - base_lat, 2),
        }

        # Per-category comparison
        cat_comparison = {}
        for cat in current.get("by_category", {}):
            base_cat = baseline.get("by_category", {}).get(cat, {})
            curr_cat = current["by_category"][cat]
            cat_comparison[cat] = {
                "accuracy": {
                    "baseline": base_cat.get("accuracy", 0),
                    "current": curr_cat.get("accuracy", 0),
                    "delta": round(curr_cat.get("accuracy", 0) - base_cat.get("accuracy", 0), 4),
                },
            }

        current["comparison"] = comparison
        current["category_comparison"] = cat_comparison
        return current


# ─── Report Printer ─────────────────────────────────────────────────


def print_report(report: dict):
    """Pretty-print evaluation results to terminal."""
    overall = report["overall"]

    print(f"\n{'=' * 60}")
    print("  EVALUATION RESULTS")
    print(f"{'=' * 60}")
    print(f"  Total Samples:  {overall['total_samples']}")
    print(f"  Accuracy:       {overall['accuracy']:.2%}")
    print(f"  Weighted F1:    {overall['weighted_f1']:.4f}")
    print(f"  Macro F1:       {overall['macro_avg']['f1']:.4f}")

    print("\n  ── Binary Detection (flagged vs not) ──")
    bd = overall["binary_detection"]
    print(f"  Precision: {bd['precision']:.4f}   Recall: {bd['recall']:.4f}   F1: {bd['f1']:.4f}")
    print(
        f"  TP={bd['true_positives']}  FP={bd['false_positives']}  "
        f"FN={bd['false_negatives']}  TN={bd['true_negatives']}"
    )

    print("\n  ── Per-Decision Metrics ──")
    print(f"  {'Decision':<10} {'Prec':>8} {'Recall':>8} {'F1':>8} {'Support':>8}")
    for label, m in sorted(overall["per_decision"].items()):
        print(f"  {label:<10} {m['precision']:>8.4f} {m['recall']:>8.4f} {m['f1']:>8.4f} {m['support']:>8d}")

    print("\n  ── Latency (ms) ──")
    lat = overall["latency_ms"]
    print(f"  Mean={lat['mean']:.4f}  P50={lat['p50']:.4f}  P95={lat['p95']:.4f}  P99={lat['p99']:.4f}")

    print("\n  ── Per-Category Accuracy ──")
    for cat, metrics in sorted(report.get("by_category", {}).items()):
        n = metrics.get("total_samples", 0)
        acc = metrics.get("accuracy", 0)
        bar = "█" * int(acc * 20) + "░" * (20 - int(acc * 20))
        print(f"  {cat:<22} {bar} {acc:.1%} (n={n})")

    if report.get("comparison"):
        comp = report["comparison"]
        print(f"\n{'=' * 60}")
        print("  COMPARISON vs BASELINE")
        print(f"{'=' * 60}")
        for metric, vals in comp.get("improvements", {}).items():
            delta = vals["delta"]
            arrow = "↑" if delta > 0 else "↓" if delta < 0 else "="
            color = ""  # Terminal colors could be added here
            print(f"  {metric:<20} {vals['baseline']:.4f} → {vals['current']:.4f}  ({arrow} {abs(delta):.4f})")

    if overall.get("errors"):
        print(f"\n  ── Errors ({len(overall['errors'])}) ──")
        for err in overall["errors"][:5]:
            print(f"  {err['id']}: {err['error'][:60]}")

    print(f"\n{'=' * 60}\n")


# ─── Main ───────────────────────────────────────────────────────────


async def main():
    parser = argparse.ArgumentParser(description="Evaluate Domain Hallucination Guard performance")
    parser.add_argument("--dataset", default="eval_dataset.json", help="Path to evaluation dataset JSON")
    parser.add_argument("--config", default=None, help="Path to domain_hallucination config.json")
    parser.add_argument("--output", default="eval_results.json", help="Path to save results JSON")
    parser.add_argument("--baseline-results", default=None, help="Path to baseline results for comparison")
    parser.add_argument("--dry-run", action="store_true", help="Run without network verification (framework test)")
    parser.add_argument("--categories", nargs="*", default=None, help="Only evaluate specific categories")

    args = parser.parse_args()

    # Try to import your adapter
    adapter = None
    if not args.dry_run:
        try:
            from domain_hallucination_guard_system.nemo_adapter import DomainHallucinationAdapter

            if args.config:
                adapter = DomainHallucinationAdapter(config_path=args.config)
            else:
                adapter = DomainHallucinationAdapter(
                    verification_level="dns",
                    enable_semantic_check=False,
                    enable_advanced_verification=True,
                )
            print("✓ Loaded DomainHallucinationAdapter")
        except ImportError:
            print("⚠ Could not import adapter, falling back to dry-run mode")
            args.dry_run = True

    runner = EvalRunner(
        dataset_path=args.dataset,
        adapter=adapter,
        dry_run=args.dry_run,
    )

    if args.baseline_results:
        report = await runner.run_comparison(args.baseline_results)
    else:
        report = await runner.run_all(categories=args.categories)

    # Print to terminal
    print_report(report)

    # Save to JSON
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
