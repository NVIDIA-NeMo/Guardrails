#!/usr/bin/env python3
"""Offline threshold sweep for strict four-class domain decisions.

This script does not call DNS, HTTP, GitHub, WHOIS, or any LLM. It reads saved
experiment details, re-maps each case's saved score to pass/warn/refine/block
under different thresholds, and writes the best-scoring simulated result.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


ROOT = Path(__file__).resolve().parent
LABELS = ["pass", "warn", "refine", "block"]


def normalize_label(value: Any) -> str:
    label = str(value or "pass").strip().lower()
    return label if label in LABELS else "pass"


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def f1_score(precision: float, recall: float) -> float:
    return safe_div(2 * precision * recall, precision + recall)


def score_to_level(score: float) -> str:
    if score >= 80:
        return "L4"
    if score >= 60:
        return "L3"
    if score >= 40:
        return "L2"
    if score >= 20:
        return "L1"
    return "L0"


def final_score(detail: Dict[str, Any], score_source: str) -> float:
    raw = detail.get("raw") or {}
    risk = raw.get("risk_score") or {}
    recal = raw.get("recalibrated_score") or {}
    if score_source == "risk":
        return float(risk.get("score") or 0.0)
    if score_source == "max":
        return max(float(risk.get("score") or 0.0), float(recal.get("recalibrated_score") or 0.0))
    return float(recal.get("recalibrated_score") or risk.get("score") or 0.0)


def threshold_decision(score: float, warn: float, refine: float, block: float) -> str:
    if score >= block:
        return "block"
    if score >= refine:
        return "refine"
    if score >= warn:
        return "warn"
    return "pass"


def apply_current_expert_policy(action: str, expert_review: Dict[str, Any] | None) -> str:
    if not isinstance(expert_review, dict) or expert_review.get("status") != "success":
        return action

    risk = str(expert_review.get("semantic_risk") or "unknown").lower()
    relevance = str(expert_review.get("domain_relevance") or "unknown").lower()
    misleading = expert_review.get("is_answer_misleading") is True

    if risk == "critical":
        return "block"
    if risk == "high" and misleading:
        return "refine"
    if relevance == "mismatch" and action in {"warn", "refine", "block"}:
        return "refine"
    return action


def apply_preserve_block_expert_policy(action: str, expert_review: Dict[str, Any] | None) -> str:
    """A stricter alternative: expert high risk does not downgrade block."""
    if not isinstance(expert_review, dict) or expert_review.get("status") != "success":
        return action

    risk = str(expert_review.get("semantic_risk") or "unknown").lower()
    relevance = str(expert_review.get("domain_relevance") or "unknown").lower()
    misleading = expert_review.get("is_answer_misleading") is True

    if risk == "critical":
        return "block"
    if risk == "high" and misleading:
        return "block" if action == "block" else "refine"
    if relevance == "mismatch" and action in {"warn", "refine"}:
        return "refine"
    return action


def remap_prediction(
    detail: Dict[str, Any],
    *,
    warn: float,
    refine: float,
    block: float,
    score_source: str,
    expert_policy: str,
) -> str:
    score = final_score(detail, score_source)
    action = threshold_decision(score, warn=warn, refine=refine, block=block)
    expert_review = (detail.get("raw") or {}).get("expert_review")
    if expert_policy == "current":
        return apply_current_expert_policy(action, expert_review)
    if expert_policy == "preserve_block":
        return apply_preserve_block_expert_policy(action, expert_review)
    return action


def confusion_matrix(details: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    matrix = {label: {pred: 0 for pred in LABELS} for label in LABELS}
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
    matrix = confusion_matrix(details)

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

    macro_precision = sum(item["precision"] for item in per_class.values()) / len(LABELS)
    macro_recall = sum(item["recall"] for item in per_class.values()) / len(LABELS)
    macro_f1 = sum(item["f1"] for item in per_class.values()) / len(LABELS)

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


def simulate_details(
    details: List[Dict[str, Any]],
    *,
    warn: float,
    refine: float,
    block: float,
    score_source: str,
    expert_policy: str,
) -> List[Dict[str, Any]]:
    remapped = []
    for item in details:
        updated = copy.deepcopy(item)
        score = final_score(item, score_source)
        predicted = remap_prediction(
            item,
            warn=warn,
            refine=refine,
            block=block,
            score_source=score_source,
            expert_policy=expert_policy,
        )
        updated["predicted_before_threshold_sweep"] = item.get("predicted")
        updated["predicted"] = predicted
        updated["correct"] = normalize_label(updated.get("expected_decision")) == predicted
        updated["threshold_sweep"] = {
            "score": score,
            "score_level": score_to_level(score),
            "score_source": score_source,
            "warn_threshold": warn,
            "refine_threshold": refine,
            "block_threshold": block,
            "expert_policy": expert_policy,
        }
        remapped.append(updated)
    return remapped


def collect_domain_details(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    details = []
    for result in data.get("results", []):
        method = result.get("domain_hallucination") or {}
        details.extend(item for item in method.get("details", []) if isinstance(item, dict))
    return details


def replace_domain_details(data: Dict[str, Any], details: List[Dict[str, Any]], metrics: Dict[str, Any]) -> Dict[str, Any]:
    output = copy.deepcopy(data)
    offset = 0
    for result in output.get("results", []):
        method = result.get("domain_hallucination") or {}
        count = len(method.get("details", []) or [])
        method["details"] = details[offset : offset + count]
        method["strict_multiclass_threshold_sweep_metrics"] = metrics
        method["metrics"] = {
            **(method.get("metrics") or {}),
            "threshold_sweep_note": "Strict four-class predictions were remapped offline from saved scores.",
            "strict_accuracy": metrics["strict_accuracy"],
            "strict_balanced_accuracy": metrics["balanced_accuracy"],
            "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"],
            "macro_f1": metrics["macro_f1"],
        }
        offset += count
    output["threshold_sweep_created_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    return output


def sweep(
    details: List[Dict[str, Any]],
    *,
    score_sources: List[str],
    expert_policies: List[str],
    warn_values: Iterable[int],
    refine_values: Iterable[int],
    block_values: Iterable[int],
) -> List[Dict[str, Any]]:
    rows = []
    for score_source in score_sources:
        for expert_policy in expert_policies:
            for warn in warn_values:
                for refine in refine_values:
                    if refine <= warn:
                        continue
                    for block in block_values:
                        if block <= refine:
                            continue
                        simulated = simulate_details(
                            details,
                            warn=float(warn),
                            refine=float(refine),
                            block=float(block),
                            score_source=score_source,
                            expert_policy=expert_policy,
                        )
                        metrics = multiclass_metrics(simulated)
                        rows.append(
                            {
                                "warn_threshold": warn,
                                "refine_threshold": refine,
                                "block_threshold": block,
                                "score_source": score_source,
                                "expert_policy": expert_policy,
                                **metrics,
                            }
                        )
    rows.sort(
        key=lambda item: (
            item["macro_f1"],
            item["balanced_accuracy"],
            item["strict_accuracy"],
            item["per_class"]["block"]["recall"],
        ),
        reverse=True,
    )
    return rows


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def print_top(rows: List[Dict[str, Any]], limit: int) -> None:
    print("| rank | score | expert_policy | warn | refine | block | strict_acc | bal_acc | macro_p | macro_r | macro_f1 | block_recall |")
    print("|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for index, row in enumerate(rows[:limit], start=1):
        print(
            "| "
            + " | ".join(
                [
                    str(index),
                    row["score_source"],
                    row["expert_policy"],
                    str(row["warn_threshold"]),
                    str(row["refine_threshold"]),
                    str(row["block_threshold"]),
                    pct(row["strict_accuracy"]),
                    pct(row["balanced_accuracy"]),
                    pct(row["macro_precision"]),
                    pct(row["macro_recall"]),
                    pct(row["macro_f1"]),
                    pct(row["per_class"]["block"]["recall"]),
                ]
            )
            + " |"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline threshold sweep for strict multiclass decisions.")
    parser.add_argument("--input", default="deepseek_expert_S2_cached_full_skip_dnsfail_full223.json")
    parser.add_argument("--output", default="threshold_sweep_best_deepseek_s2_full223.json")
    parser.add_argument("--summary-output", default="threshold_sweep_summary_deepseek_s2_full223.json")
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument("--score-sources", nargs="+", default=["recalibrated", "risk", "max"], choices=["recalibrated", "risk", "max"])
    parser.add_argument("--expert-policies", nargs="+", default=["current", "preserve_block", "none"], choices=["current", "preserve_block", "none"])
    parser.add_argument("--warn-min", type=int, default=0)
    parser.add_argument("--warn-max", type=int, default=50)
    parser.add_argument("--refine-min", type=int, default=5)
    parser.add_argument("--refine-max", type=int, default=80)
    parser.add_argument("--block-min", type=int, default=10)
    parser.add_argument("--block-max", type=int, default=100)
    parser.add_argument("--step", type=int, default=5)
    args = parser.parse_args()

    source_path = ROOT / args.input
    data = json.loads(source_path.read_text(encoding="utf-8"))
    details = collect_domain_details(data)

    rows = sweep(
        details,
        score_sources=args.score_sources,
        expert_policies=args.expert_policies,
        warn_values=range(args.warn_min, args.warn_max + 1, args.step),
        refine_values=range(args.refine_min, args.refine_max + 1, args.step),
        block_values=range(args.block_min, args.block_max + 1, args.step),
    )
    if not rows:
        raise SystemExit("No threshold combinations evaluated.")

    best = rows[0]
    best_details = simulate_details(
        details,
        warn=float(best["warn_threshold"]),
        refine=float(best["refine_threshold"]),
        block=float(best["block_threshold"]),
        score_source=best["score_source"],
        expert_policy=best["expert_policy"],
    )
    best_metrics = multiclass_metrics(best_details)
    output_data = replace_domain_details(data, best_details, best_metrics)
    output_data["threshold_sweep_best"] = {
        "source_file": source_path.name,
        "score_source": best["score_source"],
        "expert_policy": best["expert_policy"],
        "warn_threshold": best["warn_threshold"],
        "refine_threshold": best["refine_threshold"],
        "block_threshold": best["block_threshold"],
        "selection_metric": "macro_f1",
        "metrics": best_metrics,
    }

    output_path = ROOT / args.output
    output_path.write_text(json.dumps(output_data, indent=2, ensure_ascii=False), encoding="utf-8")

    summary_path = ROOT / args.summary_output
    summary = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "input": source_path.name,
        "evaluated_combinations": len(rows),
        "top": rows[: args.top],
        "best": output_data["threshold_sweep_best"],
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print_top(rows, args.top)
    print(f"\nBest output saved to {output_path}")
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()

