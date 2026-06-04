#!/usr/bin/env python3
"""Per-model threshold sweep (Method A).

For each model × strategy file, run an independent threshold search on the
full 223-case dataset, then collect the best result per model for fair
cross-model comparison.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List

# Reuse all logic from threshold_sweep_strict_multiclass
from threshold_sweep_strict_multiclass import (
    collect_domain_details,
    sweep,
    simulate_details,
    multiclass_metrics,
    replace_domain_details,
)

ROOT = Path(__file__).resolve().parent

# All raw result files per model × strategy
MODEL_FILES = {
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

SWEEP_CONFIG = dict(
    score_sources=["recalibrated", "risk", "max"],
    expert_policies=["current", "preserve_block", "none"],
    warn_values=range(0, 51, 5),
    refine_values=range(5, 81, 5),
    block_values=range(10, 101, 5),
)


def run_sweep_for_file(filepath: Path) -> Dict[str, Any]:
    """Run full threshold sweep on one result file, return best row + best metrics."""
    data = json.loads(filepath.read_text(encoding="utf-8"))
    details = collect_domain_details(data)
    if not details:
        return {"error": "no details found"}

    rows = sweep(details, **SWEEP_CONFIG)
    if not rows:
        return {"error": "no sweep rows"}

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

    return {
        "file": filepath.name,
        "n_details": len(details),
        "n_combinations": len(rows),
        "best_threshold": {
            "warn": best["warn_threshold"],
            "refine": best["refine_threshold"],
            "block": best["block_threshold"],
            "score_source": best["score_source"],
            "expert_policy": best["expert_policy"],
        },
        "metrics": {
            "strict_accuracy": best_metrics["strict_accuracy"],
            "balanced_accuracy": best_metrics["balanced_accuracy"],
            "macro_f1": best_metrics["macro_f1"],
            "macro_precision": best_metrics["macro_precision"],
            "macro_recall": best_metrics["macro_recall"],
            "block_recall": best_metrics["per_class"]["block"]["recall"],
            "block_precision": best_metrics["per_class"]["block"]["precision"],
            "block_f1": best_metrics["per_class"]["block"]["f1"],
            "pass_recall": best_metrics["per_class"]["pass"]["recall"],
            "per_class": best_metrics["per_class"],
            "confusion_matrix": best_metrics["confusion_matrix"],
        },
        "top5": rows[:5],
    }


def main() -> None:
    all_results: Dict[str, Any] = {}
    best_per_model: Dict[str, Any] = {}

    for model, strategies in MODEL_FILES.items():
        print(f"\n{'='*60}")
        print(f"Model: {model}")
        model_results = {}

        for strategy, filename in strategies.items():
            filepath = ROOT / filename
            if not filepath.exists():
                print(f"  [{strategy}] MISSING: {filename}")
                continue

            print(f"  [{strategy}] Sweeping {filename} ...", end=" ", flush=True)
            t0 = time.time()
            result = run_sweep_for_file(filepath)
            elapsed = time.time() - t0

            if "error" in result:
                print(f"ERROR: {result['error']}")
            else:
                m = result["metrics"]
                thr = result["best_threshold"]
                print(
                    f"macro_f1={m['macro_f1']:.4f}  "
                    f"bal_acc={m['balanced_accuracy']:.4f}  "
                    f"block_recall={m['block_recall']:.4f}  "
                    f"[{thr['warn']}/{thr['refine']}/{thr['block']} {thr['score_source']} {thr['expert_policy']}]  "
                    f"({elapsed:.1f}s)"
                )

            model_results[strategy] = result

        all_results[model] = model_results

        # Pick the best strategy for this model (by macro_f1)
        valid = {s: r for s, r in model_results.items() if "error" not in r}
        if valid:
            best_strategy = max(valid, key=lambda s: valid[s]["metrics"]["macro_f1"])
            best_per_model[model] = {
                "best_strategy": best_strategy,
                **valid[best_strategy],
            }

    # Print cross-model comparison table
    print(f"\n{'='*60}")
    print("PER-MODEL BEST THRESHOLD COMPARISON")
    print(f"{'='*60}")
    print(
        f"{'Model':<14} {'Strategy':<6} {'macro_f1':>10} {'bal_acc':>9} "
        f"{'strict_acc':>11} {'block_recall':>13} {'warn/refine/block':>18} {'score':>12} {'policy':>14}"
    )
    print("-" * 120)

    for model in MODEL_FILES:
        if model not in best_per_model:
            print(f"{model:<14} {'N/A':<6}")
            continue
        b = best_per_model[model]
        m = b["metrics"]
        thr = b["best_threshold"]
        print(
            f"{model:<14} {b['best_strategy']:<6} "
            f"{m['macro_f1']:>10.4f} {m['balanced_accuracy']:>9.4f} "
            f"{m['strict_accuracy']:>11.4f} {m['block_recall']:>13.4f} "
            f"{thr['warn']:>5}/{thr['refine']:>5}/{thr['block']:>5}  "
            f"{thr['score_source']:>12} {thr['expert_policy']:>14}"
        )

    # Save full results
    output = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "method": "A_per_model_own_threshold",
        "description": "Each model searched its own optimal threshold on full223. Comparison is fair.",
        "sweep_config": {
            "score_sources": SWEEP_CONFIG["score_sources"],
            "expert_policies": SWEEP_CONFIG["expert_policies"],
            "warn_range": "0-50 step 5",
            "refine_range": "5-80 step 5",
            "block_range": "10-100 step 5",
        },
        "best_per_model": best_per_model,
        "all_results": all_results,
    }

    out_path = ROOT / "per_model_threshold_sweep_results.json"
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nFull results saved to {out_path.name}")

    # Also save a clean summary
    summary = {
        "created_at": output["created_at"],
        "method": "A_per_model_own_threshold",
        "comparison": {
            model: {
                "best_strategy": d["best_strategy"],
                "best_threshold": d["best_threshold"],
                "metrics": d["metrics"],
            }
            for model, d in best_per_model.items()
        },
    }
    summary_path = ROOT / "per_model_threshold_sweep_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Summary saved to {summary_path.name}")


if __name__ == "__main__":
    main()
