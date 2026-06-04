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
Run comparison experiments for URL/domain hallucination detection.

The three experiments compare the domain_hallucination detector against
existing NeMo/library hallucination or fact-checking methods:

Experiment A: NeMo library/hallucination
Experiment B: NeMo self_check/facts
Experiment C: factchecking/align_score

Each experiment is evaluated on both eval_dataset.json and
expanded_dataset.json. The domain_hallucination system is run without expert
review; expert review is recorded as a future optional extension.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


ROOT = Path(__file__).resolve().parent
FILES_ROOT = ROOT.parent.parent
REPO_ROOT = ROOT.parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DATASETS = ["eval_dataset.json", "expanded_dataset.json"]
FLAGGED_DECISIONS = {"block", "refine"}


@dataclass
class MethodResult:
    """Single method prediction for one test case."""

    predicted: str
    latency_ms: float
    mode: str
    raw: Dict[str, Any] = field(default_factory=dict)
    skipped: bool = False
    error: Optional[str] = None


@dataclass
class MethodSpec:
    """Metadata for one detector used in a comparison."""

    name: str
    label: str
    module: str
    callable_name: str
    detection_target: str
    requires: List[str]
    notes: str


METHOD_SPECS = {
    "ours": MethodSpec(
        name="ours",
        label="domain_hallucination",
        module="nemoguardrails.library.domain_hallucination.actions",
        callable_name="analyze_answer",
        detection_target="URLs, domains, and GitHub repositories",
        requires=["network for DNS/GitHub checks"],
        notes="Expert review disabled in current experiments; can be enabled later.",
    ),
    "nemo_hallucination": MethodSpec(
        name="nemo_hallucination",
        label="library/hallucination",
        module="nemoguardrails.library.hallucination.actions",
        callable_name="self_check_hallucination",
        detection_target="Answer self-consistency across additional generations",
        requires=["NeMo Rails config", "LLM", "LLMTaskManager"],
        notes="Suitable baseline for whether generic hallucination checks catch fabricated links.",
    ),
    "self_check_facts": MethodSpec(
        name="self_check_facts",
        label="self_check/facts",
        module="nemoguardrails.library.self_check.facts.actions",
        callable_name="self_check_facts",
        detection_target="Whether an answer is grounded in relevant_chunks",
        requires=["relevant_chunks", "LLM", "LLMTaskManager"],
        notes="No-context behavior is measured explicitly because URL truth is often absent from RAG context.",
    ),
    "align_score": MethodSpec(
        name="align_score",
        label="factchecking/align_score",
        module="nemoguardrails.library.factchecking.align_score.actions",
        callable_name="alignscore_check_facts",
        detection_target="Evidence-response factual alignment score",
        requires=["AlignScore endpoint or fallback self-check configuration"],
        notes="Generic fact-check baseline; not designed for DNS/GitHub existence checks.",
    ),
}


EXPERIMENTS = {
    "A": {
        "name": "nemo_hallucination_vs_domain",
        "baseline": "nemo_hallucination",
        "question": "Can NeMo generic hallucination detection catch fabricated URLs/repos?",
    },
    "B": {
        "name": "self_check_facts_vs_domain",
        "baseline": "self_check_facts",
        "question": "Is context-grounded fact checking sufficient for URL hallucination?",
    },
    "C": {
        "name": "align_score_vs_domain",
        "baseline": "align_score",
        "question": "Can a generic factual-alignment score recognize URL/repo fabrication?",
    },
}


class Metrics:
    """Collects binary URL-hallucination metrics."""

    def __init__(self) -> None:
        self.y_true: List[str] = []
        self.y_pred: List[str] = []
        self.latencies: List[float] = []
        self.details: List[Dict[str, Any]] = []
        self.skipped: List[Dict[str, str]] = []
        self.errors: List[Dict[str, str]] = []

    def add(self, case: Dict[str, Any], result: MethodResult) -> None:
        expected = decision_to_binary(case["expected_decision"])
        predicted = decision_to_binary(result.predicted)
        self.y_true.append(expected)
        self.y_pred.append(predicted)
        self.latencies.append(result.latency_ms)
        self.details.append(
            {
                "id": case["id"],
                "category": case.get("category"),
                "expected_decision": case["expected_decision"],
                "expected_binary": expected,
                "predicted": result.predicted,
                "predicted_binary": predicted,
                "correct": expected == predicted,
                "latency_ms": round(result.latency_ms, 4),
                "mode": result.mode,
                "raw": result.raw,
            }
        )

    def add_skip(self, case: Dict[str, Any], reason: str) -> None:
        self.skipped.append({"id": case["id"], "reason": reason})

    def add_error(self, case: Dict[str, Any], error: str) -> None:
        self.errors.append({"id": case["id"], "error": error})

    def compute(self) -> Dict[str, Any]:
        total = len(self.y_true)
        if total == 0:
            return {
                "total_samples": 0,
                "evaluated_samples": 0,
                "skipped_samples": len(self.skipped),
                "errors": self.errors,
                "skipped": self.skipped,
            }

        tp = sum(1 for t, p in zip(self.y_true, self.y_pred) if t == "flagged" and p == "flagged")
        fp = sum(1 for t, p in zip(self.y_true, self.y_pred) if t == "not_flagged" and p == "flagged")
        fn = sum(1 for t, p in zip(self.y_true, self.y_pred) if t == "flagged" and p == "not_flagged")
        tn = sum(1 for t, p in zip(self.y_true, self.y_pred) if t == "not_flagged" and p == "not_flagged")

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        accuracy = (tp + tn) / total

        by_category: Dict[str, Dict[str, int]] = defaultdict(lambda: {"total": 0, "correct": 0})
        for detail in self.details:
            cat = detail["category"] or "unknown"
            by_category[cat]["total"] += 1
            by_category[cat]["correct"] += int(detail["correct"])

        sorted_latencies = sorted(self.latencies)
        p95_index = min(int(len(sorted_latencies) * 0.95), len(sorted_latencies) - 1)

        return {
            "total_samples": total + len(self.skipped) + len(self.errors),
            "evaluated_samples": total,
            "skipped_samples": len(self.skipped),
            "error_samples": len(self.errors),
            "accuracy": round(accuracy, 4),
            "url_hallucination_recall": round(recall, 4),
            "false_positive_rate": round(fp / (fp + tn), 4) if fp + tn else 0.0,
            "precision": round(precision, 4),
            "f1": round(f1, 4),
            "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
            "latency_ms": {
                "mean": round(sum(self.latencies) / len(self.latencies), 4),
                "p95": round(sorted_latencies[p95_index], 4),
                "min": round(min(self.latencies), 4),
                "max": round(max(self.latencies), 4),
            },
            "by_category": {
                cat: {
                    "total": item["total"],
                    "accuracy": round(item["correct"] / item["total"], 4) if item["total"] else 0.0,
                }
                for cat, item in sorted(by_category.items())
            },
            "errors": self.errors,
            "skipped": self.skipped,
        }


class IncrementalRecorder:
    """Persist per-case progress so long experiment runs are recoverable."""

    def __init__(self, path: Path, args: argparse.Namespace) -> None:
        self.path = path
        self.enabled = not getattr(args, "disable_incremental_save", False)
        self.events: List[Dict[str, Any]] = []
        self.started_at = time.strftime("%Y-%m-%dT%H:%M:%S")
        self.args = {
            "experiments": args.experiments,
            "datasets": args.datasets,
            "domain_verification_level": args.domain_verification_level,
            "enable_semantic_check": not args.disable_domain_semantic_check,
            "enable_advanced_verification": not args.disable_domain_advanced_verification,
            "expert_review_enabled": args.enable_domain_expert_review,
            "expert_review_min_level": args.domain_expert_review_min_level
            if args.enable_domain_expert_review
            else None,
            "enable_tls_verification": not args.disable_domain_tls,
            "enable_whois_verification": not args.disable_domain_whois,
        }
        if self.enabled and getattr(args, "resume_from_partial", False) and self.path.exists():
            try:
                existing = json.loads(self.path.read_text(encoding="utf-8"))
                self.events = list(existing.get("events", []))
                self.started_at = str(existing.get("started_at") or self.started_at)
            except Exception:
                self.events = []

    def record(
        self,
        *,
        experiment_id: str,
        experiment_name: str,
        dataset_name: str,
        method: str,
        case: Dict[str, Any],
        result: Optional[MethodResult] = None,
        error: Optional[str] = None,
    ) -> None:
        if not self.enabled:
            return

        event: Dict[str, Any] = {
            "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "experiment_id": experiment_id,
            "experiment_name": experiment_name,
            "dataset": dataset_name,
            "method": method,
            "id": case.get("id"),
            "category": case.get("category"),
            "expected_decision": case.get("expected_decision"),
        }
        if result is not None:
            event.update(
                {
                    "predicted": result.predicted,
                    "latency_ms": round(result.latency_ms, 4),
                    "mode": result.mode,
                    "skipped": result.skipped,
                    "error": result.error,
                    "raw": result.raw,
                }
            )
        if error is not None:
            event["error"] = error

        self.events.append(event)
        self.save()

    def find_event(
        self,
        *,
        experiment_id: str,
        experiment_name: str,
        dataset_name: str,
        method: str,
        case_id: str,
    ) -> Optional[Dict[str, Any]]:
        for event in reversed(self.events):
            if (
                event.get("experiment_id") == experiment_id
                and event.get("experiment_name") == experiment_name
                and event.get("dataset") == dataset_name
                and event.get("method") == method
                and str(event.get("id") or "") == case_id
            ):
                return event
        return None

    def save(self) -> None:
        if not self.enabled:
            return
        payload = {
            "started_at": self.started_at,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "repo_root": str(REPO_ROOT),
            "args": self.args,
            "events": self.events,
        }
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)


class NemoRuntime:
    """Lazy NeMo runtime for LLM-backed baseline actions."""

    def __init__(self, config_path: Path) -> None:
        from nemoguardrails.llm.models.initializer import init_llm_model
        from nemoguardrails.llm.taskmanager import LLMTaskManager
        from nemoguardrails.rails.llm.config import RailsConfig

        self.config_path = config_path
        self.rails_config = RailsConfig.from_path(str(config_path))
        self.llm_task_manager = LLMTaskManager(self.rails_config)

        main_model = None
        for model_config in self.rails_config.models:
            if model_config.type == "main":
                main_model = model_config
                break
        if main_model is None:
            raise ValueError("No main model configured for baseline runtime.")

        kwargs = dict(main_model.parameters or {})
        if main_model.api_key_env_var:
            api_key = os.environ.get(main_model.api_key_env_var)
            if api_key:
                kwargs["api_key"] = api_key

        self.llm = init_llm_model(
            model_name=main_model.model or "",
            provider_name=main_model.engine,
            kwargs=kwargs,
            mode=main_model.mode,
        )


_NEMO_RUNTIME: Optional[NemoRuntime] = None


def get_nemo_runtime(args: argparse.Namespace) -> NemoRuntime:
    global _NEMO_RUNTIME
    if _NEMO_RUNTIME is None:
        config_path = FILES_ROOT / "02_configs" / args.nemo_config
        _NEMO_RUNTIME = NemoRuntime(config_path=config_path)
    return _NEMO_RUNTIME


def decision_to_binary(decision: str) -> str:
    return "flagged" if decision in FLAGGED_DECISIONS else "not_flagged"


def load_dataset(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def case_context(case: Dict[str, Any], include_relevant_chunks: bool) -> Dict[str, Any]:
    context = {
        "user_message": case.get("user_query", ""),
        "bot_message": case.get("llm_answer", ""),
    }
    if include_relevant_chunks:
        chunks = case.get("relevant_chunks") or case.get("evidence") or []
        context["relevant_chunks"] = chunks
        context["relevant_chunks_sep"] = chunks if isinstance(chunks, list) else [str(chunks)]
    return context


async def call_llm_with_retries(
    llm: Any,
    prompt: str,
    max_retries: int,
    llm_params: Optional[Dict[str, Any]] = None,
    stop: Optional[List[str]] = None,
) -> Optional[str]:
    from nemoguardrails.actions.llm.utils import llm_call

    attempts = max(1, max_retries)
    last_error = None
    for attempt in range(attempts):
        try:
            response = await llm_call(llm, prompt, stop=stop, llm_params=llm_params or {})
            return response.content
        except Exception as exc:
            last_error = exc
            if attempt + 1 < attempts:
                await asyncio.sleep(0.5 * (attempt + 1))
    if last_error:
        print(f"LLM retry exhausted: {last_error}")
    return None


async def run_hallucination_baseline_sequential(
    runtime: NemoRuntime,
    user_query: str,
    bot_response: str,
    extra_responses_count: int,
    max_retries: int,
) -> tuple[bool, Dict[str, Any]]:
    from nemoguardrails.actions.llm.utils import get_multiline_response, strip_quotes
    from nemoguardrails.llm.types import Task

    extra_responses: List[str] = []
    for _ in range(max(1, extra_responses_count)):
        raw_extra = await call_llm_with_retries(
            llm=runtime.llm,
            prompt=user_query,
            max_retries=max_retries,
            llm_params={"temperature": 1.0, "max_tokens": 256},
        )
        if raw_extra is not None:
            extra_responses.append(strip_quotes(get_multiline_response(raw_extra)))

    if not extra_responses:
        return False, {
            "agreement": "na",
            "extra_responses_count": 0,
            "extra_responses_requested": extra_responses_count,
            "baseline_error": "No extra LLM responses were generated.",
        }

    prompt = runtime.llm_task_manager.render_task_prompt(
        task=Task.SELF_CHECK_HALLUCINATION,
        context={
            "statement": bot_response,
            "paragraph": ". ".join(extra_responses),
        },
    )
    stop = runtime.llm_task_manager.get_stop_tokens(task=Task.SELF_CHECK_HALLUCINATION)
    agreement = await call_llm_with_retries(
        llm=runtime.llm,
        prompt=prompt,
        stop=stop,
        max_retries=max_retries,
        llm_params={"temperature": runtime.rails_config.lowest_temperature},
    )
    agreement_text = (agreement or "").lower().strip()
    return "no" in agreement_text, {
        "agreement": agreement_text or "na",
        "extra_responses_count": len(extra_responses),
        "extra_responses_requested": extra_responses_count,
    }


async def run_ours(case: Dict[str, Any], args: argparse.Namespace) -> MethodResult:
    spec = METHOD_SPECS["ours"]
    module = importlib.import_module(spec.module)

    started = time.perf_counter()
    if args.enable_domain_expert_review:
        runtime = get_nemo_runtime(args)
        self_check = getattr(module, "self_check_domain_hallucination")
        result = await self_check(
            llm=runtime.llm,
            context={
                "bot_message": case.get("llm_answer", ""),
                "user_message": case.get("user_query", ""),
            },
            verification_level=args.domain_verification_level,
            enable_semantic_check=not args.disable_domain_semantic_check,
            enable_advanced_verification=not args.disable_domain_advanced_verification,
            skip_secondary_checks_on_dns_failure=args.skip_secondary_checks_on_dns_failure,
            enable_tls_verification=not args.disable_domain_tls,
            enable_whois_verification=not args.disable_domain_whois,
            enable_expert_review=True,
            expert_review_min_level=args.domain_expert_review_min_level,
            github_token=args.github_token or os.environ.get("GITHUB_TOKEN"),
        )
    else:
        analyze_answer = getattr(module, spec.callable_name)
        result = await analyze_answer(
            answer=case.get("llm_answer", ""),
            user_query=case.get("user_query", ""),
            verification_level=args.domain_verification_level,
            enable_semantic_check=not args.disable_domain_semantic_check,
            enable_advanced_verification=not args.disable_domain_advanced_verification,
            skip_secondary_checks_on_dns_failure=args.skip_secondary_checks_on_dns_failure,
            enable_tls_verification=not args.disable_domain_tls,
            enable_whois_verification=not args.disable_domain_whois,
            github_token=args.github_token or os.environ.get("GITHUB_TOKEN"),
        )
    latency = (time.perf_counter() - started) * 1000
    decision = result.get("decision", {}).get("action", "pass")

    detection = result.get("detection", {}) or {}
    extraction = result.get("extraction", {}) or {}
    verification_results = result.get("verification_results", {}) or {}
    verification_summary = {}
    for source in ("dns", "http", "tls", "whois", "github"):
        items = verification_results.get(source, []) or []
        statuses: Dict[str, int] = {}
        for item in items:
            if not isinstance(item, dict):
                continue
            status = str(item.get("status") or "unknown")
            statuses[status] = statuses.get(status, 0) + 1
        verification_summary[source] = {
            "count": len(items),
            "statuses": statuses,
        }

    return MethodResult(
        predicted=decision,
        latency_ms=latency,
        mode="domain_hallucination_live_expert"
        if args.enable_domain_expert_review
        else "domain_hallucination_live_no_expert",
        raw={
            "verification_level": args.domain_verification_level,
            "skip_secondary_checks_on_dns_failure": args.skip_secondary_checks_on_dns_failure,
            "tls_verification_enabled": not args.disable_domain_tls,
            "whois_verification_enabled": not args.disable_domain_whois,
            "expert_review_enabled": args.enable_domain_expert_review,
            "expert_review_min_level": args.domain_expert_review_min_level
            if args.enable_domain_expert_review
            else None,
            "expert_review": result.get("expert_review"),
            "risk_score": result.get("risk_score"),
            "recalibrated_score": result.get("recalibrated_score"),
            "issues_count": len(detection.get("issues", []) or []),
            "extracted_counts": {
                "urls": len(extraction.get("urls", []) or []),
                "domains": len(extraction.get("domains", []) or []),
                "github_repos": len(extraction.get("github_repos", []) or []),
            },
            "verification_summary": verification_summary,
            "semantic_check_enabled": not args.disable_domain_semantic_check,
            "advanced_verification_enabled": not args.disable_domain_advanced_verification,
        },
    )


async def run_baseline(case: Dict[str, Any], method: str, args: argparse.Namespace) -> MethodResult:
    spec = METHOD_SPECS[method]

    try:
        module = importlib.import_module(spec.module)
        action_fn = getattr(module, spec.callable_name)
    except Exception as exc:
        return MethodResult(
            predicted="pass",
            latency_ms=0.0,
            mode=f"{method}_skipped",
            skipped=True,
            error=f"Could not import {spec.module}.{spec.callable_name}: {exc}",
        )

    if method == "nemo_hallucination":
        try:
            runtime = get_nemo_runtime(args)
        except Exception as exc:
            return MethodResult(
                predicted="pass",
                latency_ms=0.0,
                mode="nemo_hallucination_skipped_runtime_unavailable",
                skipped=True,
                error=f"Could not initialize NeMo runtime from {args.nemo_config}: {exc}",
                raw={"connected_action": f"{spec.module}.{spec.callable_name}"},
            )

        started = time.perf_counter()
        hallucinated, raw_result = await run_hallucination_baseline_sequential(
            runtime=runtime,
            user_query=case.get("user_query", ""),
            bot_response=case.get("llm_answer", ""),
            extra_responses_count=args.hallucination_extra_responses,
            max_retries=args.hallucination_retries,
        )
        latency = (time.perf_counter() - started) * 1000
        return MethodResult(
            predicted="block" if hallucinated else "pass",
            latency_ms=latency,
            mode="nemo_hallucination_live_deepseek_sequential",
            raw={
                "connected_action": f"{spec.module}.{spec.callable_name}",
                "hallucinated": bool(hallucinated),
                "requires_extra_llm_calls": True,
                **raw_result,
            },
        )

    if method == "self_check_facts":
        context = case_context(case, include_relevant_chunks=True)
        has_context = bool(context.get("relevant_chunks"))
        if not has_context and not args.run_context_free_self_check_facts:
            return MethodResult(
                predicted="pass",
                latency_ms=0.0,
                mode="self_check_facts_skipped_no_relevant_chunks",
                skipped=True,
                error=(
                    "Dataset case has no relevant_chunks. Use --run-context-free-self-check-facts "
                    "to record the NeMo no-context pass behavior."
                ),
                raw={"connected_action": f"{spec.module}.{spec.callable_name}"},
            )

        if not has_context:
            started = time.perf_counter()
            latency = (time.perf_counter() - started) * 1000
            return MethodResult(
                predicted="pass",
                latency_ms=latency,
                mode="self_check_facts_context_free_behavior",
                raw={
                    "connected_action": f"{spec.module}.{spec.callable_name}",
                    "has_relevant_chunks": False,
                    "assumption": "No evidence means the NeMo action returns pass before LLM fact checking.",
                },
            )

        try:
            runtime = get_nemo_runtime(args)
        except Exception as exc:
            return MethodResult(
                predicted="pass",
                latency_ms=0.0,
                mode="self_check_facts_skipped_runtime_unavailable",
                skipped=True,
                error=f"Could not initialize NeMo runtime from {args.nemo_config}: {exc}",
                raw={"connected_action": f"{spec.module}.{spec.callable_name}"},
            )

        started = time.perf_counter()
        score = await action_fn(
            llm_task_manager=runtime.llm_task_manager,
            context=context,
            llm=runtime.llm,
            config=runtime.rails_config,
        )
        latency = (time.perf_counter() - started) * 1000
        return MethodResult(
            predicted="block" if float(score) < 0.5 else "pass",
            latency_ms=latency,
            mode="self_check_facts_live_deepseek",
            raw={
                "connected_action": f"{spec.module}.{spec.callable_name}",
                "has_relevant_chunks": True,
                "score": score,
            },
        )

    if method == "align_score":
        if not args.alignscore_endpoint:
            return MethodResult(
                predicted="pass",
                latency_ms=0.0,
                mode="align_score_skipped_requires_endpoint",
                skipped=True,
                error="Provide --alignscore-endpoint, e.g. http://localhost:5000/alignscore_base.",
                raw={"connected_action": f"{spec.module}.{spec.callable_name}"},
            )

        context = case_context(case, include_relevant_chunks=True)
        evidence = context.get("relevant_chunks") or []
        if not evidence and not args.run_context_free_alignscore:
            return MethodResult(
                predicted="pass",
                latency_ms=0.0,
                mode="align_score_skipped_no_relevant_chunks",
                skipped=True,
                error=(
                    "Dataset case has no relevant_chunks. Use --run-context-free-alignscore "
                    "to record AlignScore no-evidence pass behavior."
                ),
                raw={"connected_action": f"{spec.module}.{spec.callable_name}"},
            )

        request_module = importlib.import_module("nemoguardrails.library.factchecking.align_score.request")
        request_fn = getattr(request_module, "alignscore_request")
        started = time.perf_counter()
        score = await request_fn(
            api_url=args.alignscore_endpoint,
            evidence=evidence,
            response=case.get("llm_answer", ""),
        )
        latency = (time.perf_counter() - started) * 1000
        score_value = 1.0 if score is None else float(score)
        return MethodResult(
            predicted="block" if score_value < 0.5 else "pass",
            latency_ms=latency,
            mode="align_score_live_endpoint",
            raw={
                "connected_action": f"{spec.module}.{spec.callable_name}",
                "endpoint": args.alignscore_endpoint,
                "has_relevant_chunks": bool(evidence),
                "score": score,
            },
        )

    raise ValueError(f"Unknown method: {method}")


async def evaluate_method(
    cases: Iterable[Dict[str, Any]],
    method: str,
    args: argparse.Namespace,
    *,
    experiment_id: str,
    experiment_name: str,
    dataset_name: str,
    recorder: Optional[IncrementalRecorder] = None,
) -> Dict[str, Any]:
    metrics = Metrics()

    for case in cases:
        existing = None
        if recorder is not None:
            existing = recorder.find_event(
                experiment_id=experiment_id,
                experiment_name=experiment_name,
                dataset_name=dataset_name,
                method=method,
                case_id=str(case.get("id") or ""),
            )
        if existing is not None:
            if existing.get("skipped"):
                metrics.add_skip(case, str(existing.get("error") or "skipped"))
            elif existing.get("predicted") is not None:
                metrics.add(
                    case,
                    MethodResult(
                        predicted=str(existing.get("predicted") or "pass"),
                        latency_ms=float(existing.get("latency_ms") or 0.0),
                        mode=str(existing.get("mode") or "resumed"),
                        raw=existing.get("raw") if isinstance(existing.get("raw"), dict) else {},
                    ),
                )
            elif existing.get("error"):
                metrics.add_error(case, str(existing.get("error")))
            continue

        try:
            result = await run_ours(case, args) if method == "ours" else await run_baseline(case, method, args)
        except Exception as exc:
            metrics.add_error(case, str(exc))
            if recorder is not None:
                recorder.record(
                    experiment_id=experiment_id,
                    experiment_name=experiment_name,
                    dataset_name=dataset_name,
                    method=method,
                    case=case,
                    error=str(exc),
                )
            continue

        if result.skipped:
            metrics.add_skip(case, result.error or "skipped")
            if recorder is not None:
                recorder.record(
                    experiment_id=experiment_id,
                    experiment_name=experiment_name,
                    dataset_name=dataset_name,
                    method=method,
                    case=case,
                    result=result,
                )
            continue

        metrics.add(case, result)
        if recorder is not None:
            recorder.record(
                experiment_id=experiment_id,
                experiment_name=experiment_name,
                dataset_name=dataset_name,
                method=method,
                case=case,
                result=result,
            )

    return {
        "method": asdict(METHOD_SPECS[method]),
        "metrics": metrics.compute(),
        "details": metrics.details,
    }


async def run_experiment(
    experiment_id: str,
    dataset_name: str,
    args: argparse.Namespace,
    recorder: Optional[IncrementalRecorder] = None,
) -> Dict[str, Any]:
    experiment = EXPERIMENTS[experiment_id]
    dataset = load_dataset(FILES_ROOT / "01_datasets" / dataset_name)
    cases = dataset.get("test_cases", [])
    if args.max_cases_per_dataset is not None:
        cases = cases[: args.max_cases_per_dataset]

    print(f"\n=== Experiment {experiment_id}: {experiment['name']} ===")
    print(f"Dataset: {dataset_name} ({len(cases)} cases)")
    print(f"Question: {experiment['question']}")

    if args.baseline_only:
        ours = {
            "method": asdict(METHOD_SPECS["ours"]),
            "metrics": {
                "total_samples": len(cases),
                "evaluated_samples": 0,
                "skipped_samples": len(cases),
                "error_samples": 0,
                "skipped": [
                    {"id": case.get("id", ""), "reason": "domain_hallucination skipped by --baseline-only"}
                    for case in cases
                ],
                "errors": [],
            },
            "details": [],
        }
    else:
        ours = await evaluate_method(
            cases,
            "ours",
            args,
            experiment_id=experiment_id,
            experiment_name=experiment["name"],
            dataset_name=dataset_name,
            recorder=recorder,
        )
    if args.skip_baseline:
        baseline = {
            "method": asdict(METHOD_SPECS[experiment["baseline"]]),
            "metrics": {
                "total_samples": len(cases),
                "evaluated_samples": 0,
                "skipped_samples": len(cases),
                "error_samples": 0,
                "skipped": [
                    {"id": case.get("id", ""), "reason": "baseline skipped by --skip-baseline"} for case in cases
                ],
                "errors": [],
            },
            "details": [],
        }
    else:
        baseline = await evaluate_method(
            cases,
            experiment["baseline"],
            args,
            experiment_id=experiment_id,
            experiment_name=experiment["name"],
            dataset_name=dataset_name,
            recorder=recorder,
        )

    return {
        "experiment_id": experiment_id,
        "experiment_name": experiment["name"],
        "dataset": dataset_name,
        "dataset_metadata": dataset.get("metadata", {}),
        "question": experiment["question"],
        "domain_hallucination": ours,
        "baseline": baseline,
    }


def selected_experiments(values: Optional[List[str]]) -> List[str]:
    if not values:
        return ["A", "B", "C"]
    selected = [value.upper() for value in values]
    unknown = [value for value in selected if value not in EXPERIMENTS]
    if unknown:
        raise SystemExit(f"Unknown experiment id(s): {', '.join(unknown)}")
    return selected


async def main_async(args: argparse.Namespace) -> None:
    experiments = selected_experiments(args.experiments)
    datasets = args.datasets or DATASETS
    output = ROOT / args.output
    recorder = IncrementalRecorder(output.with_suffix(output.suffix + ".partial.json"), args)

    results = []
    for experiment_id in experiments:
        for dataset_name in datasets:
            results.append(await run_experiment(experiment_id, dataset_name, args, recorder=recorder))

    report = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "repo_root": str(REPO_ROOT),
        "datasets": datasets,
        "experiments": experiments,
        "system_under_test": {
            "name": "domain_hallucination",
            "verification_level": args.domain_verification_level,
            "enable_semantic_check": not args.disable_domain_semantic_check,
            "enable_advanced_verification": not args.disable_domain_advanced_verification,
            "skip_secondary_checks_on_dns_failure": args.skip_secondary_checks_on_dns_failure,
            "enable_tls_verification": not args.disable_domain_tls,
            "enable_whois_verification": not args.disable_domain_whois,
            "expert_review_enabled": args.enable_domain_expert_review,
            "expert_review_min_level": args.domain_expert_review_min_level
            if args.enable_domain_expert_review
            else None,
            "expert_review_future_optional": True,
            "baseline_skipped": args.skip_baseline,
            "baseline_only": args.baseline_only,
        },
        "method_specs": {name: asdict(spec) for name, spec in METHOD_SPECS.items()},
        "results": results,
    }

    with open(output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\nSaved comparison report to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run A/B/C baseline comparisons for domain hallucination detection.")
    parser.add_argument(
        "--experiments",
        nargs="*",
        default=None,
        help="Experiment IDs to run: A, B, C. Defaults to all three.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Dataset files under files/01_datasets/. Defaults to eval_dataset.json and expanded_dataset.json.",
    )
    parser.add_argument(
        "--max-cases-per-dataset",
        type=int,
        default=None,
        help="Limit cases per dataset for smoke tests or quick strategy trials.",
    )
    parser.add_argument(
        "--output",
        default="ablation_comparison_results.json",
        help="Output JSON file under this script directory.",
    )
    parser.add_argument(
        "--disable-incremental-save",
        action="store_true",
        help="Disable per-case partial result writing. By default progress is saved after each case.",
    )
    parser.add_argument(
        "--resume-from-partial",
        action="store_true",
        help="Reuse completed case results from the output partial JSON if present.",
    )
    parser.add_argument(
        "--domain-verification-level",
        default="dns",
        choices=["none", "dns", "http", "full"],
        help="Verification level for our domain_hallucination detector.",
    )
    parser.add_argument(
        "--disable-domain-semantic-check",
        action="store_true",
        help="Disable our semantic relevance check. By default it is enabled.",
    )
    parser.add_argument(
        "--disable-domain-advanced-verification",
        action="store_true",
        help="Disable our advanced verification heuristics. By default they are enabled.",
    )
    parser.add_argument(
        "--skip-secondary-checks-on-dns-failure",
        action="store_true",
        help="In full/http modes, skip HTTP/TLS/WHOIS for domains that already fail DNS.",
    )
    parser.add_argument(
        "--disable-domain-tls",
        action="store_true",
        help="Disable TLS checks in full mode.",
    )
    parser.add_argument(
        "--disable-domain-whois",
        action="store_true",
        help="Disable WHOIS/RDAP checks in full mode.",
    )
    parser.add_argument(
        "--enable-domain-expert-review",
        action="store_true",
        help="Enable expert semantic review for domain_hallucination using the NeMo-configured LLM.",
    )
    parser.add_argument(
        "--domain-expert-review-min-level",
        default="L2",
        choices=["L0", "L1", "L2", "L3", "L4"],
        help="Minimum rule risk level required before expert review is invoked.",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Run only domain_hallucination and mark the baseline as skipped.",
    )
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Run only the selected baseline and mark domain_hallucination as skipped.",
    )
    parser.add_argument(
        "--nemo-config",
        default="deepseek_config.yml",
        help="NeMo config file under files/ for LLM-backed baselines.",
    )
    parser.add_argument(
        "--github-token",
        default=None,
        help="Optional GitHub token for our GitHub repo existence checks.",
    )
    parser.add_argument(
        "--alignscore-endpoint",
        default=None,
        help="Optional AlignScore endpoint, e.g. http://localhost:5000/alignscore_base.",
    )
    parser.add_argument(
        "--run-context-free-self-check-facts",
        action="store_true",
        help="Record self_check_facts no-context behavior without invoking an LLM.",
    )
    parser.add_argument(
        "--run-context-free-alignscore",
        action="store_true",
        help="Record AlignScore no-evidence behavior instead of skipping cases without relevant_chunks.",
    )
    parser.add_argument(
        "--hallucination-extra-responses",
        type=int,
        default=2,
        help="Number of extra responses for the NeMo hallucination baseline.",
    )
    parser.add_argument(
        "--hallucination-retries",
        type=int,
        default=2,
        help="Retries for each DeepSeek call in the NeMo hallucination baseline.",
    )
    args = parser.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
