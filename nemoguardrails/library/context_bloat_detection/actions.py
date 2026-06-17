# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Context bloat detection action

Detects context-manipulation attacks where attacker-controlled content
(retrieved chunks or user input) is padded, oversized, or repetitively
structured to cause system prompt forgetting, bury instructions mid-context
(harder to detect), or exhaust token budget.

Checks:
    * Size cap
    * Entropy (sampling for very large inputs)
    * Longest repeated character
    * Repeated n-grams
    * Check order:  size > entropy > run > repetition

Wire as retrieval rail (RAG chunks) or input rail.
"""

import logging
import math
from collections import Counter
from typing import List, Optional, TypedDict

from nemoguardrails import RailsConfig
from nemoguardrails.actions import action

log = logging.getLogger(__name__)

# Entropy is statistically stable well below this threshold; sampling avoids O(n) work.
ENTROPY_SAMPLE_THRESHOLD = 10000
ENTROPY_SAMPLE_CHARS = 8000


class ContextBloatResult(TypedDict):
    is_bloat: bool
    action: str
    text: str
    reason: Optional[str]
    detections: List[str]
    metrics: dict


def _stratified_sample(text: str, sample_chars: int) -> str:
    third = sample_chars // 3
    mid = len(text) // 2
    return text[:third] + text[mid - third // 2 : mid + third // 2] + text[-third:]


def _shannon_entropy(text: str) -> float:
    """Samples large inputs to bound runtime."""
    if not text:
        return 0.0
    sample = _stratified_sample(text, ENTROPY_SAMPLE_CHARS) if len(text) > ENTROPY_SAMPLE_THRESHOLD else text
    counts = Counter(sample)
    total = len(sample)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def _repetition_ratio(text: str, n: int = 3) -> float:
    """High values are a padding-attack signature."""
    if not text:
        return 0.0
    tokens = text.split()
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    counter = Counter(ngrams)
    repeated = sum(c - 1 for c in counter.values() if c > 1)
    return repeated / len(ngrams) if ngrams else 0.0


def _longest_run_ratio(text: str) -> float:
    """Fraction of text that is the longest run of a single character."""
    if not text:
        return 0.0
    n = len(text)
    longest = 1
    i = 0
    while i < n:
        j = i + 1
        while j < n and text[j] == text[i]:
            j += 1
        if j - i > longest:
            longest = j - i
        i = j
    return longest / n


def _validate_config(config: RailsConfig) -> None:
    cfg = getattr(config.rails.config, "context_bloat_detection", None)
    if cfg is None:
        raise ValueError("context_bloat_detection configuration is missing in RailsConfig.")
    if cfg.action not in {"reject", "truncate", "warn"}:
        raise ValueError(f"Expected 'reject', 'truncate', or 'warn' but got {cfg.action!r}.")


def _check_entropy(text: str, cfg, detections: List[str], metrics: dict) -> Optional[ContextBloatResult]:
    entropy = _shannon_entropy(text)
    metrics["entropy"] = round(entropy, 3)
    if entropy < cfg.min_entropy:
        detections.append("low_entropy")
        if cfg.action in ("reject", "truncate"):
            log.info(f"context bloat detected: low_entropy | entropy={entropy:.3f}")
            return ContextBloatResult(
                is_bloat=True,
                action="reject",
                text=text,
                reason="low_entropy",
                detections=detections,
                metrics=metrics,
            )
    return None


def _check_longest_run(text: str, cfg, detections: List[str], metrics: dict) -> Optional[ContextBloatResult]:
    run_ratio = _longest_run_ratio(text)
    metrics["longest_run_ratio"] = round(run_ratio, 3)
    if run_ratio > cfg.max_run_ratio:
        detections.append("long_run")
        if cfg.action in ("reject", "truncate"):
            log.info(f"context bloat detected: long_run | run_ratio={run_ratio:.3f}")
            return ContextBloatResult(
                is_bloat=True,
                action="reject",
                text=text,
                reason="long_run",
                detections=detections,
                metrics=metrics,
            )
    return None


def _check_repetition(text: str, cfg, detections: List[str], metrics: dict) -> Optional[ContextBloatResult]:
    rep_ratio = _repetition_ratio(text, n=cfg.ngram_size)
    metrics["repetition_ratio"] = round(rep_ratio, 3)
    if rep_ratio > cfg.max_repetition_ratio:
        detections.append("high_repetition")
        if cfg.action in ("reject", "truncate"):
            log.info(f"context bloat detected: high_repetition | rep_ratio={rep_ratio:.3f}")
            return ContextBloatResult(
                is_bloat=True,
                action="reject",
                text=text,
                reason="high_repetition",
                detections=detections,
                metrics=metrics,
            )
    return None


@action()
async def context_bloat_detection(text: str, config: RailsConfig) -> ContextBloatResult:
    """Detect context-bloat / context-manipulation attacks.
    Check order is cheapest first to enable early-exit.

    Args:
        text: The text to inspect (joined chunks or user message).
        config: RailsConfig with rails.config.context_bloat_detection settings.

    Returns:
        ContextBloatResult with is_bloat flag, processed text, reason, metrics.
    """
    _validate_config(config)
    cfg = config.rails.config.context_bloat_detection

    char_count = len(text) if text else 0
    detections: List[str] = []
    metrics: dict = {"chars": char_count}

    if char_count > cfg.max_chars:
        detections.append("size_cap_exceeded")
        log.info(f"context bloat detected: size_cap_exceeded | chars={char_count}")
        if cfg.action == "reject":
            return ContextBloatResult(
                is_bloat=True,
                action=cfg.action,
                text=text,
                reason="size_cap_exceeded",
                detections=detections,
                metrics=metrics,
            )
        if cfg.action == "truncate":
            text = text[: cfg.max_chars]

    if char_count >= cfg.min_chars:
        for check in (_check_entropy, _check_longest_run, _check_repetition):
            result = check(text, cfg, detections, metrics)
            if result is not None:
                return result

    is_bloat = bool(detections)
    reason = ", ".join(detections) if detections else None
    if is_bloat:
        log.info(f"context bloat detected: {reason} | metrics={metrics}")
    return ContextBloatResult(
        is_bloat=is_bloat,
        action=cfg.action,
        text=text,
        reason=reason,
        detections=detections,
        metrics=metrics,
    )
