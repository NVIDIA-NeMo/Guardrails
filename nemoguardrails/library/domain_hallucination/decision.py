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

"""Decision engine for enforcement actions."""

from __future__ import annotations

from typing import Any, Dict

from . import config as _config_module


class PolicyThresholds:
    """Configurable policy thresholds."""

    def __init__(
        self,
        fail_threshold: float = 60.0,
        refine_threshold: float = 40.0,
        warn_threshold: float = 20.0,
    ):
        self.fail_threshold = fail_threshold
        self.refine_threshold = refine_threshold
        self.warn_threshold = warn_threshold


def make_decision(
    risk_score: Dict[str, Any],
    recalibrated_score: Dict[str, Any] | None = None,
    policy: PolicyThresholds | None = None,
    verification_level: str = "dns",
) -> Dict[str, Any]:
    """Make enforcement decision based on risk score."""
    policy = policy or PolicyThresholds()
    recalibrated_score = recalibrated_score or {}

    # Use recalibrated score if available, otherwise use initial score
    recal_score = recalibrated_score.get("recalibrated_score")
    final_score = float(recal_score if recal_score is not None else risk_score.get("score", 0.0))
    recal_level = recalibrated_score.get("recalibrated_level")
    level = recal_level if recal_level is not None else risk_score.get("level", "L0")

    # Determine action based on score and verification level
    if final_score >= policy.fail_threshold:
        action = "block"
        reason = f"Risk score {final_score} exceeds fail threshold {policy.fail_threshold}"
    elif final_score >= policy.refine_threshold:
        action = "refine"
        reason = f"Risk score {final_score} exceeds refine threshold {policy.refine_threshold}"
    elif final_score >= policy.warn_threshold:
        action = "warn"
        reason = f"Risk score {final_score} exceeds warn threshold {policy.warn_threshold}"
    else:
        action = "pass"
        reason = f"Risk score {final_score} below warn threshold {policy.warn_threshold}"

    # Adjust decision based on verification level
    if verification_level == "none":
        if action in {"block", "refine"}:
            action = "warn"
            reason = f"{reason}; downgraded due to verification_level=none"
    elif verification_level == "dns":
        pass  # DNS-level is default
    elif verification_level == "http":
        pass  # HTTP-level requires more evidence

    return {
        "action": action,
        "reason": reason,
        "level": level,
        "score": final_score,
        "threshold_fail": policy.fail_threshold,
        "threshold_refine": policy.refine_threshold,
        "threshold_warn": policy.warn_threshold,
        "verification_level": verification_level,
    }


def apply_decision(
    decision: Dict[str, Any],
    answer: str = "",
    cfg=None,
) -> Dict[str, Any]:
    """Apply enforcement decision to answer."""
    action = str(decision.get("action", "pass")).lower()
    reason = str(decision.get("reason", ""))
    if cfg is None:
        cfg = _config_module.get_config()
    enforcement = cfg.enforcement
    append_notice = enforcement.append_verification_notice

    refine_suffix = "\n\n[Refined by domain guard]" if append_notice else ""
    warn_suffix = "\n\n[Please verify external links independently]" if append_notice else ""

    if action == "block":
        modified_answer = enforcement.block_message
        return {
            "action": "block",
            "reason": reason,
            "original_answer": answer,
            "modified_answer": modified_answer,
            "enforced": True,
        }

    elif action == "refine":
        modified_answer = f"{enforcement.refine_message}\n\n{answer}{refine_suffix}"
        return {
            "action": "refine",
            "reason": reason,
            "original_answer": answer,
            "modified_answer": modified_answer,
            "enforced": True,
        }

    elif action == "warn":
        modified_answer = f"{enforcement.warn_message}\n\n{answer}{warn_suffix}"
        return {
            "action": "warn",
            "reason": reason,
            "original_answer": answer,
            "modified_answer": modified_answer,
            "enforced": True,
        }

    else:  # pass
        return {
            "action": "pass",
            "reason": reason,
            "original_answer": answer,
            "modified_answer": answer,
            "enforced": False,
        }
