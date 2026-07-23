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
import json
import logging
import os
from typing import Literal, Optional

from nemoguardrails.actions import action
from nemoguardrails.actions.rail_outcome import RailOutcome
from nemoguardrails.http import HTTPClient, http_call
from nemoguardrails.utils import new_uuid

log = logging.getLogger(__name__)

ACTIVEFENCE_DETAILED_RULES = {
    "abusive_or_harmful.harassment_or_bullying": (
        0.8,
        "ActiveFence moderation triggered. The harassment or bullying risk score exceeded the threshold.",
    ),
    "abusive_or_harmful.profanity": (
        0.7,
        "ActiveFence moderation triggered. The profanity risk score exceeded the threshold.",
    ),
    "abusive_or_harmful.hate_speech": (
        0.6,
        "ActiveFence moderation triggered. The hate speech risk score exceeded the threshold.",
    ),
    "abusive_or_harmful.child_grooming": (
        0.4,
        "ActiveFence moderation triggered. The child grooming risk score exceeded the threshold.",
    ),
    "abusive_or_harmful.general_violence": (
        0.7,
        "ActiveFence moderation triggered. The general violence risk score exceeded the threshold.",
    ),
    "self_harm.general": (
        0.8,
        "ActiveFence moderation triggered. The self harm risk score exceeded the threshold.",
    ),
    "adult_content.general": (
        0.3,
        "ActiveFence moderation triggered. The adult content risk score exceeded the threshold.",
    ),
    "privacy_violation.pii": (
        0.8,
        "ActiveFence moderation triggered. The privacy violation risk score exceeded the threshold.",
    ),
}


def _activefence_simple_blocked(max_risk_score: float) -> bool:
    return max_risk_score > 0.7


def _activefence_triggered_violation(violations: dict[str, float]) -> Optional[str]:
    return next(
        (
            violation_type
            for violation_type, (threshold, _) in ACTIVEFENCE_DETAILED_RULES.items()
            if violations.get(violation_type, 0) > threshold
        ),
        None,
    )


def _activefence_outcome(
    max_risk_score: float,
    violations: dict[str, float],
    threshold_mode: Literal["simple", "detailed"] = "simple",
) -> RailOutcome:
    triggered_violation = _activefence_triggered_violation(violations) if threshold_mode == "detailed" else None
    metadata = {
        "max_risk_score": max_risk_score,
        "violations": violations,
        "threshold_mode": threshold_mode,
        "triggered_violation": triggered_violation,
    }
    if triggered_violation is not None:
        reason = ACTIVEFENCE_DETAILED_RULES[triggered_violation][1]
        return RailOutcome.block(reason=reason, metadata=metadata)
    if threshold_mode != "detailed" and _activefence_simple_blocked(max_risk_score):
        return RailOutcome.block(
            reason="ActiveFence moderation triggered. The maximum risk score exceeded the threshold.",
            metadata=metadata,
        )
    return RailOutcome.allow(metadata=metadata)


@action(is_system_action=True)
async def call_activefence_api(
    text: Optional[str] = None,
    threshold_mode: Literal["simple", "detailed"] = "simple",
    http_client: Optional[HTTPClient] = None,
    **kwargs,
) -> RailOutcome:
    api_key = os.environ.get("ACTIVEFENCE_API_KEY")

    if api_key is None:
        raise ValueError("ACTIVEFENCE_API_KEY environment variable not set.")

    url = "https://apis.activefence.com/sync/v3/content/text"
    headers = {"af-api-key": api_key, "af-source": "nemo-guardrails"}
    data = {
        "text": text,
        "content_id": "ng-" + new_uuid(),
    }

    response = await http_call(
        http_client,
        "POST",
        url,
        headers=headers,
        json=data,
        raise_for_status=False,
    )
    if response.status_code != 200:
        raise ValueError(f"ActiveFence call failed with status code {response.status_code}.\nDetails: {response.text}")
    response_json = response.json()
    log.info(json.dumps(response_json, indent=True))
    violations = response_json["violations"]

    violations_dict = {}
    max_risk_score = 0.0
    for violation in violations:
        if violation["risk_score"] > max_risk_score:
            max_risk_score = violation["risk_score"]
        violations_dict[violation["violation_type"]] = violation["risk_score"]

    return _activefence_outcome(max_risk_score, violations_dict, threshold_mode)
