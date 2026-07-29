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

import logging
from typing import Literal, Optional

from nemoguardrails.actions import action
from nemoguardrails.actions.rail_outcome import RailOutcome
from nemoguardrails.library.gcp_moderate_text.rail import GCP_LANGUAGE_PACKAGE
from nemoguardrails.manifests import require_python_package

log = logging.getLogger(__name__)

GCP_TEXT_DETAILED_THRESHOLDS = {
    "Toxic": 0.8,
    "Insult": 0.7,
    "Profanity": 0.6,
    "Derogatory": 0.4,
    "Violent": 0.8,
    "Sexual": 0.7,
    "Death, Harm & Tragedy": 0.8,
    "Firearms & Weapons": 0.8,
    "Illicit Drugs": 0.8,
    "Public Safety": 0.8,
    "Health": 0.8,
    "Religion & Belief": 0.8,
    "War & Conflict": 0.8,
    "Politics": 0.8,
    "Finance": 0.8,
    "Legal": 0.8,
}


def _gcp_text_simple_blocked(max_risk_score: float) -> bool:
    return max_risk_score > 0.8


def _gcp_text_triggered_violation(violations: dict[str, float]) -> Optional[str]:
    return next(
        (name for name, threshold in GCP_TEXT_DETAILED_THRESHOLDS.items() if violations.get(name, 0) > threshold),
        None,
    )


def _gcp_text_moderation_outcome(
    max_risk_score: float,
    violations: dict[str, float],
    threshold_mode: Literal["simple", "detailed"] = "simple",
) -> RailOutcome:
    triggered_violation = _gcp_text_triggered_violation(violations) if threshold_mode == "detailed" else None
    metadata = {
        "max_risk_score": max_risk_score,
        "violations": violations,
        "threshold_mode": threshold_mode,
        "triggered_violation": triggered_violation,
    }
    if triggered_violation is not None:
        return RailOutcome.block(
            reason=f"GCP text moderation triggered. {triggered_violation} exceeded the threshold.",
            metadata=metadata,
        )
    if threshold_mode != "detailed" and _gcp_text_simple_blocked(max_risk_score):
        return RailOutcome.block(
            reason="GCP text moderation triggered. The maximum risk score exceeded the threshold.",
            metadata=metadata,
        )
    return RailOutcome.allow(metadata=metadata)


@action(
    name="call_gcpnlp_api",
    is_system_action=True,
)
async def call_gcp_text_moderation_api(
    context: Optional[dict] = None,
    threshold_mode: Literal["simple", "detailed"] = "simple",
    **kwargs,
) -> RailOutcome:
    """
    Application Default Credentials (ADC) is a strategy used by the GCP authentication libraries to automatically
    find credentials based on the application environment. ADC searches for credentials in the following locations (Search order):
    1. GOOGLE_APPLICATION_CREDENTIALS environment variable
    2. User credentials set up by using the Google Cloud CLI
    3. The attached service account, returned by the metadata server

    For more information check https://cloud.google.com/docs/authentication/application-default-credentials
    """
    language_v2 = require_python_package("gcp_moderate_text", GCP_LANGUAGE_PACKAGE)

    context = context or {}
    user_message = context.get("user_message")
    client = language_v2.LanguageServiceAsyncClient()

    # Initialize request argument(s)
    document = language_v2.Document()
    document.content = user_message
    document.type_ = language_v2.Document.Type.PLAIN_TEXT

    response = await client.moderate_text(document=document)

    violations_dict = {}
    max_risk_score = 0.0
    for violation in response.moderation_categories:
        if violation.confidence > max_risk_score:
            max_risk_score = violation.confidence
        violations_dict[violation.name] = violation.confidence

    return _gcp_text_moderation_outcome(max_risk_score, violations_dict, threshold_mode)
