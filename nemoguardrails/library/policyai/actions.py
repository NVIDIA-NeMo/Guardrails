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
"""
PolicyAI Integration for NeMo Guardrails.

PolicyAI provides content moderation and policy enforcement capabilities
for LLM applications. This integration allows using PolicyAI as an input
and output rail for content moderation.

For more information, see: https://musubilabs.ai
"""

import json
import logging
import os
from typing import Optional

from nemoguardrails.actions import action
from nemoguardrails.actions.rail_outcome import RailOutcome
from nemoguardrails.http import HTTPClient, http_call

log = logging.getLogger(__name__)


def _policyai_outcome(metadata: dict) -> RailOutcome:
    metadata = dict(metadata)
    reason = metadata.pop("reason", None)
    if metadata["assessment"] == "UNSAFE":
        return RailOutcome.block(reason=reason, metadata=metadata)
    return RailOutcome.allow(reason=reason, metadata=metadata)


@action(is_system_action=True)
async def call_policyai_api(
    text: Optional[str] = None,
    tag_name: Optional[str] = None,
    http_client: Optional[HTTPClient] = None,
    **kwargs,
) -> RailOutcome:
    """
    Call the PolicyAI API to evaluate content.

    Args:
        text: The text content to evaluate.
        tag_name: Optional tag name for the PolicyAI evaluation.
                  If not provided, uses POLICYAI_TAG_NAME env var or "prod".

    Returns:
        RailOutcome indicating whether the content is blocked. Assessment, category,
        severity, and exception_message are metadata; reason is stored on RailOutcome.reason.
    """
    api_key = os.environ.get("POLICYAI_API_KEY")

    if api_key is None:
        raise ValueError("POLICYAI_API_KEY environment variable not set.")

    base_url = os.environ.get("POLICYAI_BASE_URL", "https://api.musubilabs.ai")
    base_url = base_url.rstrip("/")

    # Get tag name from parameter, env var, or default
    if tag_name is None:
        tag_name = os.environ.get("POLICYAI_TAG_NAME", "prod")

    url = f"{base_url}/policyai/v1/decisions/evaluate/{tag_name}"

    headers = {
        "Musubi-Api-Key": api_key,
        "Content-Type": "application/json",
    }

    data = {
        "content": [
            {
                "type": "TEXT",
                "content": text,
            }
        ],
    }

    response = await http_call(
        http_client,
        "POST",
        url,
        headers=headers,
        json=data,
        timeout=30,
        raise_for_status=False,
    )
    if response.status_code != 200:
        raise ValueError(f"PolicyAI call failed with status code {response.status_code}.\nDetails: {response.text}")
    response_json = response.json()
    log.info(json.dumps(response_json, indent=2))

    results = response_json.get("data", [])

    if not results:
        raise ValueError(
            f"PolicyAI returned no policy results for tag '{tag_name}'. Ensure policies are attached to this tag."
        )

    successful_results = [r for r in results if r.get("status") != "failed"]
    if not successful_results:
        raise ValueError(f"All PolicyAI policy evaluations failed for tag '{tag_name}'. Check policy configurations.")

    overall_assessment = "SAFE"
    triggered_category = "Safe"
    max_severity = 0
    reason = "Content passed all policy checks"

    for result in successful_results:
        assessment = result.get("assessment", "SAFE")
        if assessment == "UNSAFE":
            overall_assessment = "UNSAFE"
            triggered_category = result.get("category", "Unknown")
            max_severity = max(max_severity, result.get("severity", 0))
            reason = result.get("reason", "Policy violation detected")
            break

    exception_message = f"PolicyAI moderation triggered. Content violated policy: {triggered_category}"

    return _policyai_outcome(
        {
            "assessment": overall_assessment,
            "category": triggered_category,
            "severity": max_severity,
            "reason": reason,
            "exception_message": exception_message,
        }
    )
