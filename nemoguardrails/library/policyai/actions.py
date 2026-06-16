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

import aiohttp

from nemoguardrails.actions import action

log = logging.getLogger(__name__)


def call_policyai_api_mapping(result: dict) -> bool:
    """
    Mapping for call_policyai_api.

    Expects result to be a dict with:
      - "assessment": "SAFE" or "UNSAFE"
      - "category": the violation category (if UNSAFE)
      - "severity": severity level 0-3

    Block (return True) if:
      1. Assessment is "UNSAFE"
    """
    assessment = result.get("assessment", "SAFE")
    return assessment == "UNSAFE"


@action(is_system_action=True, output_mapping=call_policyai_api_mapping)
async def call_policyai_api(
    text: Optional[str] = None,
    tag_name: Optional[str] = None,
    **kwargs,
):
    """
    Call the PolicyAI API to evaluate content.

    Args:
        text: The text content to evaluate.
        tag_name: Optional tag name for the PolicyAI evaluation.
                  If not provided, uses POLICYAI_TAG_NAME env var or "prod".

    Returns:
        dict with:
          - assessment: "SAFE" or "UNSAFE"
          - category: the violation category (if UNSAFE)
          - severity: severity level 0-3
          - reason: explanation for the decision
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

    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(
            url=url,
            headers=headers,
            json=data,
        ) as response:
            if response.status != 200:
                raise ValueError(
                    f"PolicyAI call failed with status code {response.status}.\nDetails: {await response.text()}"
                )
            response_json = await response.json()
            log.info(json.dumps(response_json, indent=2))

            # PolicyAI returns results in "data" array for tag-based evaluation
            results = response_json.get("data", [])

            # Fail-closed: If no policies are attached to the tag, raise an error
            # rather than silently allowing content through
            if not results:
                raise ValueError(
                    f"PolicyAI returned no policy results for tag '{tag_name}'. "
                    "Ensure policies are attached to this tag."
                )

            # Check if all policies failed evaluation
            successful_results = [r for r in results if r.get("status") != "failed"]
            if not successful_results:
                raise ValueError(
                    f"All PolicyAI policy evaluations failed for tag '{tag_name}'. Check policy configurations."
                )

            # Aggregate results - if ANY policy returns UNSAFE, overall is UNSAFE
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
                    break  # Stop at first UNSAFE result

            # Pre-format exception message for Colang 1.x compatibility
            # (Colang 1.x doesn't support string concatenation in create event)
            exception_message = f"PolicyAI moderation triggered. Content violated policy: {triggered_category}"

            return {
                "assessment": overall_assessment,
                "category": triggered_category,
                "severity": max_severity,
                "reason": reason,
                "exception_message": exception_message,
            }
