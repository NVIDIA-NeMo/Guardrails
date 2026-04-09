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
import os
from typing import Optional

import aiohttp

from nemoguardrails.actions import action

log = logging.getLogger(__name__)

VEROQ_API_BASE = "https://api.veroq.ai"


def veroq_check_facts_mapping(result: dict) -> bool:
    """Block if any claims are contradicted or trust score is below 0.7."""

    trust_score = result.get("trust_score", 1.0)
    contradictions = result.get("claims_contradicted", 0)
    return contradictions > 0 or trust_score < 0.7


@action(
    name="call veroq api",
    is_system_action=True,
    output_mapping=veroq_check_facts_mapping,
)
async def call_veroq_api(
    context: Optional[dict] = None,
    max_claims: int = 5,
    **kwargs,
) -> dict:
    """Check facts in LLM output using VeroQ Shield.

    Calls the VeroQ verify/output endpoint which extracts claims from the
    bot response and fact-checks each one against live sources.

    Returns a dict with trust_score, claims_contradicted, and claims details.
    """
    api_key = os.environ.get("VEROQ_API_KEY")

    if api_key is None:
        raise ValueError("VEROQ_API_KEY environment variable not set.")

    bot_response = context.get("bot_message")

    if not bot_response or len(bot_response.strip()) < 20:
        log.info("VeroQ: bot response too short to verify, passing through.")
        return {
            "trust_score": 1.0,
            "claims_contradicted": 0,
            "claims": [],
        }

    url = f"{VEROQ_API_BASE}/api/v1/verify/output"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    data = {
        "text": bot_response,
        "max_claims": max_claims,
        "source": "nemo-guardrails",
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url=url, headers=headers, json=data) as response:
            if response.status != 200:
                raise ValueError(
                    f"VeroQ API call failed with status code {response.status}.\n"
                    f"Details: {await response.text()}"
                )

            result = await response.json()

            trust_score = result.get("overall_confidence", 0)
            claims = result.get("claims", [])
            claims_contradicted = sum(
                1 for c in claims if c.get("verdict") == "contradicted"
            )
            corrections = [
                {
                    "claim": c.get("text", ""),
                    "correction": c.get("correction", ""),
                }
                for c in claims
                if c.get("verdict") == "contradicted" and c.get("correction")
            ]

            log.info(
                "VeroQ Shield: trust_score=%.2f, claims=%d, contradicted=%d",
                trust_score,
                len(claims),
                claims_contradicted,
            )

            return {
                "trust_score": trust_score,
                "claims_contradicted": claims_contradicted,
                "corrections": corrections,
                "claims": claims,
            }
