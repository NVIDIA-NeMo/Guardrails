# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from typing import Optional

import aiohttp

from nemoguardrails.actions import action
from nemoguardrails.utils import new_uuid

log = logging.getLogger(__name__)


def call_activefence_api_mapping(result: dict) -> bool:
    """
    Mapping for call_activefence_api.

    Expects result to be a dict with:
      - "max_risk_score": a float representing the overall risk.
      - "violations": a dict mapping violation type (str) to risk score (float).

    Block (return True) if any of the following conditions are met:
      1. Overall max risk score > 0.7.
      2. Individual thresholds are exceeded:
         - "abusive_or_harmful.harassment_or_bullying" > 0.8
         - "abusive_or_harmful.profanity" > 0.7
         - "abusive_or_harmful.hate_speech" > 0.6
         - "abusive_or_harmful.child_grooming" > 0.4
         - "abusive_or_harmful.general_violence" > 0.7
         - "self_harm.general" > 0.8
         - "adult_content.general" > 0.3
         - "privacy_violation.pii" > 0.8
    """
    max_risk = result.get("max_risk_score", 0.0)
    if max_risk > 0.7:
        return True

    violations = result.get("violations", {})

    thresholds = {
        "abusive_or_harmful.harassment_or_bullying": 0.8,
        "abusive_or_harmful.profanity": 0.7,
        "abusive_or_harmful.hate_speech": 0.6,
        "abusive_or_harmful.child_grooming": 0.4,
        "abusive_or_harmful.general_violence": 0.7,
        "self_harm.general": 0.8,
        "adult_content.general": 0.3,
        "privacy_violation.pii": 0.8,
    }

    for violation_type, threshold in thresholds.items():
        if violations.get(violation_type, 0) > threshold:
            return True

    return False


@action(is_system_action=True, output_mapping=call_activefence_api_mapping)
async def call_activefence_api(text: Optional[str] = None, **kwargs):
    api_key = os.environ.get("ACTIVEFENCE_API_KEY")

    if api_key is None:
        raise ValueError("ACTIVEFENCE_API_KEY environment variable not set.")

    url = "https://apis.activefence.com/sync/v3/content/text"
    headers = {"af-api-key": api_key, "af-source": "nemo-guardrails"}
    data = {
        "text": text,
        "content_id": "ng-" + new_uuid(),
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(
            url=url,
            headers=headers,
            json=data,
        ) as response:
            if response.status != 200:
                raise ValueError(
                    f"ActiveFence call failed with status code {response.status}.\n"
                    f"Details: {await response.text()}"
                )
            response_json = await response.json()
            log.info(json.dumps(response_json, indent=True))
            violations = response_json["violations"]

            violations_dict = {}
            max_risk_score = 0.0
            for violation in violations:
                if violation["risk_score"] > max_risk_score:
                    max_risk_score = violation["risk_score"]
                violations_dict[violation["violation_type"]] = violation["risk_score"]

            return {"max_risk_score": max_risk_score, "violations": violations_dict}
