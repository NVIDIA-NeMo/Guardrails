# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from typing import Optional

try:
    from openai import OpenAI
except ImportError:
    pass


from nemoguardrails.actions import action

log = logging.getLogger(__name__)


def openai_text_moderation_mapping(result: dict) -> bool:
    """Determines whether the output from the OpenAI moderation API should be blocked.
    Returns True (i.e. block the output) if any of the conditions are met.
    """
    # Check overall risk score if available.
    flagged = result.get("results", [{}])[0].get("flagged", False)

    if flagged:
        return True

    confidence_threshold = 0.8
    scores = result["results"][0]["category_scores"]

    if any(score > confidence_threshold for score in scores.values()):
        return True

    # If none of the thresholds are exceeded, allow the output.
    return False


@action(
    name="openai_moderation_api",
    is_system_action=True,
    output_mapping=openai_text_moderation_mapping,
)
async def call_openai_text_moderation_api(
    context: Optional[dict] = None, **kwargs
) -> dict:
    try:
        from openai import OpenAI

    except ImportError:
        raise ImportError(
            "Could not import openai, please install it with " "`pip install openai`."
        )

    user_message = context.get("user_message")
    client = OpenAI()

    response = client.moderations.create(
        model="omni-moderation-latest",
        input=user_message,
    )

    result = response.model_dump()["results"][0]
    result["categories"] = dict(result.get("categories", {}))
    result["category_scores"] = {
        str(k): float(v) for k, v in result.get("category_scores", {}).items()
    }
    result["flagged"] = bool(result.get("flagged", False))

    return result


call_openai_text_moderation_api.action_meta.setdefault("tags", []).append("moderation")
