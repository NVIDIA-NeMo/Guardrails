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
from typing import Dict, Optional, Union

from nemoguardrails.actions import action

log = logging.getLogger(__name__)


def cleanlab_output_mapping(result: dict) -> bool:
    """Block if the trustworthiness score is below 0.6."""

    score = result.get("trustworthiness_score", 1)
    return score < 0.6


@action(
    name="call cleanlab api",
    is_system_action=True,
    output_mapping=cleanlab_output_mapping,
)
async def call_cleanlab_api(
    context: Optional[dict] = None,
    **kwargs,
) -> Union[ValueError, ImportError, Dict]:
    api_key = os.environ.get("CLEANLAB_API_KEY")

    if api_key is None:
        raise ValueError("CLEANLAB_API_KEY environment variable not set.")

    try:
        from cleanlab_studio import Studio
    except ImportError:
        raise ImportError(
            "Please install cleanlab-studio using 'pip install --upgrade cleanlab-studio' command"
        )

    bot_response = context.get("bot_message")
    user_input = context.get("user_message")

    studio = Studio(api_key)
    cleanlab_tlm = studio.TLM()

    if bot_response:
        trustworthiness_result = await cleanlab_tlm.get_trustworthiness_score_async(
            user_input, response=bot_response
        )
        trustworthiness_score = trustworthiness_result["trustworthiness_score"]
    else:
        raise ValueError(
            "Cannot compute trustworthiness score without a valid response from the LLM"
        )

    log.info(f"Trustworthiness Score: {trustworthiness_score}")
    return {"trustworthiness_score": trustworthiness_score}
