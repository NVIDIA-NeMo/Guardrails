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

from nemoguardrails.actions import action
from nemoguardrails.actions.rail_outcome import RailOutcome

log = logging.getLogger(__name__)

CLEANLAB_TRUSTWORTHINESS_THRESHOLD = 0.6


def _cleanlab_outcome(trustworthiness_score: float) -> RailOutcome:
    if trustworthiness_score < CLEANLAB_TRUSTWORTHINESS_THRESHOLD:
        return RailOutcome.block(metadata={"trustworthiness_score": trustworthiness_score})
    return RailOutcome.allow(metadata={"trustworthiness_score": trustworthiness_score})


@action(
    name="call_cleanlab_api",
    is_system_action=True,
)
async def call_cleanlab_api(
    context: Optional[dict] = None,
    user_message: Optional[str] = None,
    bot_message: Optional[str] = None,
    **kwargs,
) -> RailOutcome:
    api_key = os.environ.get("CLEANLAB_API_KEY")

    if api_key is None:
        raise ValueError("CLEANLAB_API_KEY environment variable not set.")

    try:
        from cleanlab_studio import Studio  # type: ignore[reportMissingImports]
    except ImportError:
        raise ImportError("Please install cleanlab-studio using 'pip install --upgrade cleanlab-studio' command")

    context = context or {}
    bot_response = bot_message if bot_message is not None else context.get("bot_message")
    user_input = user_message if user_message is not None else context.get("user_message")

    studio = Studio(api_key)
    cleanlab_tlm = studio.TLM()

    if bot_response:
        trustworthiness_result = await cleanlab_tlm.get_trustworthiness_score_async(user_input, response=bot_response)
        trustworthiness_score = trustworthiness_result["trustworthiness_score"]
    else:
        raise ValueError("Cannot compute trustworthiness score without a valid response from the LLM")

    log.info(f"Trustworthiness Score: {trustworthiness_score}")
    return _cleanlab_outcome(trustworthiness_score)
