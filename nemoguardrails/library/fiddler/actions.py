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

import logging
import os
from typing import Callable, Optional

import aiohttp

from nemoguardrails import RailsConfig
from nemoguardrails.actions import action
from nemoguardrails.rails.llm.config import FiddlerGuardrails

log = logging.getLogger(__name__)


async def call_fiddler_guardrail(
    endpoint: str,
    data: dict,
    guardrail_name: str,
    score_key: str,
    threshold: float,
    compare: Callable[[float, float], bool],
    default_score: float,
) -> bool:
    api_key = os.environ.get("FIDDLER_API_KEY")

    if api_key is None:
        raise ValueError("FIDDLER_API_KEY environment variable not set.")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(endpoint, headers=headers, json={"data": data}) as response:
                if response.status != 200:
                    log.error(f"{guardrail_name} could not be run. Fiddler API returned status code {response.status}")
                    return False

                response_json = await response.json()
                if score_key == "safety":
                    detection_score = max(
                        response_json.get(key, default_score)
                        for key in [
                            "fdl_harmful",
                            "fdl_violent",
                            "fdl_unethical",
                            "fdl_illegal",
                            "fdl_sexual",
                            "fdl_racist",
                            "fdl_jailbreaking",
                            "fdl_harassing",
                            "fdl_hateful",
                            "fdl_sexist",
                        ]
                    )
                else:
                    detection_score = response_json.get(score_key, default_score)
                return compare(detection_score, threshold)
    except aiohttp.ClientError as e:
        log.error(f"{guardrail_name} request failed: {e}")
        return False
    except (KeyError, ValueError, IndexError) as e:
        log.error(f"Error processing {guardrail_name} response: {e}")
        return False


@action(name="call fiddler safety on user message", is_system_action=True)
async def call_fiddler_safety_user(config: RailsConfig, context: Optional[dict] = None):
    fiddler_config: FiddlerGuardrails = getattr(config.rails.config, "fiddler")
    base_url = fiddler_config.fiddler_endpoint

    if base_url is None:
        log.error("Fiddler endpoint not set in config")
        return False

    user_message = context.get("user_message", "")
    if not user_message:
        log.error("Fiddler Jailbreak Guardrails could not be run. User message must be provided.")
        return False

    data = {"prompt": [user_message]}
    return await call_fiddler_guardrail(
        endpoint=base_url + "/v3/guardrails/ftl-safety",
        data=data,
        guardrail_name="Fiddler Jailbreak Guardrails",
        score_key="safety",
        threshold=fiddler_config.safety_threshold,
        compare=lambda score, threshold: score >= threshold,
        default_score=0,
    )


@action(name="call fiddler safety on bot message", is_system_action=True)
async def call_fiddler_safety_bot(config: RailsConfig, context: Optional[dict] = None):
    fiddler_config: FiddlerGuardrails = getattr(config.rails.config, "fiddler")
    base_url = fiddler_config.fiddler_endpoint

    if base_url is None:
        log.error("Fiddler endpoint not set in config")
        return False

    bot_message = context.get("bot_message", "")
    if not bot_message:
        log.error("Fiddler Safety Guardrails could not be run. Bot message must be provided.")
        return False

    data = {"prompt": [bot_message]}
    return await call_fiddler_guardrail(
        endpoint=base_url + "/v3/guardrails/ftl-safety",
        data=data,
        guardrail_name="Fiddler Safety Guardrails",
        score_key="safety",
        threshold=fiddler_config.safety_threshold,
        compare=lambda score, threshold: score >= threshold,
        default_score=0,
    )


@action(name="call fiddler faithfulness", is_system_action=True)
async def call_fiddler_faithfulness(config: RailsConfig, context: Optional[dict] = None):
    fiddler_config: FiddlerGuardrails = getattr(config.rails.config, "fiddler")
    base_url = fiddler_config.fiddler_endpoint

    if base_url is None:
        log.error("Fiddler endpoint not set in config")
        return False

    bot_message = context.get("bot_message", "")
    knowledge = context.get("relevant_chunks", [])
    if not bot_message:
        log.error("Fiddler Faithfulness Guardrails could not be run. Chatbot message must be provided.")
        return False

    data = {"response": [bot_message], "context": [knowledge]}
    return await call_fiddler_guardrail(
        endpoint=base_url + "/v3/guardrails/ftl-response-faithfulness",
        data=data,
        guardrail_name="Fiddler Faithfulness Guardrails",
        score_key="fdl_faithful_score",
        threshold=fiddler_config.faithfulness_threshold,
        compare=lambda score, threshold: score <= threshold,
        default_score=1,
    )
