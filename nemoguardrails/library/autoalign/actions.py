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

import copy
import json
import logging
import os
from typing import Any, Dict, List, Optional

import aiohttp

from nemoguardrails.actions import action
from nemoguardrails.llm.taskmanager import LLMTaskManager

log = logging.getLogger(__name__)

GUARDRAIL_RESPONSE_TEXT = {
    "confidential_info_detection": "Confidential Information violation",
    "gender_bias_detection": "Stereotypical bias",
    "harm_detection": "Potential harm to human",
    "toxicity_detection": "Toxicity in text",
    "tonal_detection": "Negative tone",
    "racial_bias_detection": "Stereotypical bias",
    "jailbreak_detection": "Jailbreak attempt",
    "intellectual_property": "Intellectual property",
}

DEFAULT_CONFIG = {
    "pii": {
        "mode": "OFF",
        "mask": False,
        "enabled_types": [
            "[BANK ACCOUNT NUMBER]",
            "[CREDIT CARD NUMBER]",
            "[DATE OF BIRTH]",
            "[DATE]",
            "[DRIVER LICENSE NUMBER]",
            "[EMAIL ADDRESS]",
            "[RACE/ETHNICITY]",
            "[GENDER]",
            "[IP ADDRESS]",
            "[LOCATION]",
            "[MONEY]",
            "[ORGANIZATION]",
            "[PASSPORT NUMBER]",
            "[PASSWORD]",
            "[PERSON NAME]",
            "[PHONE NUMBER]",
            "[PROFESSION]",
            "[SOCIAL SECURITY NUMBER]",
            "[USERNAME]",
            "[SECRET_KEY]",
            "[TRANSACTION_ID]",
            "[RELIGION]",
        ],
    },
    "confidential_info_detection": {"mode": "OFF"},
    "gender_bias_detection": {"mode": "OFF"},
    "harm_detection": {"mode": "OFF"},
    "toxicity_detection": {"mode": "OFF"},
    "racial_bias_detection": {"mode": "OFF"},
    "tonal_detection": {"mode": "OFF"},
    "jailbreak_detection": {"mode": "OFF"},
    "intellectual_property": {"mode": "OFF"},
}

default_groundedness_config = {"groundedness_checker": {"verify_response": False}}


def autoalign_output_api_mapping(result: dict) -> bool:
    """
    Mapping for autoalign_output_api.

    Expects result to be a dict with a key "guardrails_triggered" (a boolean).
    Returns True (block) if guardrails were triggered.
    """
    return result.get("guardrails_triggered", False)


def autoalign_groundedness_output_api_mapping(result: float) -> bool:
    """
    Mapping for autoalign_groundedness_output_api.

    Expects result to be a numeric score.
    Returns True (block) if the score is below the default groundedness threshold.
    """
    DEFAULT_GROUNDEDNESS_THRESHOLD = 0.5
    return result < DEFAULT_GROUNDEDNESS_THRESHOLD


def autoalign_factcheck_output_api_mapping(result: float) -> bool:
    """
    Mapping for autoalign_factcheck_output_api.

    Expects result to be a numeric score.
    Returns True (block) if the score is below the default factcheck threshold.
    """
    DEFAULT_FACTCHECK_THRESHOLD = 0.5
    return result < DEFAULT_FACTCHECK_THRESHOLD


def process_autoalign_output(responses: List[Any], show_toxic_phrases: bool = False):
    """Processes the output provided AutoAlign API"""

    response_dict = {
        "guardrails_triggered": False,
        "pii": {"guarded": False, "response": ""},
    }
    prefixes = set()
    suffix = ""
    for response in responses:
        if response["guarded"]:
            if response["task"] == "toxicity_detection":
                response_dict["guardrails_triggered"] = True
                prefixes.add(GUARDRAIL_RESPONSE_TEXT[response["task"]])
                suffix = " Toxic phrases: " + ", ".join(response["output_data"])
                response_dict[response["task"]] = {
                    "guarded": True,
                    "response": [GUARDRAIL_RESPONSE_TEXT[response["task"]], suffix],
                }
            elif response["task"] == "pii":
                start_index = len("PII redacted text: ")
                response_dict["pii"] = {
                    "guarded": True,
                    "response": response["response"][start_index:],
                }
            else:
                prefixes.add(GUARDRAIL_RESPONSE_TEXT[response["task"]])
                response_dict["guardrails_triggered"] = True
                response_dict[response["task"]] = {
                    "guarded": True,
                    "response": GUARDRAIL_RESPONSE_TEXT[response["task"]],
                }
        else:
            response_dict[response["task"]] = {"guarded": False, "response": ""}

    response_dict["combined_response"] = ""
    if len(prefixes) > 0:
        response_dict["combined_response"] = ", ".join(prefixes) + " detected."
        if (
            "toxicity_detection" in response_dict.keys()
            and response_dict["toxicity_detection"]["guarded"]
            and show_toxic_phrases
        ):
            response_dict["combined_response"] += suffix
    return response_dict


async def autoalign_infer(
    request_url: str,
    text: str,
    task_config: Optional[Dict[Any, Any]] = None,
    show_toxic_phrases: bool = False,
    multi_language: bool = False,
):
    """Checks whether the given text passes through the applied guardrails."""
    api_key = os.environ.get("AUTOALIGN_API_KEY")
    if api_key is None:
        raise ValueError("AUTOALIGN_API_KEY environment variable not set.")

    headers = {"x-api-key": api_key}
    config = copy.deepcopy(DEFAULT_CONFIG)
    # enable the select guardrail
    for task in task_config.keys():
        if task != "factcheck":
            config[task]["mode"] = "DETECT"
        if task_config[task]:
            config[task].update(task_config[task])
    request_body = {"prompt": text, "config": config, "multi_language": multi_language}

    guardrails_configured = []

    async with aiohttp.ClientSession() as session:
        async with session.post(
            url=request_url,
            headers=headers,
            json=request_body,
        ) as response:
            if response.status != 200:
                raise ValueError(
                    f"AutoAlign call failed with status code {response.status}.\n"
                    f"Details: {await response.text()}"
                )
            async for line in response.content:
                line_text = line.strip()
                if len(line_text) > 0:
                    resp = json.loads(line_text)
                    guardrails_configured.append(resp)
            processed_response = process_autoalign_output(
                guardrails_configured, show_toxic_phrases
            )
    return processed_response


async def autoalign_groundedness_infer(
    request_url: str,
    text: str,
    documents: List[str],
    guardrails_config: Optional[Dict[Any, Any]] = None,
):
    """Checks the groundedness for the text using the given documents and provides a fact-checking score"""
    groundness_config = copy.deepcopy(default_groundedness_config)
    api_key = os.environ.get("AUTOALIGN_API_KEY")
    if api_key is None:
        raise ValueError("AUTOALIGN_API_KEY environment variable not set.")
    headers = {"x-api-key": api_key}
    if guardrails_config:
        groundness_config.update(guardrails_config)
    request_body = {"prompt": text, "documents": documents, "config": groundness_config}
    async with aiohttp.ClientSession() as session:
        async with session.post(
            url=request_url,
            headers=headers,
            json=request_body,
        ) as response:
            if response.status != 200:
                raise ValueError(
                    f"AutoAlign call failed with status code {response.status}.\n"
                    f"Details: {await response.text()}"
                )
            async for line in response.content:
                resp = json.loads(line)
                if resp["task"] == "groundedness_checker":
                    if resp["response"].startswith("Factcheck Score: "):
                        return float(resp["response"][17:])
    return 1.0


async def autoalign_factcheck_infer(
    request_url: str,
    user_message: str,
    bot_message: str,
    guardrails_config: Optional[Dict[str, Any]] = None,
    multi_language: bool = False,
):
    api_key = os.environ.get("AUTOALIGN_API_KEY")
    if api_key is None:
        raise ValueError("AUTOALIGN_API_KEY environment variable not set.")
    headers = {"x-api-key": api_key}
    request_body = {
        "labels": {"session_id": "nemo", "api_key": api_key},
        "content_moderation_docs": [
            {
                "content": bot_message,
                "type": "text_content",
                "file_name": "HighlightedText.txt",
            }
        ],
        "user_query": user_message,
        "config": guardrails_config,
        "multi_language": multi_language,
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(
            url=request_url,
            headers=headers,
            json=request_body,
        ) as response:
            if response.status != 200:
                raise ValueError(
                    f"AutoAlign call failed with status code {response.status}.\n"
                    f"Details: {await response.text()}"
                )
            factcheck_response = await response.json()
            return factcheck_response["all_overall_fact_scores"][0]
    return 1.0


@action(name="autoalign_input_api")
async def autoalign_input_api(
    llm_task_manager: LLMTaskManager,
    context: Optional[dict] = None,
    show_autoalign_message: bool = True,
    show_toxic_phrases: bool = False,
    **kwargs,
):
    """Calls AutoAlign API for the user message and guardrail configuration provided"""
    user_message = context.get("user_message")
    autoalign_config = llm_task_manager.config.rails.config.autoalign
    autoalign_api_url = autoalign_config.parameters.get("endpoint")
    multi_language = autoalign_config.parameters.get("multi_language", False)
    if not autoalign_api_url:
        raise ValueError("Provide the autoalign endpoint in the config")
    task_config = getattr(autoalign_config.input, "guardrails_config")
    if not task_config:
        raise ValueError("Provide the guardrails and their configuration")
    text = user_message

    autoalign_response = await autoalign_infer(
        autoalign_api_url,
        text,
        task_config,
        show_toxic_phrases,
        multi_language=multi_language,
    )
    if autoalign_response["guardrails_triggered"] and show_autoalign_message:
        log.warning(
            f"AutoAlign on Input: {autoalign_response['combined_response']}",
        )
    else:
        if autoalign_response["pii"]["guarded"] and show_autoalign_message:
            log.warning(
                f"AutoAlign on Input: {autoalign_response['pii']['response']}",
            )

    return autoalign_response


@action(name="autoalign_output_api", output_mapping=autoalign_output_api_mapping)
async def autoalign_output_api(
    llm_task_manager: LLMTaskManager,
    context: Optional[dict] = None,
    show_autoalign_message: bool = True,
    show_toxic_phrases: bool = False,
    **kwargs,
):
    """Calls AutoAlign API for the bot message and guardrail configuration provided"""
    bot_message = context.get("bot_message")
    autoalign_config = llm_task_manager.config.rails.config.autoalign
    autoalign_api_url = autoalign_config.parameters.get("endpoint")
    multi_language = autoalign_config.parameters.get("multi_language", False)
    if not autoalign_api_url:
        raise ValueError("Provide the autoalign endpoint in the config")
    task_config = getattr(autoalign_config.output, "guardrails_config")
    if not task_config:
        raise ValueError("Provide the guardrails and their configuration")

    text = bot_message
    autoalign_response = await autoalign_infer(
        autoalign_api_url,
        text,
        task_config,
        show_toxic_phrases,
        multi_language=multi_language,
    )
    if autoalign_response["guardrails_triggered"] and show_autoalign_message:
        log.warning(
            f"AutoAlign on LLM Response: {autoalign_response['combined_response']}",
        )

    return autoalign_response


@action(
    name="autoalign_groundedness_output_api",
    output_mapping=autoalign_groundedness_output_api_mapping,
)
async def autoalign_groundedness_output_api(
    llm_task_manager: LLMTaskManager,
    context: Optional[dict] = None,
    factcheck_threshold: float = 0.0,
    show_autoalign_message: bool = True,
    **kwargs,
):
    """Calls AutoAlign groundedness check API and checks whether the bot message is factually grounded according to given
    documents"""

    bot_message = context.get("bot_message")
    documents = context.get("relevant_chunks_sep", [])

    autoalign_config = llm_task_manager.config.rails.config.autoalign
    autoalign_groundedness_api_url = autoalign_config.parameters.get(
        "groundedness_check_endpoint"
    )
    guardrails_config = getattr(autoalign_config.output, "guardrails_config", None)
    if not autoalign_groundedness_api_url:
        raise ValueError(
            "Provide the autoalign groundedness check endpoint in the config"
        )
    text = bot_message
    score = await autoalign_groundedness_infer(
        request_url=autoalign_groundedness_api_url,
        text=text,
        documents=documents,
        guardrails_config=guardrails_config,
    )
    if score < factcheck_threshold and show_autoalign_message:
        log.warning(
            f"Groundedness violation in llm response has been detected by AutoAlign with fact check score {score}"
        )
    return score


@action(
    name="autoalign_factcheck_output_api",
    output_mapping=autoalign_factcheck_output_api_mapping,
)
async def autoalign_factcheck_output_api(
    llm_task_manager: LLMTaskManager,
    context: Optional[dict] = None,
    factcheck_threshold: float = 0.0,
    show_autoalign_message: bool = True,
):
    """Calls Autoalign Factchecker API and checks if the user message is factually answered by the bot message"""

    user_message = context.get("user_message")
    bot_message = context.get("bot_message")
    autoalign_config = llm_task_manager.config.rails.config.autoalign
    autoalign_factcheck_api_url = autoalign_config.parameters.get("fact_check_endpoint")
    multi_language = autoalign_config.parameters.get("multi_language", False)

    guardrails_config = getattr(autoalign_config.output, "guardrails_config", None)
    if not autoalign_factcheck_api_url:
        raise ValueError("Provide the autoalign fact check endpoint in the config")
    score = await autoalign_factcheck_infer(
        request_url=autoalign_factcheck_api_url,
        user_message=user_message,
        bot_message=bot_message,
        guardrails_config=guardrails_config,
        multi_language=multi_language,
    )

    if score < factcheck_threshold and show_autoalign_message:
        log.warning(
            f"Factcheck violation in llm response has been detected by AutoAlign with fact check score {score}"
        )
    return score
