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
from typing import Optional
from urllib import parse

import aiohttp

from nemoguardrails.actions import action
from nemoguardrails.actions.actions import ActionResult
from nemoguardrails.utils import new_event_dict

log = logging.getLogger(__name__)

APP_ID = os.environ.get("WOLFRAM_ALPHA_APP_ID")
API_URL_BASE = f"https://api.wolframalpha.com/v2/result?appid={APP_ID}"


@action(name="wolfram alpha request")
async def wolfram_alpha_request(query: Optional[str] = None, context: Optional[dict] = None):
    """Makes a request to the Wolfram Alpha API.

    Args:
        query (Optional[str]): The query for Wolfram Alpha. Defaults to None.
        context (Optional[dict]): The context for the execution of the action. Defaults to None.

    Returns:
        ActionResult or str: The result of the Wolfram Alpha request.

    Raises:
        Exception: If no query is provided to Wolfram Alpha.
    """
    # If we don't have an explicit query, we take the last user message
    if query is None and context is not None:
        query = context.get("last_user_message") or "2+3"

    if query is None:
        raise Exception("No query was provided to Wolfram Alpha.")

    if APP_ID is None:
        return ActionResult(
            return_value=False,
            events=[
                new_event_dict("BotIntent", intent="inform wolfram alpha app id not set"),
                new_event_dict(
                    "StartUtteranceBotAction",
                    script="Wolfram Alpha app ID is not set. Please set the WOLFRAM_ALPHA_APP_ID environment variable.",
                ),
                new_event_dict("BotIntent", intent="stop"),
            ],
        )

    url = API_URL_BASE + "&" + parse.urlencode({"i": query})

    log.info(f"Wolfram Alpha: executing request for: {query}")

    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            if resp.status != 200:
                log.info(f"Wolfram Alpha request failed : {query}")
                return ActionResult(
                    return_value=False,
                    events=[
                        new_event_dict("BotIntent", intent="inform wolfram alpha not working"),
                        new_event_dict(
                            "StartUtteranceBotAction",
                            script="Apologies, but I cannot answer this question at this time. I am having trouble getting the answer from Wolfram Alpha.",
                        ),
                        new_event_dict("BotIntent", intent="stop"),
                    ],
                )

            result = await resp.text()

            log.info(f"Wolfram Alpha: the result was {result}.")
            return result
