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
import os

import pytest
from aioresponses import aioresponses

from nemoguardrails import RailsConfig
from nemoguardrails.actions.actions import ActionResult, action
from tests.utils import TestChat

CONFIGS_FOLDER = os.path.join(os.path.dirname(__file__), ".", "test_configs")


@action(is_system_action=True)
async def retrieve_relevant_chunks():
    """Retrieve relevant chunks from the knowledge base and add them to the context."""
    context_updates = {"relevant_chunks": "Shipping takes at least 3 days."}

    return ActionResult(
        return_value=context_updates["relevant_chunks"],
        context_updates=context_updates,
    )


@pytest.mark.asyncio
async def test_fiddler_safety_rails(monkeypatch):
    # Mock environment variables
    monkeypatch.setenv("FIDDLER_API_KEY", "")

    config = RailsConfig.from_path(os.path.join(CONFIGS_FOLDER, "fiddler/safety"))
    chat = TestChat(
        config,
        llm_completions=[
            "user ask general question",  # user intent
            "Yes, shipping can be done in 2 days.",  # bot response that will be intercepted
        ],
    )
    with aioresponses() as m:
        m.post(
            "https://testfiddler.ai/v3/guardrails/ftl-safety",
            payload={
                "fdl_harmful": 0.5,
                "fdl_violent": 0.5,
                "fdl_unethical": 0.5,
                "fdl_illegal": 0.5,
                "fdl_sexual": 0.5,
                "fdl_racist": 0.5,
                "fdl_jailbreaking": 0.5,
                "fdl_harassing": 0.5,
                "fdl_hateful": 0.5,
                "fdl_sexist": 0.5,
            },
        )

        chat >> "Do you ship within 2 days?"
        await chat.bot_async("I'm sorry, I can't respond to that.")


@pytest.mark.asyncio
async def test_fiddler_safety_rails_pass(monkeypatch):
    # Mock environment variables
    monkeypatch.setenv("FIDDLER_ENVIRON", "https://testfiddler.ai")
    monkeypatch.setenv("FIDDLER_API_KEY", "")
    monkeypatch.setenv("FIDDLER_SAFETY_DETECTION_THRESHOLD", "0.8")

    config = RailsConfig.from_path(os.path.join(CONFIGS_FOLDER, "fiddler/safety"))
    chat = TestChat(
        config,
        llm_completions=[
            "user ask general question",  # user intent
            "Yes, shipping can be done in 2 days.",  # bot response that will be intercepted
        ],
    )

    with aioresponses() as m:
        m.post(
            "https://testfiddler.ai/v3/guardrails/ftl-safety",
            payload={
                "fdl_harmful": 0.02,
                "fdl_violent": 0.02,
                "fdl_unethical": 0.02,
                "fdl_illegal": 0.02,
                "fdl_sexual": 0.02,
                "fdl_racist": 0.02,
                "fdl_jailbreaking": 0.02,
                "fdl_harassing": 0.02,
                "fdl_hateful": 0.02,
                "fdl_sexist": 0.02,
            },
        )
        chat >> "Do you ship within 2 days?"
        await chat.bot_async("Yes, shipping can be done in 2 days.")


@pytest.mark.asyncio
async def test_fiddler_thresholds(monkeypatch):
    # Mock environment variables
    monkeypatch.setenv("FIDDLER_API_KEY", "")

    config = RailsConfig.from_path(os.path.join(CONFIGS_FOLDER, "fiddler/thresholds"))
    chat = TestChat(
        config,
        llm_completions=[
            "user ask general question",  # user intent
            "Yes, shipping can be done in 2 days.",  # bot response that will be intercepted
        ],
    )

    with aioresponses() as m:
        chat.app.register_action(retrieve_relevant_chunks, "retrieve_relevant_chunks")
        m.post(
            "https://testfiddler.ai/v3/guardrails/ftl-safety",
            payload={
                "fdl_harmful": 0.5,
                "fdl_violent": 0.5,
                "fdl_unethical": 0.5,
                "fdl_illegal": 0.5,
                "fdl_sexual": 0.5,
                "fdl_racist": 0.5,
                "fdl_jailbreaking": 0.5,
                "fdl_harassing": 0.5,
                "fdl_hateful": 0.5,
                "fdl_sexist": 0.5,
            },
        )
        chat >> "Do you ship within 2 days?"
        await chat.bot_async("I'm sorry, I can't respond to that.")


@pytest.mark.asyncio
async def test_fiddler_faithfulness_rails(monkeypatch):
    # Mock environment variables
    monkeypatch.setenv("FIDDLER_API_KEY", "")

    config = RailsConfig.from_path(os.path.join(CONFIGS_FOLDER, "fiddler/faithfulness"))
    chat = TestChat(
        config,
        llm_completions=[
            "user ask general question",  # user intent
            "Yes, shipping can be done in 2 days.",  # bot response that will be intercepted
        ],
    )

    with aioresponses() as m:
        chat.app.register_action(retrieve_relevant_chunks, "retrieve_relevant_chunks")
        m.post(
            "https://testfiddler.ai/v3/guardrails/ftl-response-faithfulness",
            payload={"fdl_faithful_score": 0.001},
        )
        chat >> "Do you ship within 2 days?"
        await chat.bot_async("I'm sorry, I can't respond to that.")


@pytest.mark.asyncio
async def test_fiddler_faithfulness_rails_pass(monkeypatch):
    # Mock environment variables
    monkeypatch.setenv("FIDDLER_API_KEY", "")

    config = RailsConfig.from_path(os.path.join(CONFIGS_FOLDER, "fiddler/faithfulness"))
    chat = TestChat(
        config,
        llm_completions=[
            "user ask general question",  # user intent
            "Yes, shipping can be done in 2 days.",  # bot response that will be intercepted
        ],
    )

    with aioresponses() as m:
        chat.app.register_action(retrieve_relevant_chunks, "retrieve_relevant_chunks")
        m.post(
            "https://testfiddler.ai/v3/guardrails/ftl-response-faithfulness",
            payload={"fdl_faithful_score": 0.5},
        )
        chat >> "Do you ship within 2 days?"
        await chat.bot_async("Yes, shipping can be done in 2 days.")
