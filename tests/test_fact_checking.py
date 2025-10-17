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
from nemoguardrails.llm.providers.trtllm import llm
from tests.utils import TestChat

CONFIGS_FOLDER = os.path.join(os.path.dirname(__file__), ".", "test_configs")


def build_kb():
    with open(os.path.join(CONFIGS_FOLDER, "fact_checking", "kb", "kb.md"), "r") as f:
        content = f.readlines()

    return content


@action(is_system_action=True)
async def retrieve_relevant_chunks():
    """Retrieve relevant chunks from the knowledge base and add them to the context."""
    context_updates = {}
    relevant_chunks = "\n".join(build_kb())
    context_updates["relevant_chunks"] = relevant_chunks

    return ActionResult(
        return_value=context_updates["relevant_chunks"],
        context_updates=context_updates,
    )


@pytest.mark.asyncio
async def test_fact_checking_greeting(httpx_mock):
    # Test 1 - Greeting - No fact-checking invocation should happen
    config = RailsConfig.from_path(os.path.join(CONFIGS_FOLDER, "fact_checking"))
    chat = TestChat(
        config, llm_completions=["  express greeting", "Hi! How can I assist today?"]
    )
    chat.app.register_action(retrieve_relevant_chunks, "retrieve_relevant_chunks")

    chat >> "hi"
    await chat.bot_async("Hi! How can I assist today?")


@pytest.mark.asyncio
async def test_fact_checking_correct(httpx_mock):
    # Test 2 - Factual statement - high alignscore
    config = RailsConfig.from_path(os.path.join(CONFIGS_FOLDER, "fact_checking"))
    chat = TestChat(
        config,
        llm_completions=[
            "  ask about guardrails",
            "NeMo Guardrails is an open-source toolkit for easily adding programmable guardrails to LLM-based conversational systems.",
        ],
    )
    chat.app.register_action(retrieve_relevant_chunks, "retrieve_relevant_chunks")

    with aioresponses() as m:
        # Fact-checking using AlignScore
        m.post(
            "http://localhost:5000/alignscore_base",
            payload={"alignscore": 0.82},
        )

        # Succeeded, no more generations needed
        chat >> "What is NeMo Guardrails?"

        await chat.bot_async(
            "NeMo Guardrails is an open-source toolkit for easily adding programmable guardrails to LLM-based conversational systems."
        )


@pytest.mark.asyncio
async def test_fact_checking_wrong(httpx_mock):
    # Test 3 - Very low alignscore - Not factual
    config = RailsConfig.from_path(os.path.join(CONFIGS_FOLDER, "fact_checking"))
    chat = TestChat(
        config,
        llm_completions=[
            "  ask about guardrails",
            "NeMo Guardrails is a closed-source proprietary toolkit by Nvidia.",
        ],
    )
    chat.app.register_action(retrieve_relevant_chunks, "retrieve_relevant_chunks")

    with aioresponses() as m:
        # Fact-checking using AlignScore
        m.post(
            "http://localhost:5000/alignscore_base",
            payload={"alignscore": 0.01},
        )

        chat >> "What is NeMo Guardrails?"

        await chat.bot_async("I don't know the answer to that.")


@pytest.mark.asyncio
async def test_fact_checking_fallback_to_self_check_correct(httpx_mock):
    # Test 4 - Factual statement - AlignScore endpoint not set up properly, use ask llm for fact-checking
    config = RailsConfig.from_path(os.path.join(CONFIGS_FOLDER, "fact_checking"))
    chat = TestChat(
        config,
        llm_completions=[
            "  ask about guardrails",
            "NeMo Guardrails is an open-source toolkit for easily adding programmable guardrails to LLM-based conversational systems.",
            "yes",
        ],
    )

    chat.app.register_action(retrieve_relevant_chunks, "retrieve_relevant_chunks")

    with aioresponses() as m:
        # Fact-checking using AlignScore
        m.post(
            "http://localhost:5000/alignscore_base",
            payload="API error 404",
        )
        chat >> "What is NeMo Guardrails?"

        await chat.bot_async(
            "NeMo Guardrails is an open-source toolkit for easily adding programmable guardrails to LLM-based conversational systems."
        )


@pytest.mark.asyncio
async def test_fact_checking_fallback_self_check_wrong(httpx_mock):
    # Test 5 - Factual statement - AlignScore endpoint not set up properly, use ask llm for fact-checking
    config = RailsConfig.from_path(os.path.join(CONFIGS_FOLDER, "fact_checking"))
    chat = TestChat(
        config,
        llm_completions=[
            "  ask about guardrails",
            "NeMo Guardrails is an closed-source toolkit for easily adding programmable guardrails to LLM-based conversational systems.",
            "no",
            "I don't know the answer to that.",
        ],
    )
    chat.app.register_action(retrieve_relevant_chunks, "retrieve_relevant_chunks")

    with aioresponses() as m:
        # Fact-checking using AlignScore
        m.post(
            "http://localhost:5000/alignscore_base",
            payload="API error 404",
        )

        chat >> "What is NeMo Guardrails?"
        await chat.bot_async("I don't know the answer to that.")
