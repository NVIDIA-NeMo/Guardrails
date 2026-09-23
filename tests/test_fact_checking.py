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

import json
import os

import pytest

from nemoguardrails import RailsConfig
from nemoguardrails.actions.actions import ActionResult, action
from nemoguardrails.http import HTTPResponse
from nemoguardrails.library.factchecking.align_score.request import alignscore_request
from nemoguardrails.testing import RecordingHTTPClient
from tests.utils import TestChat

CONFIGS_FOLDER = os.path.join(os.path.dirname(__file__), ".", "test_configs")


def _response(payload, *, status: int = 200) -> HTTPResponse:
    return HTTPResponse(
        status_code=status,
        headers={"content-type": "application/json"},
        content=json.dumps(payload).encode(),
    )


@pytest.mark.asyncio
async def test_alignscore_request_uses_shared_client():
    client = RecordingHTTPClient([_response({"alignscore": 0.82})])

    result = await alignscore_request(
        api_url="http://alignscore.example/score",
        evidence=["NeMo Guardrails is open source."],
        response="NeMo Guardrails is open source.",
        http_client=client,
    )

    assert result == 0.82
    request = client.requests[0]
    assert request.method == "POST"
    assert request.url == "http://alignscore.example/score"
    assert request.json == {
        "evidence": ["NeMo Guardrails is open source."],
        "claim": "NeMo Guardrails is open source.",
    }


@pytest.mark.asyncio
async def test_alignscore_request_returns_one_without_evidence():
    client = RecordingHTTPClient()

    assert await alignscore_request(evidence=[], http_client=client) == 1.0
    assert client.requests == []


@pytest.mark.asyncio
async def test_alignscore_request_returns_none_for_non_200():
    client = RecordingHTTPClient([_response({}, status=503)])

    result = await alignscore_request(evidence=["evidence"], response="claim", http_client=client)

    assert result is None


@pytest.mark.asyncio
async def test_alignscore_request_returns_none_without_score():
    client = RecordingHTTPClient([_response({"result": "unknown"})])

    result = await alignscore_request(evidence=["evidence"], response="claim", http_client=client)

    assert result is None


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
    chat = TestChat(config, llm_completions=["  express greeting", "Hi! How can I assist today?"])
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

    # Fact-checking using AlignScore
    httpx_mock.add_response(
        method="POST",
        url="http://localhost:5000/alignscore_base",
        json={"alignscore": 0.82},
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

    # Fact-checking using AlignScore
    httpx_mock.add_response(
        method="POST",
        url="http://localhost:5000/alignscore_base",
        json={"alignscore": 0.01},
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

    # Fact-checking using AlignScore
    httpx_mock.add_response(
        method="POST",
        url="http://localhost:5000/alignscore_base",
        json="API error 404",
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

    # Fact-checking using AlignScore
    httpx_mock.add_response(
        method="POST",
        url="http://localhost:5000/alignscore_base",
        json="API error 404",
    )

    chat >> "What is NeMo Guardrails?"
    await chat.bot_async("I don't know the answer to that.")
