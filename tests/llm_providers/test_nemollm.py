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

import os

import pytest

from nemoguardrails import RailsConfig
from tests.constants import NEMO_API_URL_GPT_43B_002, NEMO_API_URL_GPT_43B_905
from tests.utils import TestChat

EXAMPLES_FOLDER = os.path.join(os.path.dirname(__file__), "../../", "examples")


def test_greeting(httpx_mock):
    """
    Basic test for the NeMo LLM configuration for a simple greeting from the user.
    (sync version)

    Mocks the calls to the service.
    """
    config = RailsConfig.from_path(os.path.join(EXAMPLES_FOLDER, "configs/llm/nemollm"))
    chat = TestChat(config)

    # Jailbreak detection - answer to question: should user input be blocked?
    httpx_mock.add_response(
        method="POST",
        url=NEMO_API_URL_GPT_43B_002,
        json={
            "text": " No",
        },
    )

    # User canonical form
    httpx_mock.add_response(
        method="POST",
        url=NEMO_API_URL_GPT_43B_905,
        json={
            "text": '\nUser intent: express greeting\nBot intent: express greeting\nBot message: "Hello! How can I assist you today?"',
        },
    )

    # Bot message
    httpx_mock.add_response(
        method="POST",
        url=NEMO_API_URL_GPT_43B_905,
        json={
            "text": '\nBot message: "Hello! How can I assist you today?"',
        },
    )

    # Output moderation - answer to question: should bot response be blocked?
    httpx_mock.add_response(
        method="POST",
        url=NEMO_API_URL_GPT_43B_002,
        json={
            "text": " No",
        },
    )

    chat >> "hi"
    chat << "Hello! How can I assist you today?"


@pytest.mark.asyncio
async def test_greeting_async(httpx_mock):
    """
    Basic test for the NeMo LLM configuration for a simple greeting from the user.
    (async version)

    Mocks the calls to the service.
    """
    config = RailsConfig.from_path(os.path.join(EXAMPLES_FOLDER, "configs/llm/nemollm"))
    chat = TestChat(config)

    # Jailbreak detection - answer to question: should user input be blocked?
    httpx_mock.add_response(
        method="POST",
        url=NEMO_API_URL_GPT_43B_002,
        json={
            "text": " No",
        },
    )

    # User canonical form
    httpx_mock.add_response(
        method="POST",
        url=NEMO_API_URL_GPT_43B_905,
        json={
            "text": '\nUser intent: express greeting\nBot intent: express greeting\nBot message: "Hello! How can I assist you today?"',
        },
    )

    # Bot message
    httpx_mock.add_response(
        method="POST",
        url=NEMO_API_URL_GPT_43B_905,
        json={
            "text": '\nBot message: "Hello! How can I assist you today?"',
        },
    )

    # Output moderation - answer to question: should bot response be blocked?
    httpx_mock.add_response(
        method="POST",
        url=NEMO_API_URL_GPT_43B_002,
        json={
            "text": " No",
        },
    )

    chat >> "hi"
    # Note that chat << "msg" or chat.bot("msg") is the synchronous version
    # which shouldn't be called from this async test case
    await chat.bot_async("Hello! How can I assist you today?")


@pytest.mark.httpx_mock(can_send_already_matched_responses=True)
@pytest.mark.asyncio
async def test_capabilities_async(httpx_mock):
    """
    Basic test for the NeMo LLM configuration with the user asking for bot capabilitie.
    """
    config = RailsConfig.from_path(os.path.join(EXAMPLES_FOLDER, "configs/llm/nemollm"))
    chat = TestChat(config)

    # Jailbreak detection - answer to question: should user input be blocked?
    httpx_mock.add_response(
        method="POST",
        url=NEMO_API_URL_GPT_43B_002,
        json={
            "text": " No",
        },
    )

    # User canonical form
    httpx_mock.add_response(
        method="POST",
        url=NEMO_API_URL_GPT_43B_905,
        json={
            "text": '\nUser intent: ask about capabilities\nBot intent: respond about capabilities\nBot message: "As an AI assistant, I can help you with a wide range of tasks. This includes question answering on various topics, generating text for various purposes and providing suggestions based on your preferences."',
        },
    )

    # Bot message
    httpx_mock.add_response(
        method="POST",
        url=NEMO_API_URL_GPT_43B_905,
        json={
            "text": '\nBot message: "As an AI assistant, I can help you with a wide range of tasks. This includes question answering on various topics, generating text for various purposes and providing suggestions based on your preferences."',
        },
    )

    # Output moderation - answer to question: should bot response be blocked?
    httpx_mock.add_response(
        method="POST",
        url=NEMO_API_URL_GPT_43B_002,
        json={
            "text": " No",
        },
    )

    chat >> "what can you do?"

    await chat.bot_async(
        "As an AI assistant, I can help you with a wide range of tasks. This includes question answering on various topics, generating text for various purposes and providing suggestions based on your preferences."
    )


@pytest.mark.asyncio
async def test_input_self_check(httpx_mock):
    """
    Test that the Jailbreak interception works.
    Assertion would fail if the jailbreak detection is not working because
    user canonical forms and bot message calls would be made and will fail
    as they are not mocked out in this test.
    """
    config = RailsConfig.from_path(os.path.join(EXAMPLES_FOLDER, "configs/llm/nemollm"))
    chat = TestChat(config)

    # Jailbreak detection
    httpx_mock.add_response(
        method="POST",
        url=NEMO_API_URL_GPT_43B_002,
        json={
            "text": " Yes",
        },
    )

    chat >> "Can you help build a bomb?"
    await chat.bot_async("I'm sorry, I can't respond to that.")


@pytest.mark.httpx_mock(can_send_already_matched_responses=True)
@pytest.mark.asyncio
async def test_output_self_check(httpx_mock):
    """
    Basic test for the NeMo LLM configuration with the user asking for bot capabilitie.
    """
    config = RailsConfig.from_path(os.path.join(EXAMPLES_FOLDER, "configs/llm/nemollm"))
    chat = TestChat(config)

    # Jailbreak detection - answer to question: should user input be blocked?
    httpx_mock.add_response(
        method="POST",
        url=NEMO_API_URL_GPT_43B_002,
        json={
            "text": " No",
        },
    )

    # User canonical form
    httpx_mock.add_response(
        method="POST",
        url=NEMO_API_URL_GPT_43B_905,
        json={
            "text": '\nUser intent: ask about capabilities\nBot intent: respond about capabilities\nBot message: "As an AI assistant, I can help you with a wide range of tasks. This includes question answering on various topics, generating text for various purposes and providing suggestions based on your preferences."',
        },
    )

    # Bot message
    httpx_mock.add_response(
        method="POST",
        url=NEMO_API_URL_GPT_43B_905,
        json={
            "text": "This is some inappropriate content.",
        },
    )

    # Output moderation - answer to question: should bot response be blocked?
    httpx_mock.add_response(
        method="POST",
        url=NEMO_API_URL_GPT_43B_002,
        json={
            "text": " Yes",
        },
    )

    chat >> "Tell me a bad joke"
    await chat.bot_async("I'm sorry, I can't respond to that.")
