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

from nemoguardrails import RailsConfig
from nemoguardrails.rails.llm.options import GenerationOptions
from tests.utils import TestChat

CONFIGS_FOLDER = os.path.join(os.path.dirname(__file__), ".", "test_configs")

OPTIONS = GenerationOptions(
    log={
        "activated_rails": True,
        "llm_calls": True,
        "internal_events": True,
        "colang_history": False,
    }
)


@pytest.mark.asyncio
async def test_parallel_rails_success():
    # Test 1 - All input/output rails pass
    config = RailsConfig.from_path(os.path.join(CONFIGS_FOLDER, "parallel_rails"))
    chat = TestChat(
        config,
        llm_completions=[
            "No",
            "Hi there! How can I assist you with questions about the ABC Company today?",
            "No",
        ],
    )

    chat >> "hi"
    result = await chat.app.generate_async(messages=chat.history, options=OPTIONS)

    # Assert the response is correct
    assert (
        result
        and result.response[0]["content"]
        == "Hi there! How can I assist you with questions about the ABC Company today?"
    )

    # Check that all rails were executed
    assert result.log.activated_rails[0].name == "self check input"
    assert (
        result.log.activated_rails[1].name == "check blocked input terms $duration=1.0"
    )
    assert (
        result.log.activated_rails[2].name == "check blocked input terms $duration=1.0"
    )
    assert result.log.activated_rails[3].name == "generate user intent"
    assert result.log.activated_rails[4].name == "self check output"
    assert (
        result.log.activated_rails[5].name == "check blocked output terms $duration=1.0"
    )
    assert (
        result.log.activated_rails[6].name == "check blocked output terms $duration=1.0"
    )

    # Time should be close to 2 seconds due to parallel processing:
    # check blocked input terms: 1s
    # check blocked output terms: 1s
    assert (
        result.log.stats.input_rails_duration < 1.5
        and result.log.stats.output_rails_duration < 1.5
    ), "Rails processing took too long, parallelization seems to be not working."


@pytest.mark.asyncio
async def test_parallel_rails_input_fail_1():
    # Test 2 - First input rail fails
    config = RailsConfig.from_path(os.path.join(CONFIGS_FOLDER, "parallel_rails"))
    chat = TestChat(
        config,
        llm_completions=[
            "Yes",
            "Hi there! How can I assist you with questions about the ABC Company today?",
            "No",
        ],
    )
    chat >> "hi, I am a unicorn!"
    await chat.bot_async("I'm sorry, I can't respond to that.")


@pytest.mark.asyncio
async def test_parallel_rails_input_fail_2():
    # Test 3 - Second input rail fails due to blocked term
    config = RailsConfig.from_path(os.path.join(CONFIGS_FOLDER, "parallel_rails"))
    chat = TestChat(
        config,
        llm_completions=[
            "No",
            "Hi there! How can I assist you with questions about the ABC Company today?",
            "No",
        ],
    )

    chat >> "hi, this is a blocked term."
    await chat.bot_async("I cannot process a term in the user message.")


@pytest.mark.asyncio
async def test_parallel_rails_output_fail_1():
    # Test 4 - First output rail fails
    config = RailsConfig.from_path(os.path.join(CONFIGS_FOLDER, "parallel_rails"))
    chat = TestChat(
        config,
        llm_completions=[
            "No",
            "Hi there! I am a unicorn!",
            "Yes",
        ],
    )

    chat >> "hi!"
    await chat.bot_async("I'm sorry, I can't respond to that.")


@pytest.mark.asyncio
async def test_parallel_rails_output_fail_2():
    # Test 4 - Second output rail fails due to blocked term
    config = RailsConfig.from_path(os.path.join(CONFIGS_FOLDER, "parallel_rails"))
    chat = TestChat(
        config,
        llm_completions=[
            "No",
            "Hi there! This is a blocked term!",
            "No",
        ],
    )

    chat >> "hi!"
    result = await chat.app.generate_async(messages=chat.history, options=OPTIONS)
    assert (
        result
        and result.response[0]["content"]
        == "I cannot express a term in the bot answer."
    )
