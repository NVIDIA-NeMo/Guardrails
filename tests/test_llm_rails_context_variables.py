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
import asyncio

import pytest

from nemoguardrails import RailsConfig
from tests.utils import TestChat


@pytest.mark.asyncio
async def test_1():
    config = RailsConfig.from_content(
        """
        define user express greeting
            "hello"

        define flow
            user express greeting
            bot express greeting
        """
    )
    chat = TestChat(
        config,
        llm_completions=[
            "express greeting",
            "Hello! I'm doing great, thank you. How can I assist you today?",
        ],
    )

    new_messages = await chat.app.generate_async(messages=[{"role": "user", "content": "hi, how are you"}])

    assert new_messages == {
        "content": "Hello! I'm doing great, thank you. How can I assist you today?",
        "role": "assistant",
    }, "message content do not match"

    # note that 2 llm call are expected as we matched the bot intent
    assert len(chat.app.explain().llm_calls) == 2, "number of llm call not as expected. Expected 2, found {}".format(
        len(chat.app.explain().llm_calls)
    )


@pytest.mark.asyncio
async def test_2():
    config = RailsConfig.from_content(
        config={
            "models": [],
            "rails": {
                "output": {
                    # run the real self check output rails
                    "flows": {"self check output"},
                    "streaming": {
                        "enabled": True,
                        "chunk_size": 4,
                        "context_size": 2,
                        "stream_first": False,
                    },
                }
            },
            "streaming": False,
            "prompts": [{"task": "self_check_output", "content": "a test template"}],
        },
        colang_content="""
        define user express greeting
          "hi"

        define flow
          user express greeting
          bot tell joke
        """,
    )

    llm_completions = [
        '  express greeting\nbot express greeting\n  "Hi, how are you doing?"',
        '  "This is a joke that should be blocked."',
        # add as many `no`` as chunks you want the output stream to check
        "No",
        "No",
        "Yes",
    ]

    chat = TestChat(
        config,
        llm_completions=llm_completions,
        streaming=True,
    )
    chunks = []
    async for chunk in chat.app.stream_async(
        messages=[{"role": "user", "content": "Hi!"}],
    ):
        chunks.append(chunk)

    # note that 6 llm call are expected as we matched the bot intent
    assert len(chat.app.explain().llm_calls) == 5, "number of llm call not as expected. Expected 5, found {}".format(
        len(chat.app.explain().llm_calls)
    )

    await asyncio.gather(*asyncio.all_tasks() - {asyncio.current_task()})
