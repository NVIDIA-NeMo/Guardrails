# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pytest

from nemoguardrails import RailsConfig
from tests.utils import TestChat

COLANG_CONTENT = """
    define user express greeting
        "hello"
        "hi"

    define flow greeting
        user express greeting
        $check_hallucination = True
        bot express greeting
"""

YAML_CONTENT = """
    models: []
    rails:
        output:
            flows:
                - self check hallucination
    prompts:
        - task: self_check_hallucination
          content: |
            You are given a task to identify if the hypothesis is in agreement with the context below.
            You will only use the contents of the context and not rely on external knowledge.
            Answer with yes/no. "context": {{ paragraph }} "hypothesis": {{ statement }} "agreement":

    enable_rails_exceptions: True
"""

config = RailsConfig.from_content(
    colang_content=COLANG_CONTENT,
    yaml_content=YAML_CONTENT,
)


@pytest.mark.asyncio
async def test_no_hallucination_detected():
    chat = TestChat(
        config,
        llm_completions=[
            "  express greeting",
            '  "Hello there!"',
            "Hello there!",
            "Hello there!",
            "yes",
        ],
    )
    messages = [{"role": "user", "content": "hi"}]
    result = await chat.app.generate_async(messages=messages)

    assert result["content"] == "Hello there!"


@pytest.mark.asyncio
async def test_hallucination_detected():
    chat = TestChat(
        config,
        llm_completions=[
            "  express greeting",
            '  "Hello there!"',
            "Something completely different",
            "Another different response",
            "no",
        ],
    )
    messages = [{"role": "user", "content": "hi"}]
    result = await chat.app.generate_async(messages=messages)

    assert result["role"] == "exception"
    assert result["content"]["type"] == "SelfCheckHallucinationRailException"


@pytest.mark.asyncio
async def test_hallucination_blocked_response():
    config_no_exceptions = RailsConfig.from_content(
        colang_content=COLANG_CONTENT,
        yaml_content="""
            models: []
            rails:
                output:
                    flows:
                        - self check hallucination
            prompts:
                - task: self_check_hallucination
                  content: |
                    Answer with yes/no. "context": {{ paragraph }} "hypothesis": {{ statement }} "agreement":
        """,
    )

    chat = TestChat(
        config_no_exceptions,
        llm_completions=[
            "  express greeting",
            '  "Hello there!"',
            "Something different",
            "Another different response",
            "no",
        ],
    )
    messages = [{"role": "user", "content": "hi"}]
    result = await chat.app.generate_async(messages=messages)

    assert result["content"] == "I don't know the answer to that."
