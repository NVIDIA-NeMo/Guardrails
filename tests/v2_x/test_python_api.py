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
import pytest

from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.rails.llm.options import GenerationResponse
from tests.utils import TestChat

config = RailsConfig.from_content(
    """
    import core

    flow math question
      $text = user said "what is 2+2"

      bot say "Let me think."
      $result = await WolframAlphaAction(query=$text)
      bot say $result

    flow greeting
      user said "hi"
      bot say "Hello!"
      user said "hi"
      bot say "Hello again!"

    flow main
      activate greeting
      activate math question
    """,
    """
    colang_version: "2.x"
    """,
)


def test_1():
    rails = LLMRails(config=config)
    messages = [{"role": "user", "content": "hi"}]

    response = rails.generate(messages=messages)

    # We should only get the input rail here.
    assert response == {"role": "assistant", "content": "Hello!"}


def test_exception_1():
    rails = LLMRails(config=config)
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "Hello!"},
        {"role": "user", "content": "How are you?"},
    ]

    with pytest.raises(ValueError) as exc_info:
        rails.generate(messages=messages)

        assert "not supported for Colang 2.0" in str(exc_info.value)


def test_state_return():
    rails = LLMRails(config=config)
    messages = [{"role": "user", "content": "hi"}]

    res = rails.generate(messages=messages, state={})

    assert isinstance(res, GenerationResponse)
    assert res.response == [{"role": "assistant", "content": "Hello!"}]

    # We submit a new message using the returned state
    res = rails.generate(messages=messages, state=res.state)

    assert res.response == [{"role": "assistant", "content": "Hello again!"}]


def test_actions_1():
    rails = LLMRails(config=config)
    messages = [{"role": "user", "content": "what is 2+2"}]

    res = rails.generate(messages=messages, state={})

    # We replace the id so that we can make a simple comparison
    tool_call_id = res.response[0]["tool_calls"][0]["id"]
    res.response[0]["tool_calls"][0]["id"] = "..."

    assert res.response == [
        {
            "tool_calls": [
                {
                    "id": "...",
                    "type": "function",
                    "function": {
                        "arguments": {"query": "what is 2+2"},
                        "name": "WolframAlphaAction",
                    },
                }
            ],
            "content": "Let me think.",
            "role": "assistant",
        }
    ]

    res = rails.generate(
        messages=[
            {
                "tool_call_id": tool_call_id,
                "role": "tool",
                "content": "The result is 4.",
            }
        ],
        state=res.state,
    )

    assert res.response == [
        {
            "content": "The result is 4.",
            "role": "assistant",
        }
    ]


@pytest.fixture
def config_2():
    return RailsConfig.from_content(
        colang_content="""
        import core

        flow main
            user said "hi"
            $datetime = await GetCurrentDateTimeAction()
            user said "there"
            bot say "hello"


        """,
        yaml_content="""
        colang_version: "2.x"
        """,
    )


def test_pattern_matching_with_python_actions(config_2):
    chat = TestChat(
        config_2,
        llm_completions=[],
    )

    chat >> "hi"
    chat >> "there"
    chat << "hello"
