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
import json
import os.path

import pytest

from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.rails.llm.options import (
    GenerationLog,
    GenerationResponse,
    GenerationStats,
)
from tests.utils import TestChat


def test_output_vars_1():
    config = RailsConfig.from_content(
        colang_content="""
                define user express greeting
                  "hi"

                define flow
                  user express greeting
                  $user_greeted = True
                  bot express greeting

                define bot express greeting
                  "Hello! How can I assist you today?"
            """,
    )
    chat = TestChat(
        config,
        llm_completions=[
            "  express greeting",
            "  express greeting",
        ],
    )

    res = chat.app.generate("hi", options={"output_vars": ["user_greeted", "something_else"]})
    output_data = res.dict().get("output_data", {})

    # We check also that a non-existent variable returns None.
    assert output_data == {"user_greeted": True, "something_else": None}

    # We also test again by trying to return the full context
    res = chat.app.generate("hi", options={"output_vars": True})
    output_data = res.dict().get("output_data", {})

    # There should be many keys
    assert len(output_data.keys()) > 5


def test_triggered_rails_info_1():
    config = RailsConfig.from_content(
        colang_content='''
            define user express greeting
              "hi"

            define flow
              user express greeting
              $user_greeted = True
              bot express greeting

            define bot express greeting
              "Hello! How can I assist you today?"

            define subflow dummy input rail
              """A dummy input rail which checks if the word "dummy" is included in the text."""
              if "dummy" in $user_message
                bot refuse to respond
                stop
        ''',
        yaml_content="""
            rails:
              input:
                flows:
                  - dummy input rail
        """,
    )
    chat = TestChat(
        config,
        llm_completions=[
            "  express greeting",
            "  express greeting",
        ],
    )

    res: GenerationResponse = chat.app.generate(
        "you are dummy", options={"log": {"activated_rails": True}, "output_vars": True}
    )

    assert res.response == "I'm sorry, I can't respond to that."

    output_data = res.output_data
    assert output_data["triggered_input_rail"] == "dummy input rail"


def test_triggered_rails_info_2():
    config = RailsConfig.from_content(
        colang_content='''
            define user express greeting
              "hi"

            define flow
              user express greeting
              $user_greeted = True
              bot express greeting

            define subflow dummy input rail
              """A dummy input rail which checks if the word "dummy" is included in the text."""
              if "dummy" in $user_message
                bot refuse to respond
                stop

            define subflow dummy output rail
              """A dummy input rail which checks if the word "dummy" is included in the text."""
              if "dummy" in $bot_message
                bot refuse to respond
                stop
        ''',
        yaml_content="""
            rails:
              input:
                flows:
                  - dummy input rail
              output:
                flows:
                  - dummy output rail
        """,
    )
    chat = TestChat(
        config,
        llm_completions=[
            "  express greeting",
            '  "You are dummy!"',
        ],
    )

    res: GenerationResponse = chat.app.generate(
        "Hello!",
        options={
            "log": {
                "activated_rails": True,
                "llm_calls": True,
                "internal_events": True,
                "colang_history": True,
            },
            "output_vars": True,
        },
    )

    assert res.response == "I'm sorry, I can't respond to that."

    assert len(res.log.activated_rails) == 5
    assert len(res.log.llm_calls) == 2
    assert len(res.log.internal_events) > 0
    assert len(res.log.colang_history) > 0


@pytest.mark.skip(reason="Run manually.")
def test_triggered_abc_bot():
    config = RailsConfig.from_path(os.path.join(os.path.dirname(__file__), "..", "examples/bots/abc"))

    rails = LLMRails(config)
    res: GenerationResponse = rails.generate("Hello!", options={"log": {"activated_rails": True}, "output_vars": True})

    print("############################")
    print(json.dumps(res.log.dict(), indent=True))

    res.log.print_summary()


@pytest.mark.skip(reason="Run manually.")
def test_triggered_rails_info_3():
    config = RailsConfig.from_content(
        yaml_content="""
            models:
              - type: main
                engine: openai
                model: gpt-3.5-turbo-instruct
        """,
    )
    rails = LLMRails(config)
    res: GenerationResponse = rails.generate(
        "Hello!",
        options={
            "log": {
                "activated_rails": True,
                "llm_calls": True,
                "internal_events": True,
                "colang_history": True,
            },
            "llm_output": True,
            "output_vars": True,
        },
    )

    print("############################")
    # print(json.dumps(res.log.dict(), indent=True))
    print(json.dumps(res.dict(), indent=True))
    res.log.print_summary()


def test_only_input_output_validation():
    config = RailsConfig.from_content(
        colang_content='''
            define user express greeting
              "hi"

            define flow
              user express greeting
              $user_greeted = True
              bot express greeting

            define subflow dummy input rail
              """A dummy input rail which checks if the word "dummy" is included in the text."""
              if "dummy" in $user_message
                bot refuse to respond
                stop

            define subflow dummy output rail
              """A dummy input rail which checks if the word "dummy" is included in the text."""
              if "dummy" in $bot_message
                bot refuse to respond
                stop
        ''',
        yaml_content="""
            rails:
              input:
                flows:
                  - dummy input rail
              output:
                flows:
                  - dummy output rail
        """,
    )
    chat = TestChat(
        config,
        llm_completions=[],
    )

    # Test only input

    res: GenerationResponse = chat.app.generate(
        "Hello!",
        options={
            "rails": ["input"],
            "log": {
                "activated_rails": True,
            },
        },
    )

    assert res.response == "Hello!"

    res = chat.app.generate(
        "Hello dummy!",
        options={
            "rails": ["input"],
            "log": {
                "activated_rails": True,
            },
        },
    )

    assert res.response == "I'm sorry, I can't respond to that."

    # Test only output

    res: GenerationResponse = chat.app.generate(
        messages=[
            {"role": "user", "content": "hi!"},
            {"role": "assistant", "content": "Hi there!"},
        ],
        options={
            "rails": ["output"],
            "log": {
                "activated_rails": True,
            },
        },
    )

    assert res.response == [{"content": "Hi there!", "role": "assistant"}]

    res = chat.app.generate(
        messages=[
            {"role": "user", "content": "hi!"},
            {"role": "assistant", "content": "Hi dummy!"},
        ],
        options={
            "rails": ["output"],
            "log": {
                "activated_rails": True,
            },
        },
    )

    assert res.response == [{"content": "I'm sorry, I can't respond to that.", "role": "assistant"}]


def test_generation_log_print_summary(capsys):
    """Test printing rais stats with dummy data"""

    stats = GenerationStats(
        input_rails_duration=1.0,
        dialog_rails_duration=2.0,
        generation_rails_duration=3.0,
        output_rails_duration=4.0,
        total_duration=10.0,  # Sum of all previous rail durations
        llm_calls_duration=8.0,  # Less than total duration
        llm_calls_count=4,  # Input, dialog, generation and output calls
        llm_calls_total_prompt_tokens=1000,
        llm_calls_total_completion_tokens=2000,
        llm_calls_total_tokens=3000,  # Sum of prompt and completion tokens
    )

    generation_log = GenerationLog(activated_rails=[], stats=stats)

    generation_log.print_summary()
    capture = capsys.readouterr()
    capture_lines = capture.out.splitlines()

    # Check the correct times were printed
    assert capture_lines[1] == "# General stats"
    assert capture_lines[3] == "- Total time: 10.00s"
    assert capture_lines[4] == "  - [1.00s][10.0%]: INPUT Rails"
    assert capture_lines[5] == "  - [2.00s][20.0%]: DIALOG Rails"
    assert capture_lines[6] == "  - [3.00s][30.0%]: GENERATION Rails"
    assert capture_lines[7] == "  - [4.00s][40.0%]: OUTPUT Rails"
    assert (
        capture_lines[8]
        == "- 4 LLM calls, 8.00s total duration, 1000 total prompt tokens, 2000 total completion tokens, 3000 total tokens."
    )
