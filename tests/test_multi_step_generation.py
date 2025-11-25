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
import textwrap

import pytest

from nemoguardrails import RailsConfig
from tests.utils import TestChat

CONFIGS_FOLDER = os.path.join(os.path.dirname(__file__), ".", "test_configs")


def test_multi_step_generation():
    """Test that the multi-step generation works as expected.

    In this test the LLM generates a flow with two steps:
      bot acknowledge the date
      bot confirm appointment
    """
    config = RailsConfig.from_path(os.path.join(CONFIGS_FOLDER, "multi_step_generation"))
    chat = TestChat(
        config,
        llm_completions=[
            "  express greeting",
            "  request appointment",
            '  "What\'s your name?"',
            "  provide date",
            "bot acknowledge the date\nbot confirm appointment",
            '  "Ok, an appointment for tomorrow."',
            '  "Your appointment is now confirmed."',
        ],
    )

    chat >> "hi"
    chat << "Hey there!"
    chat >> "i need to make an appointment"
    chat << "I can certainly help you with that.\nWhat's your name?"
    chat >> "I want to come tomorrow"
    chat << "Ok, an appointment for tomorrow.\nYour appointment is now confirmed."


@pytest.mark.skip("This test is not working as expected")
def test_multi_step_generation_with_parsing_error():
    """Test that the multi-step generation works as expected.

    In this test the LLM generates a flow with two steps and a broken one:
      bot acknowledge the date
      bot confirm appointment
      something that breaks parsing

    The last step is broken and should be ignored.
    """

    config = RailsConfig.from_path(os.path.join(CONFIGS_FOLDER, "multi_step_generation"))
    chat = TestChat(
        config,
        llm_completions=[
            "  express greeting",
            "  request appointment",
            '  "What\'s your name?"',
            "  provide date",
            "bot acknowledge the date\nbot confirm appointment\nsomething that breaks parsing",
            '  "Ok, an appointment for tomorrow."',
            '  "Your appointment is now confirmed."',
        ],
    )

    chat >> "hi"
    chat << "Hey there!"
    chat >> "i need to make an appointment"
    chat << "I can certainly help you with that.\nWhat's your name?"
    chat >> "I want to come tomorrow"
    chat << "Ok, an appointment for tomorrow.\nYour appointment is now confirmed."


LONGER_FLOW = textwrap.dedent(
    """
    bot acknowledge the date
    bot ask name again
    user inform name
    # Extract the name. If not provided say "unknown".
    $name = ...
    if $name == "unknown"
      bot ask name again
    bot confirm appointment
    """
).strip()


def test_multi_step_generation_longer_flow():
    """Test that the multi-step generation works as expected.

    In this test the LLM generates a longer flow:
      bot acknowledge the date
      bot ask name again
      user inform name
      # Extract the name. If not provided say "unknown".
      $name = ...
      if $name == "unknown"
        bot ask name again
      bot confirm appointment
    """
    config = RailsConfig.from_path(os.path.join(CONFIGS_FOLDER, "multi_step_generation"))
    chat = TestChat(
        config,
        llm_completions=[
            "  express greeting",
            "  request appointment",
            '  "What\'s your name?"',
            "  provide date",
            f"{LONGER_FLOW}",
            '  "Ok, an appointment for tomorrow."',
            '  "What is your name again?"',
            "  inform name",
            '  "John"',
            '  "Your appointment is now confirmed."',
        ],
    )

    chat >> "hi"
    chat << "Hey there!"
    chat >> "i need to make an appointment"
    chat << "I can certainly help you with that.\nWhat's your name?"
    chat >> "I want to come tomorrow"
    chat << "Ok, an appointment for tomorrow.\nWhat is your name again?"
    chat >> "My name is John"
    chat << "Your appointment is now confirmed."
