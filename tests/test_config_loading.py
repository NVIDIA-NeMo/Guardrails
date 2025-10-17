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

from nemoguardrails import RailsConfig
from nemoguardrails.rails.llm.config import Instruction


def test_default_instructions():
    config = RailsConfig.from_content(
        """
        define user express greeting
          "hello"
        """
    )

    assert config.instructions == [
        Instruction(
            type="general",
            content="Below is a conversation between a helpful AI assistant and a user. "
            "The bot is designed to generate human-like text based on the input that it receives. "
            "The bot is talkative and provides lots of specific details. "
            "If the bot does not know the answer to a question, it truthfully says it does not know.",
        )
    ]


def test_instructions_override():
    config = RailsConfig.from_content(
        """
        define user express greeting
          "hello"
        """,
        """
        instructions:
        - type: "general"
          content: |
            Below is a conversation between a helpful AI assistant and a user.
        """,
    )

    assert config.instructions == [
        Instruction(
            type="general",
            content="Below is a conversation between a helpful AI assistant and a user.\n",
        )
    ]
