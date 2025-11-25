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
from tests.utils import TestChat


def test_1():
    """Test that multi-line responses are processed correctly."""
    config = RailsConfig.from_content(
        """
        define user express greeting
            "hello"

        define flow
            user express greeting
            bot express greeting and list of thing to help
        """
    )
    chat = TestChat(
        config,
        llm_completions=[
            "  express greeting",
            '  "Hello, there! \nI can help you with:\n\n 1. Answering questions \n2. Sending messages"\n\n',
        ],
    )

    chat >> "hello there!"
    (chat << "Hello, there!\nI can help you with:\n1. Answering questions\n2. Sending messages")


def test_1_single_call():
    """Test that multi-line responses are processed correctly by single LLM call configuration."""
    config: RailsConfig = RailsConfig.from_content(
        """
        define user express greeting
            "hello"

        define flow
            user express greeting
            bot express greeting and list of thing to help
        """
    )
    # Set the single call flag.
    config.rails.dialog.single_call.enabled = True

    chat = TestChat(
        config,
        llm_completions=[
            "  express greeting\n"
            "bot express greeting and list of thing to help\n"
            '  "Hello, there! \nI can help you with:\n\n 1. Answering questions \n2. Sending messages"',
        ],
    )

    chat >> "hello there!"
    (chat << "Hello, there!\nI can help you with:\n1. Answering questions\n2. Sending messages")
