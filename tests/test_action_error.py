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


def test_action_not_found():
    """Test that setting variables in context works correctly."""
    config = RailsConfig.from_content(
        """
        define user express greeting
            "hello"

        define flow
            user express greeting
            execute fetch_user_profile
            bot express greeting
        """
    )
    chat = TestChat(
        config,
        llm_completions=[
            "  express greeting",
            '  "Hello John!"',
        ],
    )

    chat >> "hello there!"
    chat << "Action 'fetch_user_profile' not found."


def test_action_internal_error():
    """Test that setting variables in context works correctly."""
    config = RailsConfig.from_content(
        """
        define user express greeting
            "hello"

        define flow
            user express greeting
            execute fetch_user_profile
            bot express greeting
        """
    )
    chat = TestChat(
        config,
        llm_completions=[
            "  express greeting",
            '  "Hello John!"',
        ],
    )

    def fetch_user_profile():
        raise Exception("Some exception")

    chat.app.register_action(fetch_user_profile)

    chat >> "hello there!"
    chat << "I'm sorry, an internal error has occurred."


def test_action_not_registered():
    """Test that an error is raised when an action is not registered."""
    config = RailsConfig.from_content(
        """
        define user express greeting
            "hello"

        define flow
            user express greeting
            execute unregistered_action
            bot express greeting
        """
    )
    chat = TestChat(
        config,
        llm_completions=[
            "  express greeting",
            '  "Hello John!"',
        ],
    )

    chat >> "hello there!"
    chat << "Action 'unregistered_action' not found."
