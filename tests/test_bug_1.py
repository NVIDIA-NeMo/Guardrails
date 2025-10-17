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

config = RailsConfig.from_content(
    """
    define user report bug
      "report bug"

    define bot express greeting
      "Hi"

    define flow log bugs
      priority 2

      user report bug
      bot ask what is wrong
      user inform bug details
      bot thank user
      execute log_conversation
    """
)


def test_1():
    """Test that branching with `when` works correctly."""
    chat = TestChat(
        config,
        llm_completions=[
            "  report bug",
            '  "What is wrong?"',
            "  inform bug details",
            '  "Thank you!"',
        ],
    )

    log = []

    async def log_conversation(context: dict):
        log.append(context.get("last_user_message"))

    chat.app.register_action(log_conversation)

    chat >> "report bug!"
    chat << "What is wrong?"
    chat >> "api not working"
    chat << "Thank you!"

    assert log == ["api not working"]
