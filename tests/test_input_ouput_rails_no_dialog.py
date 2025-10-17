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
    define flow example input rail
      if $user_message == "stupid"
        bot refuse to respond
        stop
      else
        $user_message = $user_message + "!"

    define flow example output rail
      $bot_message = $bot_message + "!"
    """,
    """
    rails:
        input:
            flows:
                - example input rail
        output:
            flows:
                - example output rail
    """,
)


def test_1():
    chat = TestChat(
        config,
        llm_completions=[],
    )

    chat >> "stupid"
    chat << "I'm sorry, I can't respond to that."


def test_2():
    chat = TestChat(
        config,
        llm_completions=[
            '  "Hello!"',
        ],
    )

    chat >> "Hello!"
    chat << "Hello!!"
