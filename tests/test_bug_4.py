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
define user express greeting
    "hello"
    "hi"
    "how are you"

define user give name
    "James"
    "Julio"
    "Mary"
    "Putu"

define bot name greeting
    "Hey $name!"

define flow greeting
    user express greeting
    if $name
        bot name greeting
    else
        bot express greeting
        bot ask name

define flow give name
    user give name
    $name = $last_user_message
    bot name greeting
    """
)


def test_1():
    chat = TestChat(
        config,
        llm_completions=[
            "  express greeting",
            '  "Hello there!"',
            '  "What is your name?"',
            "  give name",
        ],
    )

    chat >> "Hi"
    chat << "Hello there!\nWhat is your name?"
    chat >> "James"
    chat << "Hey James!"
