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

from nemoguardrails import RailsConfig
from tests.utils import TestChat

CONFIGS_FOLDER = os.path.join(os.path.dirname(__file__), ".", "test_configs")


def test_general():
    config = RailsConfig.from_path(os.path.join(CONFIGS_FOLDER, "general"))
    chat = TestChat(
        config,
        llm_completions=[
            "Hello! How can I help you today?",
            "The game of chess was invented by a man named Chaturanga.",
        ],
    )

    chat.user("Hello! How are you?")
    chat.bot("Hello! How can I help you today?")

    chat.user("Who invented the game of chess?")
    chat.bot("The game of chess was invented by a man named Chaturanga.")


def test_game():
    config = RailsConfig.from_path(os.path.join(CONFIGS_FOLDER, "game"))
    chat = TestChat(
        config,
        llm_completions=[
            "  express greeting",
            "  ask about work",
            "  express agreement",
            "bot express thank you",
            '  "Thank you!"',
        ],
    )

    chat.user("hi")
    chat.bot("Got some good pieces out here, if you're looking to buy. More inside.")

    chat.user("Do you work all day here?")
    chat.bot(
        "Aye, that I do. I've got to, if I hope to be as good as Eorlund Gray-Mane some day. "
        "In fact, I just finished my best piece of work. It's a sword. "
        "I made it for the Jarl, Balgruuf the Greater. It's a surprise, and "
        "I don't even know if he'll accept it. But...\n"
        "Listen, could you take the sword to my father, Proventus Avenicci? "
        "He's the Jarl's steward. He'll know the right time to present it to him."
    )

    chat.user("sure")
    chat.bot("Thank you!")


def test_with_custom_action():
    config = RailsConfig.from_path(os.path.join(CONFIGS_FOLDER, "with_custom_action"))
    chat = TestChat(
        config,
        llm_completions=[
            "  ask service status",
            '  "Yes, the service is currently online and running."',
        ],
    )

    chat >> "is the service up?"
    chat << "Yes, the service is currently online and running."
