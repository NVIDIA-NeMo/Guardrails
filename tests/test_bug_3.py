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
    define user ask general question
      "What is the capital of France?"
      "Who invented the game of chess?"

    define user ask capabilities
      "What can you do?"
      "Tell me what you can do"

    define user bey
      "bey"

    define flow
      user ask general question
      bot respond to general question

    define flow
      user ask capabilities
      bot inform capabilities

    define flow
      user ask general question
      bot respond to general question
      user ask general question
      bot respond to general question
      user ask general question
      bot respond to general question
      user bey
      bot respond bey back

    define bot respond bey back
      "Bey back!"

    define bot inform capabilities
      "I am an AI assistant built to showcase Security features of NeMo Guardrails!"
    """
)


def test_1():
    chat = TestChat(
        config,
        llm_completions=[
            "  ask general question",
            '  "The capital of France is Paris."',
            "  ask general question",
            '  "The capital of Germany is Berlin."',
            "  ask general question",
            '  "The capital of Romania is Bucharest."',
            "  bey",
        ],
    )

    chat >> "What is the capital of France?"
    chat << "The capital of France is Paris."
    chat >> "and Germany?"
    chat << "The capital of Germany is Berlin."
    chat >> "and Romania?"
    chat << "The capital of Romania is Bucharest."
    chat >> "bey"
    chat << "Bey back!"
