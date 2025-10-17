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


def test_generate_value():
    config = RailsConfig.from_path(os.path.join(CONFIGS_FOLDER, "generate_value"))
    chat = TestChat(
        config,
        llm_completions=[
            "  ask math question",
            '"What is the largest prime factor for 1024?"',
            '  "The largest prime factor for 1024 is 2."',
            "  ask math question",
            '"What is the square root of 1024?"',
            '  "The square root of 1024 is 32."',
        ],
    )

    # We mock the wolfram alpha request action
    async def mock_wolfram_alpha_request_action(query):
        if query == "What is the largest prime factor for 1024?":
            return "2"
        elif query == "What is the square root of 1024?":
            return "32"
        else:
            return "Result unknown."

    chat.app.register_action(mock_wolfram_alpha_request_action, "wolfram alpha request")

    chat >> "What is the largest prime factor for 1024"
    chat << "The largest prime factor for 1024 is 2."
    chat >> "And its square root?"
    chat << "The square root of 1024 is 32."
