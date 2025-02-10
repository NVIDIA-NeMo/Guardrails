# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


def test_general_instructions_get_included_when_no_canonical_forms_are_defined():
    config: RailsConfig = RailsConfig.from_content(
        config={
            "models": [],
            "instructions": [
                {
                    "type": "general",
                    "content": "This is a conversation between a user and a bot.",
                }
            ],
        }
    )

    chat = TestChat(
        config,
        llm_completions=[
            "  Hello there!",
        ],
    )

    chat >> "hello there!"
    chat << "Hello there!"

    info = chat.app.explain()
    assert "This is a conversation between a user and a bot." in info.llm_calls[0].prompt
