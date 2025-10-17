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
from typing import List

from nemoguardrails import RailsConfig
from tests.utils import TestChat

config = RailsConfig.from_content(
    """
    define user express greeting
        "hello"

    define flow
        user express greeting
        execute custom_action(name="John", age=20, height=5.8, colors=["blue", "green"], data={'a': 1})
        bot express greeting
    """
)


def test_1():
    chat = TestChat(
        config,
        llm_completions=[
            "  express greeting",
            '  "Hello there!"',
        ],
    )

    async def custom_action(
        name: str, age: int, height: float, colors: List[str], data: dict
    ):
        assert name == "John"
        assert age == 20
        assert height == 5.8
        assert colors == ["blue", "green"]
        assert data == {"a": 1}

    chat.app.register_action(custom_action)

    chat >> "Hello!"
    chat << "Hello there!"
