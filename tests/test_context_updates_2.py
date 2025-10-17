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

    define flow
        $mode = $mode or "strict"

        if $mode == "strict"
            user express greeting
            bot express greeting
    """
)


def test_1():
    """Test that branching with `when` works correctly."""
    chat = TestChat(
        config,
        llm_completions=[
            "  express greeting",
            '  "Hello there!"',
        ],
    )

    chat >> "Hello!"
    chat << "Hello there!"
