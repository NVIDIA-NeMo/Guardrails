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
from nemoguardrails.rails.llm.config import colang_path_dirs
from tests.utils import TestChat

CONFIGS_FOLDER = os.path.join(os.path.dirname(__file__), "../test_configs")


def test_1():
    config = RailsConfig.from_path(os.path.join(CONFIGS_FOLDER, "with_imports_1"))

    chat = TestChat(
        config,
        llm_completions=[],
    )

    chat >> "hi"
    chat << "Hello there!"


def test_2():
    config = RailsConfig.from_content(
        colang_content="""
        import core

        flow main
          user said "hi"
          bot say "Hello there, it's working!"
    """,
        config={"colang_version": "2.x"},
    )

    chat = TestChat(
        config,
        llm_completions=[],
    )

    chat >> "hi"
    chat << "Hello there, it's working!"


def test_3():
    # This config just imports another one, to check that actions are correctly
    # loaded.
    colang_path_dirs.append(
        os.path.join(os.path.dirname(__file__), "..", "test_configs")
    )

    config = RailsConfig.from_content(
        colang_content="""
            import with_custom_action_v2_x
        """,
        config={"colang_version": "2.x"},
    )

    chat = TestChat(
        config,
        llm_completions=[],
    )

    chat >> "start"
    chat << "8"

    colang_path_dirs.pop()
