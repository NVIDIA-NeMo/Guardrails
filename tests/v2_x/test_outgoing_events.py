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
import pytest

from nemoguardrails import RailsConfig
from tests.utils import TestChat


@pytest.fixture
def config_1():
    return RailsConfig.from_content(
        colang_content="""
            flow user said $text
              match UtteranceUserActionFinished(final_transcript=$text)

            flow bot say $text
              await UtteranceBotAction(script=$text)

            flow wait
              await SomeRandomAction()

            # ---
            flow a
              match EventA()
              send EventC()

            flow b
              match EventB()
              send EventD()

            flow c
              match EventD()
              bot say "Hello"

            flow main
              activate a
              activate b
              activate c

              user said "hi"
              send EventA()
              send EventB()
              wait
            """,
        yaml_content="""
            colang_version: "2.x"
            """,
    )


def test_outgoing_events(config_1):
    chat = TestChat(
        config_1,
        llm_completions=[],
    )

    chat >> "hi"
    chat << "Hello"
