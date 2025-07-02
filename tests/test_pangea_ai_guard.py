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

import pytest
from pytest_httpx import HTTPXMock

from nemoguardrails import RailsConfig
from tests.utils import TestChat


@pytest.mark.unit
def test_pangea_ai_guard_output(httpx_mock: HTTPXMock, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("PANGEA_API_TOKEN", "test-token")
    httpx_mock.add_response(
        is_reusable=True,
        json={
            "result": {
                "blocked": False,
                "transformed": True,
                "prompt_messages": [
                    {
                        "role": "assistant",
                        "content": "James Bond's email is <EMAIL_ADDRESS>",
                    }
                ],
            }
        },
    )

    config = RailsConfig.from_content(
        yaml_content="""
            models: []
            rails:
              input:
                flows:
                  - pangea ai guard input
              output:
                flows:
                  - pangea ai guard output
        """,
        colang_content="""
            define user express greeting
              "hi"

            define flow
              user express greeting
              bot express greeting

            define bot inform answer unknown
              "I can't answer that."
        """,
    )

    chat = TestChat(
        config,
        llm_completions=[
            "  express greeting",
            '  "James Bond\'s email is j.bond@mi6.co.uk"',
        ],
    )

    chat >> "Hi!"
    chat << "James Bond's email is <EMAIL_ADDRESS>"
