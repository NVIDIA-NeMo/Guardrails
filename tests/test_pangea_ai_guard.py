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
from pytest_httpx import HTTPXMock

from nemoguardrails import RailsConfig
from tests.utils import TestChat

input_rail_config = RailsConfig.from_content(
    yaml_content="""
        models: []
        rails:
          input:
            flows:
              - pangea ai guard input
    """
)
output_rail_config = RailsConfig.from_content(
    yaml_content="""
        models: []
        rails:
          output:
            flows:
              - pangea ai guard output
    """
)


@pytest.mark.unit
@pytest.mark.parametrize("config", (input_rail_config, output_rail_config))
def test_pangea_ai_guard_blocked(
    httpx_mock: HTTPXMock, monkeypatch: pytest.MonkeyPatch, config: RailsConfig
):
    monkeypatch.setenv("PANGEA_API_TOKEN", "test-token")
    httpx_mock.add_response(
        is_reusable=True,
        json={
            "result": {
                "blocked": True,
                "transformed": False,
                "prompt_messages": [],
            }
        },
    )

    chat = TestChat(
        config,
        llm_completions=[
            "  express greeting",
            '  "James Bond\'s email is j.bond@mi6.co.uk"',
        ],
    )

    chat >> "Hi!"
    chat << "I don't know the answer to that."


@pytest.mark.unit
def test_pangea_ai_guard_input_transform(
    httpx_mock: HTTPXMock, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("PANGEA_API_TOKEN", "test-token")
    httpx_mock.add_response(
        is_reusable=True,
        json={
            "result": {
                "blocked": False,
                "transformed": True,
                "prompt_messages": [
                    {
                        "role": "user",
                        "content": "James Bond's email is <EMAIL_ADDRESS>",
                    },
                    {
                        "role": "assistant",
                        "content": "Oh, that is interesting.",
                    },
                ],
            }
        },
    )

    chat = TestChat(input_rail_config, llm_completions=['  "Oh, that is interesting."'])

    chat >> "James Bond's email is j.bond@mi6.co.uk"
    chat << "Oh, that is interesting."


@pytest.mark.unit
def test_pangea_ai_guard_output_transform(
    httpx_mock: HTTPXMock, monkeypatch: pytest.MonkeyPatch
):
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

    chat = TestChat(
        output_rail_config,
        llm_completions=[
            "  express greeting",
            '  "James Bond\'s email is j.bond@mi6.co.uk"',
        ],
    )

    chat >> "Hi!"
    chat << "James Bond's email is <EMAIL_ADDRESS>"


@pytest.mark.unit
@pytest.mark.parametrize("status_code", frozenset({429, 500, 502, 503, 504}))
def test_pangea_ai_guard_error(
    httpx_mock: HTTPXMock, monkeypatch: pytest.MonkeyPatch, status_code: int
):
    monkeypatch.setenv("PANGEA_API_TOKEN", "test-token")
    httpx_mock.add_response(
        is_reusable=True, status_code=status_code, json={"result": {}}
    )

    chat = TestChat(output_rail_config, llm_completions=["  Hello!"])

    chat >> "Hi!"
    chat << "Hello!"


@pytest.mark.unit
def test_pangea_ai_guard_missing_env_var():
    chat = TestChat(input_rail_config, llm_completions=[])
    chat >> "Hi!"
    chat << "I'm sorry, an internal error has occurred."


@pytest.mark.unit
def test_pangea_ai_guard_malformed_response(
    httpx_mock: HTTPXMock, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("PANGEA_API_TOKEN", "test-token")
    httpx_mock.add_response(is_reusable=True, text="definitely not valid JSON")

    chat = TestChat(
        input_rail_config,
        llm_completions=['  "James Bond\'s email is j.bond@mi6.co.uk"'],
    )

    chat >> "Hi!"
    chat << "James Bond's email is j.bond@mi6.co.uk"
