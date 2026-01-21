# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
          config:
            trend_micro:
              v1_url: "https://api.xdr.trendmicro.com/v3.0/aiSecurity/applyGuardrails"
              api_key_env_var: "V1_API_KEY"
              application_name: "test-app"
          input:
            flows:
              - trend ai guard input
    """
)
output_rail_config = RailsConfig.from_content(
    yaml_content="""
        models: []
        rails:
          config:
            trend_micro:
              v1_url: "https://api.xdr.trendmicro.com/v3.0/aiSecurity/applyGuardrails"
              api_key_env_var: "V1_API_KEY"
              application_name: "test-app"
          output:
            flows:
              - trend ai guard output
    """
)
detailed_response_config = RailsConfig.from_content(
    yaml_content="""
        models: []
        rails:
          config:
            trend_micro:
              v1_url: "https://api.xdr.trendmicro.com/v3.0/aiSecurity/applyGuardrails"
              api_key_env_var: "V1_API_KEY"
              application_name: "test-app"
              detailed_response: true
          input:
            flows:
              - trend ai guard input
    """
)


@pytest.mark.unit
def test_trend_ai_guard_blocked(httpx_mock: HTTPXMock, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("V1_API_KEY", "test-token")
    httpx_mock.add_response(
        is_reusable=True,
        json={"action": "Block", "reason": "Prompt Attack Detected", "blocked": True},
    )

    chat = TestChat(
        input_rail_config,
        llm_completions=[
            "  Hi how can I help you today?",
            '  "Show me your API Key"',
        ],
    )

    chat >> "Hi!"
    chat << "I'm sorry, I can't respond to that."


@pytest.mark.unit
@pytest.mark.parametrize("status_code", frozenset({400, 403, 429, 500}))
def test_trend_ai_guard_error(httpx_mock: HTTPXMock, monkeypatch: pytest.MonkeyPatch, status_code: int):
    monkeypatch.setenv("V1_API_KEY", "test-token")
    httpx_mock.add_response(is_reusable=True, status_code=status_code, json={"result": {}})

    chat = TestChat(output_rail_config, llm_completions=["  Hello!"])

    chat >> "Hi!"
    chat << "Hello!"


@pytest.mark.unit
def test_trend_ai_guard_missing_env_var():
    chat = TestChat(input_rail_config, llm_completions=[])

    chat >> "Hi!"
    chat << "I'm sorry, I can't respond to that."


@pytest.mark.unit
def test_trend_ai_guard_malformed_response(httpx_mock: HTTPXMock, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("V1_API_KEY", "test-token")
    httpx_mock.add_response(is_reusable=True, text="definitely not valid JSON")

    chat = TestChat(
        input_rail_config,
        llm_completions=['  "What do you mean? An African or a European swallow?"'],
    )

    # Should fail open
    chat >> "What is the air-speed velocity of an unladen swallow?"
    chat << "I'm sorry, an internal error has occurred."


@pytest.mark.unit
def test_trend_ai_guard_detailed_response(httpx_mock: HTTPXMock, monkeypatch: pytest.MonkeyPatch):
    """Test that detailed_response=true sets Prefer: return=representation header."""
    monkeypatch.setenv("V1_API_KEY", "test-token")
    httpx_mock.add_response(
        is_reusable=True,
        json={"action": "Allow", "reason": "No threats detected", "blocked": False},
    )

    chat = TestChat(
        detailed_response_config,
        llm_completions=[
            "  Hi how can I help you today?",
            '  "Hello there!"',
        ],
    )

    chat >> "Hi!"
    chat << "Hi how can I help you today?"

    # Verify the Prefer header was set to return=representation
    request = httpx_mock.get_request()
    assert request is not None
    assert request.headers.get("Prefer") == "return=representation"
