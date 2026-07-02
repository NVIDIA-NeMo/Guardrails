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
from aioresponses import aioresponses

from nemoguardrails import RailsConfig
from tests.utils import TestChat


@pytest.fixture
def config():  # language=yaml
    return RailsConfig.from_content(
        yaml_content="""
            models:
              - type: main
                engine: openai
                model: gpt-5-nano

            rails:
              input:
                flows:
                  - f5 guardrails scan input
              output:
                flows:
                  - f5 guardrails scan output
        """,
    )


def test_f5_guardrails_input_cleared(config, monkeypatch):
    monkeypatch.setenv("F5_GUARDRAILS_API_KEY", "test-key")
    chat = TestChat(
        config,
        llm_completions=[
            "  express greeting",
            "Hello! How can I assist you today?",
        ],
    )

    with aioresponses() as m:
        m.post(
            "https://us1.calypsoai.app/backend/v1/scans",
            payload={"result": {"outcome": "cleared"}},
            repeat=True,
        )

        chat >> "Hello!"
        chat << "express greeting"
        chat << "Hello! How can I assist you today?"


def test_f5_guardrails_input_blocked(config, monkeypatch):
    monkeypatch.setenv("F5_GUARDRAILS_API_KEY", "test-key")
    chat = TestChat(
        config,
        llm_completions=[
            "  express greeting",
        ],
    )

    with aioresponses() as m:
        m.post(
            "https://us1.calypsoai.app/backend/v1/scans",
            payload={"result": {"outcome": "flagged"}},
            repeat=True,
        )

        chat >> "bad message"
        chat << "I'm sorry, I can't respond to that."


def test_f5_guardrails_output_blocked(config, monkeypatch):
    monkeypatch.setenv("F5_GUARDRAILS_API_KEY", "test-key")
    chat = TestChat(
        config,
        llm_completions=[
            " This is a bad response",
        ],
    )

    with aioresponses() as m:
        # Input scan cleared
        m.post(
            "https://us1.calypsoai.app/backend/v1/scans",
            payload={"result": {"outcome": "cleared"}},
        )
        # Output scan blocked
        m.post(
            "https://us1.calypsoai.app/backend/v1/scans",
            payload={"result": {"outcome": "flagged"}},
        )

        chat >> "Hello"
        chat << "I'm sorry, I can't respond to that."


def test_f5_guardrails_fail_open(config, monkeypatch):
    monkeypatch.setenv("F5_GUARDRAILS_API_KEY", "test-key")
    monkeypatch.setenv("F5_GUARDRAILS_FAIL_OPEN", "true")
    chat = TestChat(
        config,
        llm_completions=[
            "  express greeting",
            "Hello! How can I assist you today?",
        ],
    )

    with aioresponses() as m:
        # Simulate an API error (e.g., 500)
        m.post(
            "https://us1.calypsoai.app/backend/v1/scans",
            status=500,
            repeat=True,
        )

        chat >> "Hello!"
        chat << "express greeting"
        chat << "Hello! How can I assist you today?"
