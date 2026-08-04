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

import json

import pytest

from nemoguardrails import RailsConfig
from nemoguardrails.actions.rail_outcome import RailOutcome, TransformTarget
from nemoguardrails.http import HTTPConnectionError, HTTPResponse
from nemoguardrails.library.prompt_security.actions import (
    _protect_text_outcome,
    protect_text,
    ps_protect_api_async,
)
from nemoguardrails.testing import RecordingHTTPClient
from tests.utils import TestChat


def mock_protect_text(return_value):
    def mock_request(*args, **kwargs):
        if isinstance(return_value, RailOutcome):
            return return_value
        target = TransformTarget.BOT_MESSAGE if kwargs.get("bot_response") else TransformTarget.USER_MESSAGE
        if return_value.get("is_blocked"):
            return RailOutcome.block(metadata=return_value)
        if return_value.get("is_modified"):
            return RailOutcome.transform([(target, return_value.get("modified_text") or "")], metadata=return_value)
        return RailOutcome.allow(metadata=return_value)

    return mock_request


@pytest.mark.unit
def test_prompt_security_protection_disabled():
    config = RailsConfig.from_content(
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
            '  "Hi! My name is John as well."',
        ],
    )

    chat.app.register_action(mock_protect_text({"is_blocked": True, "is_modified": False}), "protect_text")
    chat >> "Hi! I am Mr. John! And my email is test@gmail.com"
    chat << "Hi! My name is John as well."


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        (
            {"is_blocked": False, "is_modified": False, "modified_text": None},
            RailOutcome.allow(metadata={"is_blocked": False, "is_modified": False, "modified_text": None}),
        ),
        (
            {"is_blocked": True, "is_modified": False, "modified_text": None},
            RailOutcome.block(metadata={"is_blocked": True, "is_modified": False, "modified_text": None}),
        ),
        (
            {"is_blocked": False, "is_modified": True, "modified_text": "masked"},
            RailOutcome.transform(
                [(TransformTarget.USER_MESSAGE, "masked")],
                metadata={"is_blocked": False, "is_modified": True, "modified_text": "masked"},
            ),
        ),
    ],
)
def test_protect_text_outcome(result, expected):
    assert _protect_text_outcome(result, TransformTarget.USER_MESSAGE) == expected


def _response(payload: dict) -> HTTPResponse:
    return HTTPResponse(status_code=200, content=json.dumps(payload).encode())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("payload", "response", "expected"),
    [
        (
            {"result": {"action": "block"}},
            None,
            {"is_blocked": True, "is_modified": False, "modified_text": None},
        ),
        (
            {
                "result": {
                    "action": "modify",
                    "prompt": {"modified_text": "safe prompt"},
                }
            },
            None,
            {"is_blocked": False, "is_modified": True, "modified_text": "safe prompt"},
        ),
        (
            {
                "result": {
                    "action": "modify",
                    "response": {"modified_text": "safe response"},
                }
            },
            "response",
            {"is_blocked": False, "is_modified": True, "modified_text": "safe response"},
        ),
    ],
)
async def test_prompt_security_api_uses_shared_client(payload, response, expected):
    client = RecordingHTTPClient([_response(payload)])

    result = await ps_protect_api_async(
        "https://prompt-security.example/protect",
        "app-id",
        prompt="prompt" if response is None else None,
        system_prompt="system",
        response=response,
        user="user",
        http_client=client,
    )

    assert result == expected
    request = client.requests[0]
    assert request.method == "POST"
    assert request.url == "https://prompt-security.example/protect"
    assert request.headers == {
        "APP-ID": "app-id",
        "Content-Type": "application/json",
    }
    assert request.json == {
        "prompt": "prompt" if response is None else None,
        "system_prompt": "system",
        "response": response,
        "user": "user",
    }
    assert request.timeout is None


@pytest.mark.asyncio
async def test_prompt_security_api_allows_on_http_failure(caplog):
    client = RecordingHTTPClient([HTTPConnectionError("connection failed")])

    result = await ps_protect_api_async(
        "https://prompt-security.example/protect",
        "app-id",
        prompt="prompt",
        http_client=client,
    )

    assert result == {
        "is_blocked": False,
        "is_modified": False,
        "modified_text": None,
    }
    assert "Error calling Prompt Security Protect API: connection failed" in caplog.text


@pytest.mark.asyncio
async def test_protect_text_forwards_shared_client(monkeypatch):
    monkeypatch.setenv("PS_PROTECT_URL", "https://prompt-security.example/protect")
    monkeypatch.setenv("PS_APP_ID", "app-id")
    client = RecordingHTTPClient(
        [
            _response({"result": {"action": "block"}}),
            _response(
                {
                    "result": {
                        "action": "modify",
                        "response": {"modified_text": "safe response"},
                    }
                }
            ),
        ]
    )

    input_outcome = await protect_text(user_prompt="prompt", http_client=client)
    output_outcome = await protect_text(bot_response="response", http_client=client)

    assert input_outcome == RailOutcome.block(
        metadata={
            "is_blocked": True,
            "is_modified": False,
            "modified_text": None,
        }
    )
    assert output_outcome == RailOutcome.transform(
        [(TransformTarget.BOT_MESSAGE, "safe response")],
        metadata={
            "is_blocked": False,
            "is_modified": True,
            "modified_text": "safe response",
        },
    )
    input_request, output_request = client.requests
    assert input_request.json == {
        "prompt": "prompt",
        "system_prompt": None,
        "response": None,
        "user": None,
    }
    assert output_request.json == {
        "prompt": None,
        "system_prompt": None,
        "response": "response",
        "user": None,
    }


@pytest.mark.asyncio
async def test_protect_text_validates_configuration(monkeypatch):
    monkeypatch.delenv("PS_PROTECT_URL", raising=False)
    monkeypatch.delenv("PS_APP_ID", raising=False)

    with pytest.raises(ValueError, match="PS_PROTECT_URL"):
        await protect_text(user_prompt="prompt")

    monkeypatch.setenv("PS_PROTECT_URL", "https://prompt-security.example/protect")
    with pytest.raises(ValueError, match="PS_APP_ID"):
        await protect_text(user_prompt="prompt")

    monkeypatch.setenv("PS_APP_ID", "app-id")
    with pytest.raises(ValueError, match="Neither user_message nor bot_message"):
        await protect_text()


@pytest.mark.unit
def test_prompt_security_protection_input():
    config = RailsConfig.from_content(
        yaml_content="""
            models: []
            rails:
              input:
                flows:
                  - protect prompt
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
            '  "Hi! My name is John as well."',
        ],
    )

    chat.app.register_action(mock_protect_text({"is_blocked": True, "is_modified": False}), "protect_text")
    chat >> "Hi! I am Mr. John! And my email is test@gmail.com"
    chat << "I can't answer that."


@pytest.mark.unit
def test_prompt_security_protection_output():
    config = RailsConfig.from_content(
        yaml_content="""
            models: []
            rails:
              output:
                flows:
                  - protect response
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
            '  "Hi! My name is John as well."',
        ],
    )

    chat.app.register_action(mock_protect_text({"is_blocked": True, "is_modified": False}), "protect_text")
    chat >> "Hi!"
    chat << "I can't answer that."
