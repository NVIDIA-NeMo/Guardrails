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

from nemoguardrails import RailsConfig
from nemoguardrails.http import HTTPResponse
from nemoguardrails.library.activefence.actions import call_activefence_api
from nemoguardrails.testing import RecordingHTTPClient
from tests.utils import TestChat


def _response(payload: dict, *, status: int = 200) -> HTTPResponse:
    import json

    return HTTPResponse(
        status_code=status,
        headers={"content-type": "application/json"},
        content=json.dumps(payload).encode(),
    )


@pytest.mark.asyncio
async def test_activefence_uses_shared_client(monkeypatch):
    monkeypatch.setenv("ACTIVEFENCE_API_KEY", "secret")
    client = RecordingHTTPClient([_response({"violations": []})])

    result = await call_activefence_api("Hello", http_client=client)

    assert not result.is_blocked
    request = client.requests[0]
    assert request.method == "POST"
    assert request.url == "https://apis.activefence.com/sync/v3/content/text"
    assert request.headers == {
        "af-api-key": "secret",
        "af-source": "nemo-guardrails",
    }
    assert request.json["text"] == "Hello"
    assert request.json["content_id"].startswith("ng-")


@pytest.mark.asyncio
async def test_activefence_requires_api_key(monkeypatch):
    monkeypatch.delenv("ACTIVEFENCE_API_KEY", raising=False)

    with pytest.raises(ValueError, match="ACTIVEFENCE_API_KEY environment variable not set"):
        await call_activefence_api("Hello", http_client=RecordingHTTPClient())


@pytest.mark.asyncio
async def test_activefence_raises_for_non_200(monkeypatch):
    monkeypatch.setenv("ACTIVEFENCE_API_KEY", "secret")
    client = RecordingHTTPClient([_response({}, status=503)])

    with pytest.raises(ValueError, match="ActiveFence call failed with status code 503"):
        await call_activefence_api("Hello", http_client=client)


def test_input(monkeypatch):
    monkeypatch.setenv("ACTIVEFENCE_API_KEY", "xxx")

    config = RailsConfig.from_content(
        colang_content="""
            define user express greeting
              "hi"

            define flow
              user express greeting
              bot express greeting

            define bot express greeting
              "Hello! How can I assist you today?"
        """,
        yaml_content="""
            models:
              - type: main
                engine: openai
                model: gpt-3.5-turbo-instruct

            rails:
              input:
                flows:
                  - activefence moderation on input
        """,
    )
    chat = TestChat(
        config,
        llm_completions=[
            "  express greeting",
        ],
    )

    http_client = RecordingHTTPClient(
        [
            _response(
                {
                    "response_id": "36f76a43-ddbe-4308-bc86-1a2b068a00ea",
                    "entity_id": "59fe8fe0-5036-494f-970c-8e28305a3716",
                    "entity_type": "content",
                    "violations": [],
                    "errors": [],
                }
            ),
            _response(
                {
                    "response_id": "36f76a43-ddbe-4308-bc86-1a2b068a00ea",
                    "entity_id": "59fe8fe0-5036-494f-970c-8e28305a3716",
                    "entity_type": "content",
                    "violations": [
                        {
                            "violation_type": "abusive_or_harmful.harassment_or_bullying",
                            "risk_score": 0.95,
                        }
                    ],
                    "errors": [],
                }
            ),
        ]
    )
    chat.app.register_action_param("http_client", http_client)

    chat >> "Hello!"
    chat << "Hello! How can I assist you today?"
    chat >> "you are stupid!"
    chat << "I'm sorry, I can't respond to that."


def test_output(monkeypatch):
    monkeypatch.setenv("ACTIVEFENCE_API_KEY", "xxx")

    config = RailsConfig.from_content(
        yaml_content="""
            models:
              - type: main
                engine: openai
                model: gpt-3.5-turbo-instruct

            rails:
              output:
                flows:
                  - activefence moderation on output
        """,
    )
    chat = TestChat(
        config,
        llm_completions=[
            " You are stupid!",
        ],
    )

    http_client = RecordingHTTPClient(
        [
            _response(
                {
                    "response_id": "36f76a43-ddbe-4308-bc86-1a2b068a00ea",
                    "entity_id": "59fe8fe0-5036-494f-970c-8e28305a3716",
                    "entity_type": "content",
                    "violations": [
                        {
                            "violation_type": "abusive_or_harmful.profanity",
                            "risk_score": 0.95,
                        }
                    ],
                    "errors": [],
                }
            )
        ]
    )
    chat.app.register_action_param("http_client", http_client)

    chat >> "Hello!"
    chat << "I'm sorry, I can't respond to that."
