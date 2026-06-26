# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import nemoguardrails.actions.math as math_actions
from nemoguardrails.actions.actions import ActionResult


class FakeWolframResponse:
    def __init__(self, status, body):
        self.status = status
        self.body = body

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def text(self):
        return self.body


class FakeWolframSession:
    def __init__(self, response):
        self.response = response
        self.url = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def get(self, url):
        self.url = url
        return self.response


@pytest.mark.asyncio
async def test_wolfram_alpha_requires_query(monkeypatch):
    monkeypatch.setattr(math_actions, "APP_ID", "app-id")

    with pytest.raises(Exception, match="No query was provided"):
        await math_actions.wolfram_alpha_request()


@pytest.mark.asyncio
async def test_wolfram_alpha_app_id_missing(monkeypatch):
    monkeypatch.setattr(math_actions, "APP_ID", None)

    result = await math_actions.wolfram_alpha_request(context={"last_user_message": "2+2"})

    assert isinstance(result, ActionResult)
    assert result.return_value is False
    assert [event["intent"] for event in result.events if event["type"] == "BotIntent"] == [
        "inform wolfram alpha app id not set",
        "stop",
    ]


@pytest.mark.asyncio
async def test_wolfram_alpha_success(monkeypatch):
    monkeypatch.setattr(math_actions, "APP_ID", "app-id")
    monkeypatch.setattr(math_actions, "API_URL_BASE", "https://example.test/v2/result?appid=app-id")
    session = FakeWolframSession(FakeWolframResponse(200, "4"))
    monkeypatch.setattr(math_actions.aiohttp, "ClientSession", lambda: session)

    result = await math_actions.wolfram_alpha_request("2 + 2")

    assert result == "4"
    assert session.url == "https://example.test/v2/result?appid=app-id&i=2+%2B+2"


@pytest.mark.asyncio
async def test_wolfram_alpha_non_200_returns_action_result(monkeypatch):
    monkeypatch.setattr(math_actions, "APP_ID", "app-id")
    monkeypatch.setattr(math_actions, "API_URL_BASE", "https://example.test/v2/result?appid=app-id")
    session = FakeWolframSession(FakeWolframResponse(500, "error"))
    monkeypatch.setattr(math_actions.aiohttp, "ClientSession", lambda: session)

    result = await math_actions.wolfram_alpha_request("integrate x")

    assert isinstance(result, ActionResult)
    assert result.return_value is False
    assert [event["intent"] for event in result.events if event["type"] == "BotIntent"] == [
        "inform wolfram alpha not working",
        "stop",
    ]
