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

"""Token limits survive HTTP request validation and reach the main model."""

import json
from unittest.mock import AsyncMock

import pytest
from aioresponses import aioresponses
from fastapi.testclient import TestClient

from nemoguardrails import Guardrails, RailsConfig
from nemoguardrails.rails import LLMRails
from nemoguardrails.server import api

MODEL_URL = "https://upstream.invalid/v1/chat/completions"


@pytest.fixture(params=["llmrails", "iorails"])
def token_limit_client(request, monkeypatch):
    config = RailsConfig.from_content(
        config={
            "models": [
                {
                    "type": "main",
                    "engine": "nim",
                    "model": "test-model",
                    "parameters": {"base_url": "https://upstream.invalid/v1", "api_key": "test-key"},
                }
            ]
        }
    )
    if request.param == "iorails":
        rails = Guardrails(config, require_iorails=True)
        assert rails.use_iorails_engine
    else:
        rails = LLMRails(config)
    monkeypatch.setattr(api, "_get_rails", AsyncMock(return_value=rails))
    with TestClient(api.app) as client:
        try:
            yield client, request.param
        finally:
            if isinstance(rails, Guardrails):
                assert client.portal is not None
                client.portal.call(rails.shutdown)


@pytest.mark.httpx_mock(should_mock=lambda request: request.url.host != "testserver")
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "token_params,option_params,expected_params",
    [
        ({"max_completion_tokens": 20}, {}, {"max_completion_tokens": 20}),
        ({"max_tokens": 20}, {}, {"max_tokens": 20}),
        ({"max_tokens": 30, "max_completion_tokens": 20}, {}, {"max_tokens": 30, "max_completion_tokens": 20}),
        ({}, {}, {}),
        ({"max_tokens": None, "max_completion_tokens": None}, {}, {}),
        ({"max_completion_tokens": 20}, {"max_completion_tokens": 40}, {"max_completion_tokens": 20}),
        ({"max_completion_tokens": None}, {"max_completion_tokens": 40}, {"max_completion_tokens": 40}),
    ],
)
def test_token_limits_reach_main_model(
    token_limit_client, httpx_mock, stream, token_params, option_params, expected_params
):
    client, engine = token_limit_client
    choice = {"index": 0, "finish_reason": "stop"}
    completion = {"id": "test-completion", "object": "chat.completion", "created": 0, "model": "test-model"}
    if stream:
        completion.update(
            object="chat.completion.chunk",
            choices=[{**choice, "delta": {"role": "assistant", "content": "Hello!"}}],
        )
        body = f"data: {json.dumps(completion)}\n\ndata: [DONE]\n\n"
        headers = {"Content-Type": "text/event-stream"}
    else:
        completion["choices"] = [{**choice, "message": {"role": "assistant", "content": "Hello!"}}]
        body = json.dumps(completion)
        headers = {"Content-Type": "application/json"}

    with aioresponses() as aiohttp_mock:
        if engine == "iorails":
            aiohttp_mock.post(MODEL_URL, body=body, headers=headers)
        else:
            httpx_mock.add_response(url=MODEL_URL, method="POST", content=body.encode(), headers=headers)

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": stream,
                "guardrails": {"config_id": "test", "options": {"llm_params": option_params}},
                **token_params,
            },
        )
        assert response.status_code == 200, response.text
        if stream:
            assert "Hello!" in response.text
            assert "data: [DONE]" in response.text
        else:
            assert response.json()["choices"][0]["message"]["content"] == "Hello!"

        if engine == "iorails":
            requests = [call.kwargs["json"] for calls in aiohttp_mock.requests.values() for call in calls]
        else:
            requests = [json.loads(request.content) for request in httpx_mock.get_requests()]
        assert len(requests) == 1
        assert {
            key: requests[0][key] for key in ("max_tokens", "max_completion_tokens") if key in requests[0]
        } == expected_params
