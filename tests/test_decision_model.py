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

import json

import pytest

from nemoguardrails import RailsConfig
from nemoguardrails.decisions.model import decision_models
from nemoguardrails.decisions.types import NoulAnswer, NoulQuestion
from nemoguardrails.http import HTTPResponse
from nemoguardrails.testing import RecordingHTTPClient


def response(name, answer):
    return HTTPResponse(status_code=200, content=json.dumps({"answers": {name: answer}}).encode())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("engine", "base_url", "endpoint", "option", "value"),
    [
        (
            "system_one",
            "http://localhost:8000/v1/",
            "http://localhost:8000/v1/systemone",
            "chat_template_kwargs",
            {"enable_thinking": False},
        ),
        (
            "openai_decisions",
            "https://proxy.example/openai",
            "https://proxy.example/openai/v1/decisions",
            "prompt_cache_options",
            {"ttl": "30m"},
        ),
    ],
)
async def test_decision_request_parameters(engine, base_url, endpoint, option, value, monkeypatch):
    monkeypatch.setenv("JUDGE_TOKEN", "test-token")
    config = RailsConfig.from_content(
        config={
            "models": [
                {
                    "type": "judge",
                    "engine": engine,
                    "model": "served-model",
                    "api_key_env_var": "JUDGE_TOKEN",
                    "parameters": {"base_url": base_url, "timeout": 2.5, option: value},
                }
            ]
        }
    )
    state = {"bot_message": "Hello."}
    if engine == "system_one":
        body = {"state": state, "questions": {"greeting": {"type": "noul", "instructions": "A greeting?"}}}
        answer = response("greeting", {"type": "noul", "noul": 0.9})
    else:
        body = {
            "input": [{"role": "user", "content": [{"type": "input_json", "data": state}]}],
            "questions": [{"type": "predicate", "name": "greeting", "instructions": "A greeting?"}],
        }
        answer = HTTPResponse(200, content=b'{"answers":[{"type":"predicate","name":"greeting","probability":0.9}]}')
    http = RecordingHTTPClient([answer])
    result = await decision_models(config.models)["judge"].decide(
        "greeting", NoulQuestion("A greeting?"), state, http_client=http
    )
    assert result == NoulAnswer(0.9)
    (request,) = http.requests
    assert (request.method, request.url, request.timeout) == ("POST", endpoint, 2.5)
    assert request.headers == {"Authorization": "Bearer test-token"}
    assert request.json == {"model": "served-model", option: value, **body}
