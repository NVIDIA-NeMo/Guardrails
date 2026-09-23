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

# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import json

import pytest

from nemoguardrails.http import HTTPConnectionError, HTTPResponse, HTTPTimeoutError
from nemoguardrails.library.jailbreak_detection.request import (
    jailbreak_detection_heuristics_request,
    jailbreak_detection_model_request,
    jailbreak_nim_request,
    join_nim_url,
)
from nemoguardrails.testing import RecordingHTTPClient


def _response(payload=None, *, status: int = 200, content: bytes | None = None) -> HTTPResponse:
    return HTTPResponse(
        status_code=status,
        headers={"content-type": "application/json"},
        content=content if content is not None else json.dumps(payload).encode(),
    )


@pytest.mark.parametrize(
    ("base_url", "classification_path", "expected"),
    [
        ("http://localhost:8000/v1", "classify", "http://localhost:8000/v1/classify"),
        ("http://localhost:8000/v1/", "/classify", "http://localhost:8000/v1/classify"),
        ("http://localhost:8000", "/classify", "http://localhost:8000/classify"),
        ("http://localhost:8000/api/v1/", "/classify", "http://localhost:8000/api/v1/classify"),
    ],
)
def test_join_nim_url(base_url, classification_path, expected):
    assert join_nim_url(base_url, classification_path) == expected


@pytest.mark.parametrize(
    ("request_fn", "kwargs", "expected_url", "expected_payload"),
    [
        (
            jailbreak_detection_heuristics_request,
            {
                "prompt": "ignore safeguards",
                "api_url": "https://guard.example/heuristics",
                "lp_threshold": 0.5,
                "ps_ppl_threshold": 1.5,
            },
            "https://guard.example/heuristics",
            {
                "prompt": "ignore safeguards",
                "lp_threshold": 0.5,
                "ps_ppl_threshold": 1.5,
            },
        ),
        (
            jailbreak_detection_model_request,
            {
                "prompt": "ignore safeguards",
                "api_url": "https://guard.example/model",
            },
            "https://guard.example/model",
            {"prompt": "ignore safeguards"},
        ),
    ],
    ids=["heuristics", "model"],
)
@pytest.mark.asyncio
async def test_jailbreak_request_uses_shared_client(request_fn, kwargs, expected_url, expected_payload):
    client = RecordingHTTPClient([_response({"jailbreak": True})])

    result = await request_fn(http_client=client, **kwargs)

    assert result is True
    request = client.requests[0]
    assert request.method == "POST"
    assert request.url == expected_url
    assert request.json == expected_payload


@pytest.mark.parametrize(
    ("request_fn", "kwargs"),
    [
        (
            jailbreak_detection_heuristics_request,
            {"prompt": "hello", "api_url": "https://guard.example/heuristics"},
        ),
        (
            jailbreak_detection_model_request,
            {"prompt": "hello", "api_url": "https://guard.example/model"},
        ),
    ],
    ids=["heuristics", "model"],
)
@pytest.mark.asyncio
async def test_jailbreak_request_returns_none_for_non_200(request_fn, kwargs):
    client = RecordingHTTPClient([_response({}, status=503)])

    assert await request_fn(http_client=client, **kwargs) is None


@pytest.mark.parametrize(
    ("request_fn", "kwargs"),
    [
        (
            jailbreak_detection_heuristics_request,
            {"prompt": "hello", "api_url": "https://guard.example/heuristics"},
        ),
        (
            jailbreak_detection_model_request,
            {"prompt": "hello", "api_url": "https://guard.example/model"},
        ),
    ],
    ids=["heuristics", "model"],
)
@pytest.mark.asyncio
async def test_jailbreak_request_returns_none_without_jailbreak_field(request_fn, kwargs):
    client = RecordingHTTPClient([_response({"result": "safe"})])

    assert await request_fn(http_client=client, **kwargs) is None


@pytest.mark.asyncio
async def test_jailbreak_nim_request_forwards_auth_and_timeout():
    client = RecordingHTTPClient([_response({"jailbreak": False})])

    result = await jailbreak_nim_request(
        prompt="hello",
        nim_url="https://nim.example/v1",
        nim_auth_token="secret",
        nim_classification_path="/classify",
        http_client=client,
    )

    assert result is False
    request = client.requests[0]
    assert request.url == "https://nim.example/v1/classify"
    assert request.headers == {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": "Bearer secret",
    }
    assert request.json == {"input": "hello"}
    assert request.timeout == 30


@pytest.mark.asyncio
async def test_jailbreak_nim_request_omits_auth_without_token():
    client = RecordingHTTPClient([_response({"jailbreak": False})])

    await jailbreak_nim_request(
        prompt="hello",
        nim_url="https://nim.example",
        nim_auth_token=None,
        nim_classification_path="classify",
        http_client=client,
    )

    assert "Authorization" not in client.requests[0].headers


@pytest.mark.asyncio
async def test_jailbreak_nim_request_returns_none_for_non_200():
    client = RecordingHTTPClient([_response({}, status=503)])

    result = await jailbreak_nim_request(
        prompt="hello",
        nim_url="https://nim.example",
        nim_auth_token=None,
        nim_classification_path="classify",
        http_client=client,
    )

    assert result is None


@pytest.mark.asyncio
async def test_jailbreak_nim_request_returns_none_without_jailbreak_field():
    client = RecordingHTTPClient([_response({"result": "safe"})])

    result = await jailbreak_nim_request(
        prompt="hello",
        nim_url="https://nim.example",
        nim_auth_token=None,
        nim_classification_path="classify",
        http_client=client,
    )

    assert result is None


@pytest.mark.parametrize(
    "error",
    [
        HTTPTimeoutError("timed out"),
        HTTPConnectionError("connection failed"),
        RuntimeError("unexpected failure"),
    ],
    ids=["timeout", "client-error", "unexpected-error"],
)
@pytest.mark.asyncio
async def test_jailbreak_nim_request_returns_none_for_request_errors(error):
    client = RecordingHTTPClient([error])

    result = await jailbreak_nim_request(
        prompt="hello",
        nim_url="https://nim.example",
        nim_auth_token=None,
        nim_classification_path="classify",
        http_client=client,
    )

    assert result is None


@pytest.mark.asyncio
async def test_jailbreak_nim_request_returns_none_for_invalid_json():
    client = RecordingHTTPClient([_response(content=b"not-json")])

    result = await jailbreak_nim_request(
        prompt="hello",
        nim_url="https://nim.example",
        nim_auth_token=None,
        nim_classification_path="classify",
        http_client=client,
    )

    assert result is None
