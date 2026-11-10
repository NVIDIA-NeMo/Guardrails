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
from nemoguardrails.library.jailbreak_detection.errors import JailbreakDetectionError
from nemoguardrails.library.jailbreak_detection.model_based.errors import JailbreakInputTooLongError
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


@pytest.mark.parametrize("backend", ["model", "nim"])
@pytest.mark.asyncio
async def test_jailbreak_model_request_rejects_oversized_input(backend):
    client = RecordingHTTPClient([_response(status=413, content=b"untrusted response body")])

    with pytest.raises(JailbreakInputTooLongError, match="input limit"):
        if backend == "model":
            await jailbreak_detection_model_request(prompt="long input", http_client=client)
        else:
            await jailbreak_nim_request(
                prompt="long input",
                nim_url="https://nim.example",
                nim_auth_token=None,
                nim_classification_path="classify",
                http_client=client,
            )


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


@pytest.fixture(params=["heuristics", "model", "nim"])
def request_backend(request):
    if request.param == "nim":
        return jailbreak_nim_request, {
            "nim_url": "https://detector.example",
            "nim_auth_token": None,
            "nim_classification_path": "classify",
        }
    fn = jailbreak_detection_model_request if request.param == "model" else jailbreak_detection_heuristics_request
    return fn, {"api_url": "https://detector.example/" + request.param}


@pytest.mark.parametrize("status", [401, 403, 429, 500, 503])
@pytest.mark.asyncio
async def test_jailbreak_request_error_status_raises(request_backend, status):
    fn, kwargs = request_backend
    client = RecordingHTTPClient([_response(status=status, content=b"private response body")])
    with pytest.raises(RuntimeError, match=f"HTTP {status}"):
        await fn(prompt="hello", http_client=client, **kwargs)


@pytest.mark.parametrize("payload", [{}, {"jailbreak": None}, {"jailbreak": 0}, {"jailbreak": "false"}, [], None])
@pytest.mark.asyncio
async def test_jailbreak_request_invalid_verdict_raises(request_backend, payload):
    fn, kwargs = request_backend
    client = RecordingHTTPClient([_response(payload)])
    with pytest.raises(RuntimeError, match="boolean jailbreak field"):
        await fn(prompt="hello", http_client=client, **kwargs)


@pytest.mark.asyncio
async def test_jailbreak_request_invalid_json_raises(request_backend):
    fn, kwargs = request_backend
    client = RecordingHTTPClient([_response(content=b"private invalid json")])
    with pytest.raises(RuntimeError, match="invalid JSON"):
        await fn(prompt="hello", http_client=client, **kwargs)


@pytest.mark.parametrize("error_type", [HTTPTimeoutError, HTTPConnectionError, RuntimeError])
@pytest.mark.asyncio
async def test_jailbreak_request_transport_error_propagates(request_backend, error_type):
    fn, kwargs = request_backend
    error = error_type("detector unavailable")
    client = RecordingHTTPClient([error])
    expected_type = RuntimeError if error_type is RuntimeError else JailbreakDetectionError
    with pytest.raises(expected_type) as caught:
        await fn(prompt="hello", http_client=client, **kwargs)
    if error_type is RuntimeError:
        assert caught.value is error
    else:
        assert caught.value.__cause__ is error


@pytest.mark.parametrize("verdict", [False, True])
@pytest.mark.asyncio
async def test_jailbreak_request_preserves_boolean_verdict(request_backend, verdict):
    fn, kwargs = request_backend
    client = RecordingHTTPClient([_response({"jailbreak": verdict})])
    assert await fn(prompt="hello", http_client=client, **kwargs) is verdict
