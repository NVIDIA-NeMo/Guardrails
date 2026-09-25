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

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI

from nemoguardrails.server.experimental._content_checker import (
    ContentAllowed,
    ContentBlocked,
    ContentCheckFailed,
    ContentInspectionPolicy,
)
from nemoguardrails.server.experimental._http_kernel import BufferedHttpResponse
from nemoguardrails.server.experimental.providers.openai.errors import (
    OPENAI_ERROR_MAPPING,
    OpenAIProxyErrorResponse,
)
from nemoguardrails.server.experimental.providers.openai.integration import create_openai_chat_router


class StaticChecker:
    """Return configured inspection outcomes while recording calls."""

    def __init__(self, input_decision=ContentAllowed(), output_decision=ContentAllowed()):
        """Configure input and output outcomes for one test."""
        self.input_decision = input_decision
        self.output_decision = output_decision
        self.calls = []
        self.policy_reads = 0

    def inspection_policy(self):
        """Enable both input and output inspection."""
        self.policy_reads += 1
        return ContentInspectionPolicy(True, True)

    async def check_input(self, check):
        """Record an input check and return its configured outcome."""
        self.calls.append(("input", check))
        return self.input_decision

    async def check_output(self, check):
        """Record an output check and return its configured outcome."""
        self.calls.append(("output", check))
        return self.output_decision


def _request_body():
    """Return a representative OpenAI Chat request body."""
    return b'{ "messages" : [ { "role" : "user", "content" : "question" } ], "model" : "gpt-example", "opaque" : { "keep" : true } }'


def _response_body():
    """Return a representative OpenAI Chat response body."""
    return b'{ "id" : "chatcmpl-example", "choices" : [ { "index" : 0, "message" : { "role" : "assistant", "content" : "answer" } } ], "opaque" : [ 1, 2 ] }'


def _json_headers(*headers):
    """Return JSON content headers with optional provider metadata."""
    return ((b"content-type", b"application/json"), *headers)


@pytest_asyncio.fixture
async def proxy_harness():
    """Provide a proxy client, checker, and dispatched-request log."""
    checker = StaticChecker()
    dispatched = []

    async def dispatch(request):
        """Return deterministic Chat or catch-all provider responses."""
        dispatched.append(request)
        if request.path == "/v1/chat/completions":
            return BufferedHttpResponse(
                200,
                ((b"content-type", b"application/json"), (b"x-request-id", b"provider-chat-id")),
                _response_body(),
            )
        return BufferedHttpResponse(
            200,
            ((b"content-type", b"application/json"), (b"x-request-id", b"provider-models-id")),
            b'{"object":"list","data":[]}',
        )

    app = FastAPI()
    app.include_router(create_openai_chat_router(checker=checker, dispatch=dispatch))
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy.test")
    try:
        yield client, checker, dispatched
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_openai_chat_passes_checks_without_rewriting_provider_http(proxy_harness):
    """Guarded Chat passes both checks without rewriting provider HTTP data."""
    client, checker, dispatched = proxy_harness
    request_body = _request_body()

    response = await client.post(
        "/v1/chat/completions?provider_option=opaque",
        content=request_body,
        headers={"content-type": "application/json", "x-provider-option": "preserve"},
    )

    assert response.status_code == 200
    assert response.content == _response_body()
    assert response.headers["x-request-id"] == "provider-chat-id"
    assert len(dispatched) == 1
    assert dispatched[0].body == request_body
    assert dispatched[0].query == b"provider_option=opaque"
    assert (b"x-provider-option", b"preserve") in dispatched[0].headers
    assert [call[0] for call in checker.calls] == ["input", "output"]
    assert checker.calls[0][1].message.content == "question"
    assert checker.calls[1][1].input_message.content == "question"
    assert checker.calls[1][1].output_content == "answer"


@pytest.mark.asyncio
async def test_openai_catchall_forwards_without_checking_content(proxy_harness):
    """Unbound provider routes bypass content inspection and preserve HTTP data."""
    client, checker, dispatched = proxy_harness

    response = await client.get("/v1/models?limit=10", headers={"x-provider-option": "preserve"})

    assert response.status_code == 200
    assert response.content == b'{"object":"list","data":[]}'
    assert response.headers["x-request-id"] == "provider-models-id"
    assert checker.calls == []
    assert dispatched[0].path == "/v1/models"
    assert dispatched[0].query == b"limit=10"


@pytest.mark.asyncio
async def test_future_chat_api_version_cannot_bypass_guarded_projection(proxy_harness):
    """A future Chat API route cannot escape the guarded operation boundary."""
    client, checker, dispatched = proxy_harness

    response = await client.post(
        "/v2/chat/completions",
        content=_request_body(),
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 422
    assert checker.calls == []
    assert dispatched == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("decision", "expected_type", "expected_code"),
    [
        (ContentBlocked("Request blocked.", "policy_rule"), "guardrails_violation", "content_blocked"),
        (ContentCheckFailed("internal checker detail"), "proxy_error", "guardrails_input_check_failed"),
    ],
)
async def test_input_guard_outcome_is_openai_shaped_and_prevents_dispatch(decision, expected_type, expected_code):
    """Stopped input produces an OpenAI error without provider dispatch."""
    checker = StaticChecker(input_decision=decision)
    dispatched = []

    async def dispatch(request):
        """Fail if a stopped input reaches provider dispatch."""
        dispatched.append(request)
        raise AssertionError("a stopped input must not be dispatched")

    app = FastAPI()
    app.include_router(create_openai_chat_router(checker=checker, dispatch=dispatch))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy.test") as client:
        response = await client.post(
            "/v1/chat/completions",
            content=_request_body(),
            headers={"content-type": "application/json"},
        )

    assert response.status_code == (400 if isinstance(decision, ContentBlocked) else 502)
    assert response.json()["error"]["type"] == expected_type
    assert response.json()["error"]["code"] == expected_code
    assert response.json()["error"]["param"] == ("policy_rule" if isinstance(decision, ContentBlocked) else None)
    assert "internal checker detail" not in response.text
    assert dispatched == []


@pytest.mark.asyncio
async def test_output_block_hides_provider_response_in_openai_error():
    """Blocked output hides the provider body and headers behind an OpenAI error."""
    checker = StaticChecker(output_decision=ContentBlocked("Response blocked."))
    provider_body = _response_body()

    async def dispatch(_request):
        """Return a provider response containing data that must not escape."""
        return BufferedHttpResponse(200, _json_headers((b"x-provider-secret", b"hidden")), provider_body)

    app = FastAPI()
    app.include_router(create_openai_chat_router(checker=checker, dispatch=dispatch))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy.test") as client:
        response = await client.post(
            "/v1/chat/completions",
            content=_request_body(),
            headers={"content-type": "application/json"},
        )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "content_blocked"
    assert provider_body not in response.content
    assert "x-provider-secret" not in response.headers


@pytest.mark.asyncio
async def test_unsupported_request_is_openai_shaped_and_not_dispatched():
    """Unsupported Chat shapes produce an OpenAI error before dispatch."""
    dispatched = []

    async def dispatch(request):
        """Fail if an unsupported request reaches provider dispatch."""
        dispatched.append(request)
        raise AssertionError("an unsupported request must not be dispatched")

    app = FastAPI()
    app.include_router(create_openai_chat_router(checker=StaticChecker(), dispatch=dispatch))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy.test") as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "question"}], "stream": True},
        )

    assert response.status_code == 422
    assert response.json()["error"]["type"] == "unsupported_request"
    assert response.json()["error"]["code"] == "unsupported_chat_completions_shape"
    assert dispatched == []


@pytest.mark.asyncio
async def test_unsupported_success_response_is_hidden_in_openai_error():
    """Unsupported successful responses are hidden behind an OpenAI error."""
    provider_body = b'{"choices":[{"message":{"role":"assistant","content":"answer","tool_calls":[]}}]}'

    async def dispatch(_request):
        """Return an unsupported provider response with private metadata."""
        return BufferedHttpResponse(200, _json_headers((b"x-provider-secret", b"hidden")), provider_body)

    app = FastAPI()
    app.include_router(create_openai_chat_router(checker=StaticChecker(), dispatch=dispatch))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy.test") as client:
        response = await client.post(
            "/v1/chat/completions",
            content=_request_body(),
            headers={"content-type": "application/json"},
        )

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "unsupported_chat_completions_response_shape"
    assert provider_body not in response.content
    assert "x-provider-secret" not in response.headers


@pytest.mark.asyncio
async def test_provider_error_response_is_preserved_without_output_check():
    """Provider error responses pass through without output inspection."""
    checker = StaticChecker()
    provider_body = b'{ "error" : { "message" : "rate limited", "type" : "provider_error" } }'

    async def dispatch(_request):
        """Return a provider-native error response."""
        return BufferedHttpResponse(
            429,
            ((b"content-type", b"application/json"), (b"x-request-id", b"provider-error-id")),
            provider_body,
        )

    app = FastAPI()
    app.include_router(create_openai_chat_router(checker=checker, dispatch=dispatch))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy.test") as client:
        response = await client.post(
            "/v1/chat/completions",
            content=_request_body(),
            headers={"content-type": "application/json"},
        )

    assert response.status_code == 429
    assert response.content == provider_body
    assert response.headers["x-request-id"] == "provider-error-id"
    assert [call[0] for call in checker.calls] == ["input"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("checker", "status_code", "code", "dispatches"),
    [
        (StaticChecker(input_decision=ContentAllowed("replacement")), 422, "request_content_not_replaceable", 0),
        (StaticChecker(output_decision=ContentAllowed("replacement")), 502, "response_content_not_replaceable", 1),
    ],
)
async def test_replacement_remains_explicitly_unsupported(checker, status_code, code, dispatches):
    """Replacement outcomes stay unsupported until buffered replacement lands."""
    dispatched = []

    async def dispatch(request):
        """Record dispatch and return a successful provider response."""
        dispatched.append(request)
        return BufferedHttpResponse(200, _json_headers(), _response_body())

    app = FastAPI()
    app.include_router(create_openai_chat_router(checker=checker, dispatch=dispatch))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy.test") as client:
        response = await client.post(
            "/v1/chat/completions",
            content=_request_body(),
            headers={"content-type": "application/json"},
        )

    assert response.status_code == status_code
    assert response.json()["error"]["code"] == code
    assert len(dispatched) == dispatches


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("content", "headers", "status_code", "code"),
    [
        (b"not json", {"content-type": "application/json"}, 400, "invalid_json"),
        (_request_body(), {}, 415, "unsupported_media_type"),
        (
            _request_body(),
            {"content-type": "application/json", "content-encoding": "gzip"},
            415,
            "unsupported_content_encoding",
        ),
    ],
)
async def test_request_representation_failures_are_native_and_stop_before_dispatch(
    content,
    headers,
    status_code,
    code,
):
    """Invalid request representations produce native errors before dispatch."""
    dispatched = []

    async def dispatch(request):
        """Fail if an invalid representation reaches provider dispatch."""
        dispatched.append(request)
        raise AssertionError("an invalid guarded representation must not be dispatched")

    app = FastAPI()
    app.include_router(create_openai_chat_router(checker=StaticChecker(), dispatch=dispatch))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy.test") as client:
        response = await client.post("/v1/chat/completions", content=content, headers=headers)

    assert response.status_code == status_code
    assert response.json()["error"]["code"] == code
    assert dispatched == []


def test_openai_error_mapping_is_the_openapi_response_authority():
    """The runtime error mapping also supplies the OpenAPI response models."""
    app = FastAPI()

    async def dispatch(_request):
        """Fail if OpenAPI generation attempts provider dispatch."""
        raise AssertionError

    app.include_router(create_openai_chat_router(checker=StaticChecker(), dispatch=dispatch))
    operation = app.openapi()["paths"]["/v1/chat/completions"]["post"]

    assert operation["requestBody"]["required"] is True
    assert operation["requestBody"]["content"]["application/json"]["schema"]["title"] == (
        "ChatCompletionsGuardedRequest"
    )
    assert set(operation["responses"]) == {
        "200",
        *(str(response.status_code) for response in OPENAI_ERROR_MAPPING.responses),
    }
    for response in OPENAI_ERROR_MAPPING.responses:
        documented = operation["responses"][str(response.status_code)]
        assert documented["description"] == response.description
        assert documented["content"]["application/json"]["schema"]["$ref"].endswith(
            f"/{OpenAIProxyErrorResponse.__name__}"
        )
