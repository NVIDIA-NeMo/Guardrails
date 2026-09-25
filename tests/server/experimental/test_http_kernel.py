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

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI

from nemoguardrails.server.experimental._buffered_kernel import (
    OperationBlocked,
    OperationCheckFailed,
)
from nemoguardrails.server.experimental._content_checker import (
    ContentAllowed,
    ContentBlocked,
    ContentCheckFailed,
    ContentInspectionPolicy,
    GuardedText,
)
from nemoguardrails.server.experimental._guarded_operation import BufferedGuardedOperation
from nemoguardrails.server.experimental._http_kernel import (
    BufferedHttpResponse,
    GuardedHttpOperation,
    create_http_proxy_router,
)


class StaticChecker:
    def __init__(self, input_decision=ContentAllowed(), output_decision=ContentAllowed()):
        self.input_decision = input_decision
        self.output_decision = output_decision
        self.calls = []
        self.policy_reads = 0

    def inspection_policy(self):
        self.policy_reads += 1
        return ContentInspectionPolicy(True, True)

    async def check_input(self, check):
        self.calls.append(("input", check))
        return self.input_decision

    async def check_output(self, check):
        self.calls.append(("output", check))
        return self.output_decision


def project_request(request):
    payload = json.loads(request.body)
    return GuardedText("user", payload["input"])


def project_response(response):
    payload = json.loads(response.body)
    return GuardedText("assistant", payload["output"])


@pytest.fixture
def guarded_operation():
    return GuardedHttpOperation(
        path="/v1/generate",
        method="POST",
        operation=BufferedGuardedOperation(
            name="test.generate",
            input_projection=project_request,
            output_projection=project_response,
        ),
    )


@pytest_asyncio.fixture
async def proxy_harness(guarded_operation):
    checker = StaticChecker()
    dispatched = []

    async def dispatch(request):
        dispatched.append(request)
        if request.path == "/v1/generate":
            return BufferedHttpResponse(
                status_code=201,
                headers=((b"content-type", b"application/json"), (b"x-provider-id", b"request-id")),
                body=b'{ "output" : "answer", "opaque" : 7 }',
            )
        return BufferedHttpResponse(
            status_code=202,
            headers=((b"content-type", b"application/octet-stream"), (b"x-provider-id", b"passthrough-id")),
            body=b"opaque-response",
        )

    def render_outcome(outcome):
        if isinstance(outcome, OperationBlocked):
            body = f"{outcome.stage.value}:{outcome.decision.message}".encode()
            return BufferedHttpResponse(400, ((b"content-type", b"text/plain"),), body)
        assert isinstance(outcome, OperationCheckFailed)
        body = f"{outcome.stage.value}:failed".encode()
        return BufferedHttpResponse(500, ((b"content-type", b"text/plain"),), body)

    app = FastAPI()
    app.include_router(
        create_http_proxy_router(
            operations=[guarded_operation],
            checker=checker,
            dispatch=dispatch,
            render_outcome=render_outcome,
        )
    )
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy.test")
    try:
        yield client, checker, dispatched
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_guarded_route_checks_content_and_preserves_provider_response(proxy_harness):
    client, checker, dispatched = proxy_harness
    body = b'{ "input" : "question", "opaque" : 3 }'

    response = await client.post(
        "/v1/generate?mode=provider",
        content=body,
        headers={"content-type": "application/json", "x-provider-option": "opaque"},
    )

    assert response.status_code == 201
    assert response.content == b'{ "output" : "answer", "opaque" : 7 }'
    assert response.headers["x-provider-id"] == "request-id"
    assert len(dispatched) == 1
    assert dispatched[0].method == "POST"
    assert dispatched[0].path == "/v1/generate"
    assert dispatched[0].query == b"mode=provider"
    assert dispatched[0].body == body
    assert (b"x-provider-option", b"opaque") in dispatched[0].headers
    assert [call[0] for call in checker.calls] == ["input", "output"]
    assert checker.calls[0][1].subject.content == "question"
    assert checker.calls[1][1].output_subject.content == "answer"


@pytest.mark.asyncio
async def test_catchall_forwards_without_calling_checker(proxy_harness):
    client, checker, dispatched = proxy_harness

    response = await client.patch(
        "/v1/provider-owned?opaque=yes",
        content=b"opaque-request",
        headers={"content-type": "application/octet-stream", "x-provider-option": "preserve"},
    )

    assert response.status_code == 202
    assert response.content == b"opaque-response"
    assert response.headers["x-provider-id"] == "passthrough-id"
    assert checker.calls == []
    assert len(dispatched) == 1
    assert dispatched[0].method == "PATCH"
    assert dispatched[0].path == "/v1/provider-owned"
    assert dispatched[0].query == b"opaque=yes"
    assert dispatched[0].body == b"opaque-request"
    assert (b"x-provider-option", b"preserve") in dispatched[0].headers


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("decision", "expected_status", "expected_body"),
    [
        (ContentBlocked("blocked"), 400, b"input:blocked"),
        (ContentCheckFailed("unavailable"), 500, b"input:failed"),
    ],
)
async def test_input_outcome_is_rendered_without_dispatch(guarded_operation, decision, expected_status, expected_body):
    checker = StaticChecker(input_decision=decision)
    dispatched = []

    async def dispatch(request):
        dispatched.append(request)
        raise AssertionError("a stopped input must not be dispatched")

    def render_outcome(outcome):
        if isinstance(outcome, OperationBlocked):
            return BufferedHttpResponse(400, (), f"{outcome.stage.value}:{outcome.decision.message}".encode())
        return BufferedHttpResponse(500, (), f"{outcome.stage.value}:failed".encode())

    app = FastAPI()
    app.include_router(
        create_http_proxy_router(
            operations=[guarded_operation],
            checker=checker,
            dispatch=dispatch,
            render_outcome=render_outcome,
        )
    )
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy.test") as client:
        response = await client.post("/v1/generate", json={"input": "question"})

    assert response.status_code == expected_status
    assert response.content == expected_body
    assert dispatched == []


@pytest.mark.asyncio
async def test_output_block_hides_provider_response(proxy_harness):
    client, checker, dispatched = proxy_harness
    checker.output_decision = ContentBlocked("blocked")

    response = await client.post("/v1/generate", json={"input": "question"})

    assert response.status_code == 400
    assert response.content == b"output:blocked"
    assert len(dispatched) == 1
    assert checker.calls[-1][0] == "output"


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/v1/generate/", "/v1//generate"])
async def test_guarded_path_variants_cannot_bypass_into_catchall(proxy_harness, path):
    client, checker, dispatched = proxy_harness

    response = await client.post(path, json={"input": "question"})

    assert response.status_code == 422
    assert dispatched == []
    assert checker.calls == []


@pytest.mark.asyncio
async def test_wrong_method_for_guarded_path_is_not_forwarded(proxy_harness):
    client, checker, dispatched = proxy_harness

    response = await client.put("/v1/generate", json={"input": "question"})

    assert response.status_code == 405
    assert response.headers["allow"] == "POST"
    assert dispatched == []
    assert checker.calls == []


@pytest.mark.asyncio
async def test_reserved_application_route_cannot_fall_through_to_provider(guarded_operation):
    dispatched = []

    async def dispatch(request):
        dispatched.append(request)
        return BufferedHttpResponse(200, (), b"provider")

    app = FastAPI()

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    app.include_router(
        create_http_proxy_router(
            operations=[guarded_operation],
            checker=StaticChecker(),
            dispatch=dispatch,
            render_outcome=lambda _outcome: BufferedHttpResponse(400, (), b"blocked"),
            reserved_routes={"/health": {"GET"}},
        )
    )
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy.test") as client:
        owned = await client.get("/health")
        wrong_method = await client.post("/health")

    assert owned.status_code == 200
    assert owned.json() == {"status": "ok"}
    assert wrong_method.status_code == 405
    assert wrong_method.headers["allow"] == "GET"
    assert dispatched == []


@pytest.mark.asyncio
async def test_checker_policy_is_bound_once_for_all_requests(proxy_harness):
    client, checker, _dispatched = proxy_harness

    await client.post("/v1/generate", json={"input": "one"})
    await client.post("/v1/generate", json={"input": "two"})

    assert checker.policy_reads == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/v1/generate", "/v1/provider-owned"])
async def test_buffered_request_limit_fails_before_dispatch(guarded_operation, path):
    dispatched = []

    async def dispatch(request):
        dispatched.append(request)
        return BufferedHttpResponse(200, (), b"response")

    app = FastAPI()
    app.include_router(
        create_http_proxy_router(
            operations=[guarded_operation],
            checker=StaticChecker(),
            dispatch=dispatch,
            render_outcome=lambda _outcome: BufferedHttpResponse(400, (), b"blocked"),
            max_request_body_bytes=4,
        )
    )
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy.test") as client:
        response = await client.post(path, content=b"12345")

    assert response.status_code == 413
    assert dispatched == []


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/v1/generate", "/v1/provider-owned"])
async def test_buffered_response_limit_hides_upstream_body(guarded_operation, path):
    async def dispatch(_request):
        return BufferedHttpResponse(200, (), b"12345")

    app = FastAPI()
    app.include_router(
        create_http_proxy_router(
            operations=[guarded_operation],
            checker=StaticChecker(),
            dispatch=dispatch,
            render_outcome=lambda _outcome: BufferedHttpResponse(400, (), b"no"),
            max_response_body_bytes=4,
        )
    )
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy.test") as client:
        response = await client.post(path, json={"input": "question"})

    assert response.status_code == 502
    assert response.content == b""


def test_duplicate_guarded_route_is_rejected(guarded_operation):
    duplicate = GuardedHttpOperation(
        path="/v1/generate",
        method="POST",
        operation=BufferedGuardedOperation(
            name="test.duplicate",
            input_projection=project_request,
            output_projection=project_response,
        ),
    )

    with pytest.raises(ValueError, match="routes must be unique"):
        create_http_proxy_router(
            operations=[guarded_operation, duplicate],
            checker=StaticChecker(),
            dispatch=lambda _request: None,
            render_outcome=lambda _outcome: None,
        )


def test_guarded_http_path_rejects_path_spanning_parameters():
    with pytest.raises(ValueError, match="path-spanning"):
        GuardedHttpOperation(
            path="/v1/{rest:path}",
            method="POST",
            operation=BufferedGuardedOperation(
                name="test.invalid",
                input_projection=project_request,
                output_projection=project_response,
            ),
        )
