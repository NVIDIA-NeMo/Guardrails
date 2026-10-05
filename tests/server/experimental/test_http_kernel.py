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

"""Test transparent HTTP proxying for buffered provider operations."""

import json

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI
from starlette.requests import Request

from nemoguardrails.server.experimental._buffered_kernel import (
    OperationBlocked,
    OperationCheckFailed,
)
from nemoguardrails.server.experimental._content_checker import (
    ContentAllowed,
    ContentBlocked,
    ContentCheckFailed,
    ContentInspectionPolicy,
)
from nemoguardrails.server.experimental._guarded_operation import BufferedGuardedOperation
from nemoguardrails.server.experimental._http_kernel import (
    BufferedHttpResponse,
    GuardedHttpOperation,
    GuardedOperationPath,
    HttpDispatchFailed,
    HttpFailureKind,
    HttpOperationFailed,
    HttpRouteRejected,
    HttpRouteRejectionKind,
    _render_response,
    _request_path,
    create_http_proxy_router,
)
from nemoguardrails.server.experimental.provider.types import GuardedMessage


class StaticChecker:
    """Provide configurable content-check results and observable calls."""

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
    return GuardedMessage("user", payload["input"])


def project_response(response):
    payload = json.loads(response.body)
    return GuardedMessage("assistant", payload["output"])


def render_test_outcome(outcome):
    if isinstance(outcome, HttpRouteRejected):
        status_codes = {
            HttpRouteRejectionKind.METHOD_NOT_ALLOWED: 405,
            HttpRouteRejectionKind.NON_CANONICAL_PATH: 422,
            HttpRouteRejectionKind.INVALID_REQUEST_PATH: 400,
        }
        return BufferedHttpResponse(status_codes[outcome.kind], (), outcome.kind.value.encode())
    if isinstance(outcome, HttpOperationFailed):
        status_codes = {
            HttpFailureKind.INVALID_CONTENT_LENGTH: 400,
            HttpFailureKind.REQUEST_BODY_TOO_LARGE: 413,
            HttpFailureKind.UPSTREAM_REQUEST_FAILED: 502,
            HttpFailureKind.RESPONSE_BODY_TOO_LARGE: 502,
        }
        return BufferedHttpResponse(status_codes[outcome.kind], (), outcome.kind.value.encode())
    if isinstance(outcome, OperationBlocked):
        body = f"{outcome.stage.value}:{outcome.decision.message}".encode()
        return BufferedHttpResponse(400, ((b"content-type", b"text/plain"),), body)
    assert isinstance(outcome, OperationCheckFailed)
    body = f"{outcome.stage.value}:failed".encode()
    return BufferedHttpResponse(500, ((b"content-type", b"text/plain"),), body)


@pytest.fixture
def guarded_operation():
    return GuardedHttpOperation(
        operation_path=GuardedOperationPath("/v1/generate"),
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

    app = FastAPI()
    app.include_router(
        create_http_proxy_router(
            operations=[guarded_operation],
            checker=checker,
            dispatch=dispatch,
            render_outcome=render_test_outcome,
        )
    )
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy.test")
    try:
        yield client, checker, dispatched
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_guarded_route_checks_content_and_preserves_provider_response(proxy_harness):
    """Check configured content while preserving the provider response."""

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
    assert checker.calls[0][1].message.content == "question"
    assert checker.calls[1][1].output_content == "answer"


@pytest.mark.asyncio
async def test_catchall_forwards_without_calling_checker(proxy_harness):
    """Forward provider-owned routes without calling the checker."""

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
    """Render stopped input checks without dispatching the request."""

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
    """Hide the original provider response when output is blocked."""

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
    """Prevent guarded path variants from reaching transparent forwarding."""

    client, checker, dispatched = proxy_harness

    response = await client.post(path, json={"input": "question"})

    assert response.status_code == 422
    assert dispatched == []
    assert checker.calls == []


@pytest.mark.asyncio
async def test_wrong_method_for_guarded_path_is_not_forwarded(proxy_harness):
    """Return method not allowed instead of forwarding a guarded path."""

    client, checker, dispatched = proxy_harness

    response = await client.put("/v1/generate", json={"input": "question"})

    assert response.status_code == 405
    assert response.headers["allow"] == "POST"
    assert dispatched == []
    assert checker.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method", "path", "expected_status"),
    [
        ("PUT", "/proxy/v1/generate", 405),
        ("POST", "/proxy/v1//generate", 422),
    ],
)
async def test_prefixed_guarded_path_variants_cannot_bypass_into_catchall(
    guarded_operation,
    method,
    path,
    expected_status,
):
    """Keep guarded path ownership when the router has a prefix."""

    checker = StaticChecker()
    dispatched = []

    async def dispatch(request):
        dispatched.append(request)
        return BufferedHttpResponse(200, (), b"provider")

    app = FastAPI()
    app.include_router(
        create_http_proxy_router(
            operations=[guarded_operation],
            checker=checker,
            dispatch=dispatch,
            render_outcome=render_test_outcome,
        ),
        prefix="/proxy",
    )
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy.test") as client:
        response = await client.request(method, path, json={"input": "question"})

    assert response.status_code == expected_status
    assert dispatched == []
    assert checker.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("prefix", ["", "/proxy"])
async def test_reserved_application_route_cannot_fall_through_to_provider(guarded_operation, prefix):
    """Keep application-owned routes out of provider forwarding."""

    dispatched = []

    async def dispatch(request):
        dispatched.append(request)
        return BufferedHttpResponse(200, (), b"provider")

    app = FastAPI()

    @app.get(f"{prefix}/health")
    async def health():
        return {"status": "ok"}

    app.include_router(
        create_http_proxy_router(
            operations=[guarded_operation],
            checker=StaticChecker(),
            dispatch=dispatch,
            render_outcome=render_test_outcome,
            reserved_routes={"/health": {"GET"}},
        ),
        prefix=prefix,
    )
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy.test") as client:
        owned = await client.get(f"{prefix}/health")
        wrong_method = await client.post(f"{prefix}/health")

    assert owned.status_code == 200
    assert owned.json() == {"status": "ok"}
    assert wrong_method.status_code == 405
    assert wrong_method.headers["allow"] == "GET"
    assert dispatched == []


@pytest.mark.asyncio
async def test_checker_policy_is_bound_once_for_all_requests(proxy_harness):
    """Read the statically configured checker settings only once."""

    client, checker, _dispatched = proxy_harness

    await client.post("/v1/generate", json={"input": "one"})
    await client.post("/v1/generate", json={"input": "two"})

    assert checker.policy_reads == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/v1/generate", "/v1/provider-owned"])
async def test_buffered_request_limit_fails_before_dispatch(guarded_operation, path):
    """Reject oversized buffered requests before provider dispatch."""

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
            render_outcome=render_test_outcome,
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
    """Hide oversized provider responses on every forwarded path."""

    async def dispatch(_request):
        return BufferedHttpResponse(200, (), b"12345")

    app = FastAPI()
    app.include_router(
        create_http_proxy_router(
            operations=[guarded_operation],
            checker=StaticChecker(),
            dispatch=dispatch,
            render_outcome=render_test_outcome,
            max_response_body_bytes=4,
        )
    )
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy.test") as client:
        response = await client.post(path, json={"input": "question"})

    assert response.status_code == 502
    assert response.content == b"response_body_too_large"


def test_duplicate_guarded_route_is_rejected(guarded_operation):
    """Reject two guarded operations with the same route shape."""

    duplicate = GuardedHttpOperation(
        operation_path=GuardedOperationPath("/v1/generate"),
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


def test_overlapping_guarded_routes_are_rejected(guarded_operation):
    """Reject routes whose precedence could select the wrong operation."""

    parameterized = GuardedHttpOperation(
        operation_path=GuardedOperationPath("/v1/{name}"),
        operation=BufferedGuardedOperation(
            name="test.parameterized",
            input_projection=project_request,
            output_projection=project_response,
        ),
    )

    with pytest.raises(ValueError, match="must not overlap"):
        create_http_proxy_router(
            operations=[parameterized, guarded_operation],
            checker=StaticChecker(),
            dispatch=lambda _request: None,
            render_outcome=lambda _outcome: None,
        )


def test_guarded_http_path_rejects_path_spanning_parameters():
    """Reject guarded templates that can consume multiple path segments."""

    with pytest.raises(ValueError, match="path-spanning"):
        GuardedOperationPath("/v1/{rest:path}")


@pytest.mark.asyncio
@pytest.mark.parametrize("content_length", ["invalid", "-1", "+2", " 2", "2,2"])
async def test_invalid_content_length_is_rendered_without_dispatch(guarded_operation, content_length):
    """Render malformed content lengths without provider dispatch."""

    dispatched = []
    outcomes = []

    async def dispatch(request):
        dispatched.append(request)
        return BufferedHttpResponse(200, (), b"response")

    def render_outcome(outcome):
        outcomes.append(outcome)
        return render_test_outcome(outcome)

    app = FastAPI()
    app.include_router(
        create_http_proxy_router(
            operations=[guarded_operation],
            checker=StaticChecker(),
            dispatch=dispatch,
            render_outcome=render_outcome,
        )
    )
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy.test") as client:
        response = await client.post(
            "/v1/generate",
            content=b"{}",
            headers={"content-length": content_length},
        )

    assert response.status_code == 400
    assert dispatched == []
    assert len(outcomes) == 1
    assert outcomes[0].kind is HttpFailureKind.INVALID_CONTENT_LENGTH


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/v1/generate", "/v1/provider-owned"])
async def test_repeated_content_length_is_rendered_without_dispatch(guarded_operation, path):
    """Reject repeated content lengths on guarded and pass-through requests."""

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
            render_outcome=render_test_outcome,
        )
    )
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy.test") as client:
        response = await client.post(
            path,
            content=b"{}",
            headers=[("content-length", "2"), ("content-length", "-1")],
        )

    assert response.status_code == 400
    assert dispatched == []


@pytest.mark.asyncio
async def test_observed_request_limit_is_enforced_without_content_length(guarded_operation):
    """Apply the request limit while streaming a body of unknown length."""

    dispatched = []

    async def content():
        yield b"12"
        yield b"345"

    async def dispatch(request):
        dispatched.append(request)
        return BufferedHttpResponse(200, (), b"response")

    app = FastAPI()
    app.include_router(
        create_http_proxy_router(
            operations=[guarded_operation],
            checker=StaticChecker(),
            dispatch=dispatch,
            render_outcome=render_test_outcome,
            max_request_body_bytes=4,
        )
    )
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy.test") as client:
        response = await client.post("/v1/generate", content=content())

    assert response.status_code == 413
    assert dispatched == []


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/v1/generate", "/v1/provider-owned"])
async def test_typed_dispatch_failure_is_rendered(guarded_operation, path):
    """Render dispatch failures explicitly reported by the HTTP client."""

    outcomes = []

    async def dispatch(_request):
        raise HttpDispatchFailed("upstream failed")

    def render_outcome(outcome):
        outcomes.append(outcome)
        return render_test_outcome(outcome)

    app = FastAPI()
    app.include_router(
        create_http_proxy_router(
            operations=[guarded_operation],
            checker=StaticChecker(),
            dispatch=dispatch,
            render_outcome=render_outcome,
        )
    )
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy.test") as client:
        response = await client.post(path, json={"input": "question"})

    assert response.status_code == 502
    assert len(outcomes) == 1
    assert outcomes[0].kind is HttpFailureKind.UPSTREAM_REQUEST_FAILED


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/v1/generate", "/v1/provider-owned"])
async def test_untyped_dispatch_failure_is_not_reclassified(guarded_operation, path):
    """Propagate unexpected dispatch errors without reclassifying them."""

    async def dispatch(_request):
        raise RuntimeError("programming failure")

    app = FastAPI()
    app.include_router(
        create_http_proxy_router(
            operations=[guarded_operation],
            checker=StaticChecker(),
            dispatch=dispatch,
            render_outcome=render_test_outcome,
        )
    )
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy.test") as client:
        with pytest.raises(RuntimeError, match="programming failure"):
            await client.post(path, json={"input": "question"})


def test_request_path_fallback_percent_encodes_unicode():
    """Percent-encode a Unicode path when raw path bytes are unavailable."""

    request = Request(
        {
            "type": "http",
            "http_version": "1.1",
            "method": "GET",
            "scheme": "http",
            "path": "/v1/café",
            "query_string": b"",
            "headers": [],
            "server": ("proxy.test", 80),
            "client": ("127.0.0.1", 1),
        }
    )

    assert _request_path(request) == b"/v1/caf%C3%A9"


async def asgi_exchange(app, path, raw_path, *, method="POST", headers=(), body=b"{}"):
    """Deliver a scope without client-side URL or framing normalization."""

    messages = []

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message):
        messages.append(message)

    await app(
        {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": method,
            "scheme": "http",
            "path": path,
            "raw_path": raw_path,
            "root_path": "",
            "query_string": b"opaque=query",
            "headers": list(headers),
            "server": ("proxy.test", 80),
            "client": ("127.0.0.1", 1),
        },
        receive,
        send,
    )
    return httpx.Response(
        messages[0]["status"],
        headers=messages[0]["headers"],
        content=b"".join(message.get("body", b"") for message in messages[1:]),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("prefix", ["", "/proxy"])
@pytest.mark.parametrize("method", ["POST", "PUT"])
@pytest.mark.parametrize(
    ("path", "raw_path"),
    [
        ("/v1/./generate", b"/v1/./generate"),
        ("/v1/x/../generate", b"/v1/x/../generate"),
        ("/v1/x/../generate", b"/v1/x/%2e%2e/generate"),
        ("/../v1/generate", b"/../v1/generate"),
        ("/v1/generate?", b"/v1/generate%3F"),
        ("/v1/generate#", b"/v1/generate%23"),
    ],
)
async def test_guarded_url_aliases_are_rendered_before_dispatch(guarded_operation, prefix, method, path, raw_path):
    """Reject aliases with one mapped outcome before checking or forwarding."""

    checker = StaticChecker()
    outcomes = []

    async def dispatch(_request):
        pytest.fail("a guarded URL alias must not reach dispatch")

    def render(outcome):
        outcomes.append(outcome)
        return render_test_outcome(outcome)

    app = FastAPI()
    app.include_router(
        create_http_proxy_router(
            operations=[guarded_operation], checker=checker, dispatch=dispatch, render_outcome=render
        ),
        prefix=prefix,
    )
    response = await asgi_exchange(app, prefix + path, prefix.encode() + raw_path, method=method)

    assert response.status_code == (422 if method == "POST" else 405)
    assert len(outcomes) == 1
    assert isinstance(outcomes[0], HttpRouteRejected)
    assert response.content == outcomes[0].kind.value.encode()
    if method == "PUT":
        assert response.headers["allow"] == "POST"
    assert checker.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("path", "raw_path"),
    [
        ("//evil.example/v1/generate", b"//evil.example/v1/generate"),
        ("/v1\\generate", b"/v1%5Cgenerate"),
        ("/v1/provider\x00", b"/v1/provider%00"),
        ("/v1/provider", b"//evil.example/v1/provider"),
        ("/v1/provider", b"/v1/provider?query"),
        ("/v1/generate", b"//evil.example/v1/generate"),
    ],
)
async def test_ambiguous_sender_paths_are_rejected(guarded_operation, path, raw_path):
    """Stop destination-changing path forms independently of sender behavior."""

    async def dispatch(_request):
        pytest.fail("an ambiguous request path must not reach dispatch")

    app = FastAPI()
    app.include_router(
        create_http_proxy_router(
            operations=[guarded_operation],
            checker=StaticChecker(),
            dispatch=dispatch,
            render_outcome=render_test_outcome,
        )
    )
    response = await asgi_exchange(app, path, raw_path)

    assert response.status_code == 400
    assert response.content == b"invalid_request_path"


@pytest.mark.asyncio
@pytest.mark.parametrize("raw_path", [b"/v1/provider%3Fopaque%23fragment", None])
async def test_unrelated_encoded_delimiters_preserve_scope_path(guarded_operation, raw_path):
    """Preserve literal decoded delimiters and their encoding outside guarded aliases."""

    dispatched = []
    checker = StaticChecker()

    async def dispatch(request):
        dispatched.append(request)
        return BufferedHttpResponse(200, (), b"opaque")

    app = FastAPI()
    app.include_router(
        create_http_proxy_router(
            operations=[guarded_operation], checker=checker, dispatch=dispatch, render_outcome=render_test_outcome
        )
    )
    response = await asgi_exchange(app, "/v1/provider?opaque#fragment", raw_path)

    assert response.status_code == 200
    assert dispatched[0].path == "/v1/provider?opaque#fragment"
    assert dispatched[0].raw_path == b"/v1/provider%3Fopaque%23fragment"
    assert dispatched[0].query == b"opaque=query"
    assert checker.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/v1/generate", "/v1/provider-owned"])
@pytest.mark.parametrize(
    "headers",
    [
        ((b"content-length", b"0"),),
        ((b"content-length", b"999"),),
        ((b"content-length", b"2"), (b"transfer-encoding", b"chunked")),
    ],
)
async def test_inconsistent_request_framing_stops_before_dispatch(guarded_operation, path, headers):
    """Reject body length mismatches and conflicting transfer framing on both paths."""

    async def dispatch(_request):
        pytest.fail("an inconsistently framed request must not reach dispatch")

    app = FastAPI()
    app.include_router(
        create_http_proxy_router(
            operations=[guarded_operation],
            checker=StaticChecker(),
            dispatch=dispatch,
            render_outcome=render_test_outcome,
        )
    )
    response = await asgi_exchange(app, path, path.encode(), headers=headers)

    assert response.status_code == 400
    assert response.content == b"invalid_content_length"


def test_response_framing_is_rebuilt_without_changing_end_to_end_values():
    """Remove connection metadata and retain encoded bytes and duplicate safe headers."""

    body = b"opaque encoded bytes"
    response = _render_response(
        BufferedHttpResponse(
            200,
            (
                (b"Content-Length", b"999"),
                (b"content-length", b"1"),
                (b"Transfer-Encoding", b"chunked"),
                (b"Connection", b"x-remove, Keep-Alive"),
                (b"X-Remove", b"connection-specific"),
                (b"keep-alive", b"timeout=5"),
                (b"content-encoding", b"gzip"),
                (b"set-cookie", b"first=1"),
                (b"set-cookie", b"second=2"),
            ),
            body,
        )
    )

    assert response.body == body
    assert response.raw_headers == [
        (b"content-encoding", b"gzip"),
        (b"set-cookie", b"first=1"),
        (b"set-cookie", b"second=2"),
        (b"content-length", str(len(body)).encode()),
    ]


@pytest.mark.parametrize(
    ("method", "status_code", "length", "expected_length"),
    [
        ("HEAD", 200, b"42", "42"),
        ("GET", 304, b"42", "42"),
        ("HEAD", 200, b"invalid", None),
        ("GET", 204, b"42", None),
        ("GET", 205, b"42", "0"),
        ("HEAD", 205, b"42", "0"),
        ("GET", 101, b"42", None),
    ],
)
def test_bodyless_response_framing_preserves_only_valid_representation_lengths(
    method, status_code, length, expected_length
):
    """Suppress forbidden bodies while retaining HEAD and 304 representation metadata."""

    response = _render_response(
        BufferedHttpResponse(status_code, ((b"content-length", length),), b"not a body"), method
    )

    assert response.body == b""
    assert response.headers.get("content-length") == expected_length


@pytest.mark.parametrize(
    "routes",
    [
        {"relative": {"GET"}},
        {"//health": {"GET"}},
        {"/health/../other": {"GET"}},
        {"/health?query": {"GET"}},
        {"/{key:unknown}": {"GET"}},
        {"/health": set()},
        {"/health": {"get"}},
        {"/health": {"BREW"}},
        {"/health": "GET"},
    ],
)
def test_invalid_reserved_declarations_fail_at_construction(guarded_operation, routes):
    """Reject reservations that cannot establish deterministic route ownership."""

    with pytest.raises(ValueError, match="reserved HTTP|Reserved HTTP"):
        create_http_proxy_router(
            operations=[guarded_operation],
            checker=StaticChecker(),
            dispatch=lambda _request: None,
            render_outcome=render_test_outcome,
            reserved_routes=routes,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(("reserved", "path"), [("/", "/.."), ("/health/", "/x/../health/")])
async def test_application_root_and_trailing_slash_reservations_are_supported(guarded_operation, reserved, path):
    """Keep application route forms reserved after ownership normalization."""

    async def dispatch(_request):
        pytest.fail("a reserved application route must not be dispatched")

    app = FastAPI()
    app.include_router(
        create_http_proxy_router(
            operations=[guarded_operation],
            checker=StaticChecker(),
            dispatch=dispatch,
            render_outcome=render_test_outcome,
            reserved_routes={reserved: {"GET"}},
        )
    )
    response = await asgi_exchange(app, path, path.encode())

    assert response.status_code == 405
    assert response.headers["allow"] == "GET"
    assert response.content == b"method_not_allowed"


@pytest.mark.parametrize(
    ("status_code", "headers", "body", "error"),
    [
        (700, (), b"", ValueError),
        (200, (), "text", TypeError),
        (200, (("header", b"value"),), b"", TypeError),
        (200, ((b"header", "value"),), b"", TypeError),
    ],
)
def test_buffered_response_rejects_invalid_wire_values(status_code, headers, body, error):
    """Require status, headers, and body values that the renderer can send."""

    with pytest.raises(error):
        BufferedHttpResponse(status_code, headers, body)


@pytest.mark.parametrize("path", ["relative", "/", "/trailing/", "/{key:unknown}"])
def test_guarded_path_rejects_invalid_templates(path):
    """Reject invalid guarded route ownership declarations."""

    with pytest.raises(ValueError, match="guarded HTTP path"):
        GuardedOperationPath(path)


@pytest.mark.parametrize("methods", [frozenset(), frozenset({"post"}), frozenset({"BREW"})])
def test_guarded_path_rejects_invalid_methods(methods):
    """Require explicit supported methods for guarded route ownership."""

    with pytest.raises(ValueError, match="Guarded HTTP methods"):
        GuardedOperationPath("/v1/generate", methods)


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
@pytest.mark.parametrize("option", ["max_request_body_bytes", "max_response_body_bytes"])
def test_router_rejects_invalid_buffer_limits(guarded_operation, option, limit):
    """Require positive integer byte limits before accepting requests."""

    with pytest.raises(ValueError, match="positive integers"):
        create_http_proxy_router(
            operations=[guarded_operation],
            checker=StaticChecker(),
            dispatch=lambda _request: None,
            render_outcome=render_test_outcome,
            **{option: limit},
        )


def test_router_rejects_empty_or_duplicate_operation_identities(guarded_operation):
    """Reject declarations that cannot identify a unique guarded operation."""

    for operations, message in [([], "At least one"), ([guarded_operation, guarded_operation], "names must be unique")]:
        with pytest.raises(ValueError, match=message):
            create_http_proxy_router(
                operations=operations,
                checker=StaticChecker(),
                dispatch=lambda _request: None,
                render_outcome=render_test_outcome,
            )


def test_disjoint_typed_and_static_guarded_routes_can_coexist(guarded_operation):
    """Allow typed parameters that cannot overlap a static operation path."""

    parameterized = GuardedHttpOperation(
        GuardedOperationPath("/v1/{key:int}"),
        BufferedGuardedOperation("test.integer", project_request, project_response),
    )
    router = create_http_proxy_router(
        operations=[parameterized, guarded_operation],
        checker=StaticChecker(),
        dispatch=lambda _request: None,
        render_outcome=render_test_outcome,
    )

    assert len(router.routes) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize(("length", "expected_status"), [("9" * 5000, 413), ("0" * 5000 + "2", 200)])
async def test_large_content_length_headers_do_not_depend_on_integer_digit_limits(
    guarded_operation, length, expected_status
):
    """Bound large declared lengths before integer conversion and accept leading zeroes."""

    async def dispatch(_request):
        return BufferedHttpResponse(200, (), b"opaque")

    app = FastAPI()
    app.include_router(
        create_http_proxy_router(
            operations=[guarded_operation],
            checker=StaticChecker(),
            dispatch=dispatch,
            render_outcome=render_test_outcome,
        )
    )
    response = await asgi_exchange(app, "/provider", b"/provider", headers=((b"content-length", length.encode()),))

    assert response.status_code == expected_status


@pytest.mark.asyncio
async def test_head_failures_have_no_response_body(guarded_operation):
    """Apply HEAD framing to both route and buffering failures."""

    async def dispatch(_request):
        pytest.fail("a rejected HEAD request must not be dispatched")

    app = FastAPI()
    app.include_router(
        create_http_proxy_router(
            operations=[guarded_operation],
            checker=StaticChecker(),
            dispatch=dispatch,
            render_outcome=render_test_outcome,
        )
    )
    for path, expected_status in [("/v1/generate", 405), ("/provider", 400)]:
        response = await asgi_exchange(app, path, path.encode(), method="HEAD", headers=((b"content-length", b"999"),))

        assert response.status_code == expected_status
        assert response.content == b""
