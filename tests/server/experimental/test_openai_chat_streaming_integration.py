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
from dataclasses import replace

import httpx
import pytest
from fastapi import FastAPI

from nemoguardrails.server.experimental._content_checker import (
    ContentAllowed,
    ContentBlocked,
    ContentInspectionPolicy,
    StreamBufferingPolicy,
)
from nemoguardrails.server.experimental._http_kernel import BufferedHttpResponse
from nemoguardrails.server.experimental._streaming_http import StreamingHttpResponse
from nemoguardrails.server.experimental.providers.openai.chat_completions.endpoint import (
    CHAT_COMPLETIONS_ENDPOINT,
)
from nemoguardrails.server.experimental.providers.openai.errors import OPENAI_ERROR_MAPPING
from nemoguardrails.server.experimental.providers.openai.integration import create_openai_chat_router


class StaticChecker:
    def __init__(self, policy, input_decision=ContentAllowed(), output_decisions=()):
        self.policy = policy
        self.input_decision = input_decision
        self.output_decisions = list(output_decisions)
        self.calls = []
        self.policy_reads = 0

    def inspection_policy(self):
        self.policy_reads += 1
        return self.policy

    async def check_input(self, check):
        self.calls.append(("input", check))
        return self.input_decision

    async def check_output(self, check):
        self.calls.append(("output", check))
        return self.output_decisions.pop(0) if self.output_decisions else ContentAllowed()


class Source:
    def __init__(self, chunks, failure=None):
        self.chunks = iter(chunks)
        self.failure = failure
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self.chunks)
        except StopIteration:
            if self.failure is not None:
                failure, self.failure = self.failure, None
                raise failure
            raise StopAsyncIteration

    async def aclose(self):
        self.closed = True


def request_body(**updates):
    payload = {
        "model": "gpt-example",
        "messages": [{"role": "user", "content": "question"}],
        "stream": True,
    }
    payload.update(updates)
    return json.dumps(payload, separators=(",", ":")).encode()


def stream_event(content=None, **delta_updates):
    delta = {"content": content} if content is not None else {"role": "assistant"}
    delta.update(delta_updates)
    payload = {
        "id": "chatcmpl-stream",
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
    }
    return b"data: " + json.dumps(payload, separators=(",", ":")).encode() + b"\n\n"


def make_client(checker, stream_dispatch=None, buffered_dispatch=None, max_response_body_bytes=10 * 1024 * 1024):
    async def reject_buffered(_request):
        raise AssertionError("a selected streaming request must not use buffered dispatch")

    app = FastAPI()
    app.include_router(
        create_openai_chat_router(
            checker=checker,
            dispatch=buffered_dispatch or reject_buffered,
            stream_dispatch=stream_dispatch,
            max_response_body_bytes=max_response_body_bytes,
        )
    )
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy.test")


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["missing_dispatch", "event", "media_type", "encoding"])
async def test_stream_errors_honor_endpoint_codes(failure, monkeypatch):
    """Endpoint codes survive failures before and after streaming headers."""
    from nemoguardrails.server.experimental.providers.openai import integration

    endpoint = replace(
        CHAT_COMPLETIONS_ENDPOINT,
        unsupported_request_code="example_request_shape",
        unsupported_response_code="example_response_shape",
    )
    monkeypatch.setattr(integration, "CHAT_COMPLETIONS_ENDPOINT", endpoint)
    source = Source([b'data: {"unexpected": "private-stream-content"}\n\n'])
    headers = [(b"content-type", b"text/plain" if failure == "media_type" else b"text/event-stream")]
    if failure == "encoding":
        headers.append((b"content-encoding", b"gzip"))

    async def dispatch(_request):
        return StreamingHttpResponse(200, tuple(headers), source)

    checker = StaticChecker(ContentInspectionPolicy(True, True, StreamBufferingPolicy(1, 0)))
    async with make_client(checker, None if failure == "missing_dispatch" else dispatch) as client:
        response = await post_stream(client)
    expected = "example_request_shape" if failure == "missing_dispatch" else "example_response_shape"
    assert ('"code":"' + expected + '"').encode() in response.content
    assert b"private-stream-content" not in response.content
    assert response.status_code == (422 if failure == "missing_dispatch" else 200 if failure == "event" else 502)
    if failure != "missing_dispatch":
        assert source.closed is True


async def post_stream(client, body=None, **kwargs):
    headers = {"content-type": "application/json", **kwargs.pop("headers", {})}
    return await client.post("/v1/chat/completions", content=body or request_body(), headers=headers, **kwargs)


@pytest.mark.asyncio
async def test_streaming_chat_uses_one_prepared_request_and_preserves_provider_sse_bytes():
    events = [
        stream_event(),
        stream_event("safe "),
        stream_event("streamed "),
        stream_event("provider "),
        stream_event("response"),
        b"data: [DONE]\n\n",
    ]
    stream = b"".join(events)
    source = Source([stream[:31], stream[31:109], stream[109:]])
    dispatched = []

    async def stream_dispatch(request):
        dispatched.append(request)
        return StreamingHttpResponse(
            200,
            (
                (b"content-type", b"text/event-stream; charset=utf-8"),
                (b"content-length", str(len(stream)).encode()),
                (b"x-request-id", b"provider-stream-id"),
            ),
            source,
        )

    checker = StaticChecker(ContentInspectionPolicy(True, True, StreamBufferingPolicy(2, 0)))
    body = request_body()
    async with make_client(checker, stream_dispatch) as client:
        response = await post_stream(
            client,
            body,
            params={"provider": "opaque"},
            headers={"x-provider-option": "preserve"},
        )

    assert response.status_code == 200
    assert response.content == stream
    assert response.headers["x-request-id"] == "provider-stream-id"
    assert "content-length" not in response.headers
    assert dispatched[0].body == body
    assert dispatched[0].query == b"provider=opaque"
    assert (b"x-provider-option", b"preserve") in dispatched[0].headers
    assert [kind for kind, _ in checker.calls] == ["input", "output", "output"]
    assert [call.output_content for kind, call in checker.calls if kind == "output"] == [
        "safe streamed ",
        "provider response",
    ]
    assert checker.policy_reads == 1
    assert source.closed is True


@pytest.mark.asyncio
async def test_non_streaming_request_uses_the_same_operation_and_buffered_dispatch():
    buffered = []
    streamed = []

    async def buffered_dispatch(request):
        buffered.append(request)
        return BufferedHttpResponse(
            200,
            ((b"content-type", b"application/json"),),
            b'{"choices":[{"index":0,"message":{"role":"assistant","content":"answer"}}]}',
        )

    async def stream_dispatch(request):
        streamed.append(request)
        raise AssertionError("a buffered request must not use streaming dispatch")

    checker = StaticChecker(ContentInspectionPolicy(True, True, StreamBufferingPolicy(1, 0)))
    async with make_client(checker, stream_dispatch, buffered_dispatch) as client:
        response = await post_stream(client, request_body(stream=False))

    assert response.status_code == 200
    assert len(buffered) == 1
    assert streamed == []
    assert [kind for kind, _ in checker.calls] == ["input", "output"]
    assert checker.policy_reads == 1


@pytest.mark.asyncio
async def test_streaming_input_block_and_unsupported_policy_stop_before_dispatch():
    dispatched = []

    async def stream_dispatch(request):
        dispatched.append(request)
        raise AssertionError("a stopped request must not be dispatched")

    blocked = StaticChecker(
        ContentInspectionPolicy(True, True, StreamBufferingPolicy(1, 0)),
        input_decision=ContentBlocked("blocked", "policy"),
    )
    async with make_client(blocked, stream_dispatch) as client:
        blocked_response = await post_stream(client)

    unsupported = StaticChecker(ContentInspectionPolicy(True, True))
    async with make_client(unsupported, stream_dispatch) as client:
        unsupported_response = await post_stream(client)

    assert blocked_response.status_code == 400
    assert blocked_response.json()["error"]["code"] == "content_blocked"
    assert blocked_response.json()["error"]["param"] == "policy"
    assert unsupported_response.status_code == 422
    assert unsupported_response.json()["error"]["code"] == "unsupported_stream_inspection_policy"
    assert dispatched == []


@pytest.mark.asyncio
async def test_missing_stream_dispatch_does_not_fall_through_to_buffered_dispatch():
    buffered = []

    async def buffered_dispatch(request):
        buffered.append(request)
        raise AssertionError("a streaming request must not fall through to buffered dispatch")

    checker = StaticChecker(ContentInspectionPolicy(True, True, StreamBufferingPolicy(1, 0)))
    async with make_client(checker, buffered_dispatch=buffered_dispatch) as client:
        response = await post_stream(client)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "unsupported_chat_completions_shape"
    assert buffered == []


@pytest.mark.asyncio
async def test_unsupported_stream_request_stops_before_stream_dispatch():
    dispatched = []

    async def stream_dispatch(request):
        dispatched.append(request)
        raise AssertionError("an unsupported request must not be dispatched")

    checker = StaticChecker(ContentInspectionPolicy(True, True, StreamBufferingPolicy(1, 0)))
    async with make_client(checker, stream_dispatch) as client:
        response = await post_stream(client, request_body(tools=[{"type": "function"}]))

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "unsupported_chat_completions_shape"
    assert dispatched == []


@pytest.mark.asyncio
async def test_streaming_output_block_hides_the_pending_provider_window():
    unsafe = stream_event("INTERNAL-") + stream_event("SECRET")

    async def stream_dispatch(_request):
        return StreamingHttpResponse(
            200,
            ((b"content-type", b"text/event-stream"),),
            Source([unsafe + b"data: [DONE]\n\n"]),
        )

    checker = StaticChecker(
        ContentInspectionPolicy(True, True, StreamBufferingPolicy(2, 0)),
        output_decisions=[ContentBlocked("blocked")],
    )
    async with make_client(checker, stream_dispatch) as client:
        response = await post_stream(client)

    assert response.status_code == 200
    assert b"INTERNAL-" not in response.content
    assert b"SECRET" not in response.content
    assert b'"code":"content_blocked"' in response.content
    assert response.content.endswith(b"data: [DONE]\n\n")


@pytest.mark.asyncio
async def test_provider_error_event_is_preserved_without_output_check():
    stream = b'data: {"error":{"message":"upstream failed","type":"server_error"}}\n\ndata: [DONE]\n\n'

    async def stream_dispatch(_request):
        return StreamingHttpResponse(200, ((b"content-type", b"text/event-stream"),), Source([stream]))

    checker = StaticChecker(ContentInspectionPolicy(True, True, StreamBufferingPolicy(2, 0)))
    async with make_client(checker, stream_dispatch) as client:
        response = await post_stream(client)

    assert response.content == stream
    assert [kind for kind, _ in checker.calls] == ["input"]


@pytest.mark.asyncio
async def test_streaming_provider_drift_is_hidden_in_native_error_events():
    hidden = b"unguarded reasoning"
    event = stream_event(reasoning_content=hidden.decode())

    async def stream_dispatch(_request):
        return StreamingHttpResponse(200, ((b"content-type", b"text/event-stream"),), Source([event]))

    checker = StaticChecker(ContentInspectionPolicy(True, True, StreamBufferingPolicy(1, 0)))
    async with make_client(checker, stream_dispatch) as client:
        response = await post_stream(client)

    assert response.status_code == 200
    assert hidden not in response.content
    assert b'"code":"unsupported_chat_completions_response_shape"' in response.content
    assert response.content.endswith(b"data: [DONE]\n\n")


@pytest.mark.asyncio
async def test_streaming_provider_http_error_is_preserved_and_bounded():
    provider_body = b'{"error":{"message":"rate limited","type":"provider_error"}}'

    async def stream_dispatch(_request):
        return BufferedHttpResponse(
            429,
            ((b"content-type", b"application/json"), (b"x-request-id", b"provider-error-id")),
            provider_body,
        )

    checker = StaticChecker(ContentInspectionPolicy(True, True, StreamBufferingPolicy(1, 0)))
    async with make_client(checker, stream_dispatch) as client:
        response = await post_stream(client)

    assert response.status_code == 429
    assert response.content == provider_body
    assert response.headers["x-request-id"] == "provider-error-id"
    assert [kind for kind, _ in checker.calls] == ["input"]

    async with make_client(checker, stream_dispatch, max_response_body_bytes=4) as client:
        oversized = await post_stream(client)

    assert oversized.status_code == 502
    assert oversized.json()["error"]["code"] == "upstream_stream_failed"
    assert provider_body not in oversized.content


@pytest.mark.asyncio
async def test_successful_non_sse_response_fails_before_headers_and_closes_source():
    source = Source([b'{"hidden":"secret-provider-payload"}'])

    async def stream_dispatch(_request):
        return StreamingHttpResponse(200, ((b"content-type", b"application/json"),), source)

    checker = StaticChecker(ContentInspectionPolicy(True, True, StreamBufferingPolicy(1, 0)))
    async with make_client(checker, stream_dispatch) as client:
        response = await post_stream(client)

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "unsupported_chat_completions_response_shape"
    assert b"secret-provider-payload" not in response.content
    assert source.closed is True


@pytest.mark.asyncio
async def test_stream_without_output_inspection_is_relayed_without_event_parsing():
    stream = b"opaque provider bytes"
    source = Source([stream[:7], stream[7:]])

    async def stream_dispatch(_request):
        return StreamingHttpResponse(200, ((b"content-type", b"text/event-stream"),), source)

    checker = StaticChecker(ContentInspectionPolicy(True, False))
    async with make_client(checker, stream_dispatch) as client:
        response = await post_stream(client)

    assert response.content == stream
    assert [kind for kind, _ in checker.calls] == ["input"]
    assert source.closed is True


@pytest.mark.asyncio
async def test_upstream_iterator_failure_is_encoded_after_streaming_starts():
    source = Source([], RuntimeError("upstream disconnected"))

    async def stream_dispatch(_request):
        return StreamingHttpResponse(200, ((b"content-type", b"text/event-stream"),), source)

    checker = StaticChecker(ContentInspectionPolicy(True, True, StreamBufferingPolicy(1, 0)))
    async with make_client(checker, stream_dispatch) as client:
        response = await post_stream(client)

    assert response.status_code == 200
    assert b'"code":"upstream_stream_failed"' in response.content
    assert response.content.endswith(b"data: [DONE]\n\n")
    assert source.closed is True


def test_openai_endpoint_creates_fresh_stream_adapters():
    assert CHAT_COMPLETIONS_ENDPOINT.stream is not None

    first = CHAT_COMPLETIONS_ENDPOINT.stream.create_adapter()
    second = CHAT_COMPLETIONS_ENDPOINT.stream.create_adapter()

    assert first is not second
    assert first.hooks is not second.hooks
    assert first.classifier is second.classifier


@pytest.mark.parametrize(
    ("configure_stream_dispatch", "media_types"),
    [(True, {"application/json", "text/event-stream"}), (False, {"application/json"})],
)
def test_openapi_documents_chat_success_modes_that_can_be_dispatched(configure_stream_dispatch, media_types):
    async def dispatch(_request):
        raise AssertionError

    app = FastAPI()
    app.include_router(
        create_openai_chat_router(
            checker=StaticChecker(ContentInspectionPolicy(True, True)),
            dispatch=dispatch,
            stream_dispatch=dispatch if configure_stream_dispatch else None,
        )
    )

    success = app.openapi()["paths"]["/v1/chat/completions"]["post"]["responses"]["200"]

    assert success["description"] == "Provider-native successful response."
    assert set(success["content"]) == media_types


@pytest.mark.asyncio
@pytest.mark.parametrize("prefix", ["", "/proxy"])
@pytest.mark.parametrize(
    "method,path,status,code",
    [
        ("GET", "/v1/chat/completions", 405, "method_not_allowed"),
        ("HEAD", "/v1/chat/completions", 405, "method_not_allowed"),
        ("POST", "/v1/chat/%2e/completions", 422, "non_canonical_path"),
        ("POST", "/v1/chat/completions%3F", 422, "non_canonical_path"),
        ("POST", "/v1/chat/completions%23", 422, "non_canonical_path"),
        ("POST", "/v1%5Cchat/completions", 400, "invalid_request_path"),
    ],
)
async def test_stream_requests_retain_native_route_rejections_before_dispatch(prefix, method, path, status, code):
    async def dispatch(_request):
        pytest.fail("a rejected streaming route must not dispatch")

    checker = StaticChecker(ContentInspectionPolicy(True, True, StreamBufferingPolicy(1, 0)))
    app = FastAPI()
    app.include_router(
        create_openai_chat_router(checker=checker, dispatch=dispatch, stream_dispatch=dispatch), prefix=prefix
    )
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy.test") as client:
        response = await client.request(
            method, prefix + path, content=request_body(), headers={"content-type": "application/json"}
        )

    assert response.status_code == status
    if method == "HEAD":
        assert response.content == b""
    else:
        assert response.json()["error"]["code"] == code
    assert response.headers.get("allow") == ("POST" if status == 405 else None)
    assert checker.calls == []
    assert status in {documented.status_code for documented in OPENAI_ERROR_MAPPING.responses}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "headers", [{"content-length": "999"}, {"content-length": "3", "transfer-encoding": "chunked"}]
)
async def test_stream_requests_use_shared_request_framing_validation(headers):
    async def dispatch(_request):
        pytest.fail("invalid request framing must stop before dispatch")

    checker = StaticChecker(ContentInspectionPolicy(True, True, StreamBufferingPolicy(1, 0)))
    async with make_client(checker, stream_dispatch=dispatch) as client:
        response = await post_stream(client, headers=headers)

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_content_length"
    assert checker.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [204, 205])
async def test_bodyless_stream_success_fails_natively_before_headers(status):
    source = Source([b"hidden provider content"])

    async def dispatch(_request):
        return StreamingHttpResponse(
            status, ((b"content-type", b"text/event-stream"), (b"x-provider-secret", b"hidden")), source
        )

    checker = StaticChecker(ContentInspectionPolicy(True, True, StreamBufferingPolicy(1, 0)))
    async with make_client(checker, stream_dispatch=dispatch) as client:
        response = await post_stream(client)

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "unsupported_chat_completions_response_shape"
    assert "x-provider-secret" not in response.headers
    assert source.closed is True
