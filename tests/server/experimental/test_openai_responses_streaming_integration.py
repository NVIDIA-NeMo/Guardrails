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

"""Exercise Responses through the existing HTTP and streaming kernels."""

import json

import httpx
import pytest
from fastapi import FastAPI

from nemoguardrails.server.experimental._content_checker import (
    ContentAllowed,
    ContentBlocked,
    ContentInspectionPolicy,
    StreamBufferingPolicy,
)
from nemoguardrails.server.experimental._guarded_proxy import create_guarded_http_operation
from nemoguardrails.server.experimental._http_kernel import create_http_proxy_router
from nemoguardrails.server.experimental._provider_routes import include_provider_routes
from nemoguardrails.server.experimental._streaming_http import StreamingHttpResponse
from nemoguardrails.server.experimental.providers.openai.errors import OPENAI_ERROR_MAPPING
from nemoguardrails.server.experimental.providers.openai.integration import create_openai_integration
from nemoguardrails.server.experimental.providers.openai.responses.endpoint import RESPONSES_ENDPOINT
from tests.server.experimental.test_openai_chat_streaming_integration import Source, StaticChecker
from tests.server.experimental.test_openai_responses_integration import ResponseBody
from tests.server.experimental.test_openai_responses_projections import response_payload
from tests.server.experimental.test_openai_responses_streaming_projections import event_bytes


async def post_stream(chunks, checker, *, failure=None):
    source = Source(chunks, failure=failure)
    dispatches = []

    async def dispatch(request):
        dispatches.append(request)
        return StreamingHttpResponse(200, ((b"content-type", b"text/event-stream"),), source)

    async def reject_buffered(request):
        raise AssertionError("Streaming request must not use buffered dispatch")

    application = FastAPI()
    application.include_router(
        create_http_proxy_router(
            operations=(
                create_guarded_http_operation(
                    RESPONSES_ENDPOINT,
                    OPENAI_ERROR_MAPPING,
                    stream_dispatch=dispatch,
                    max_stream_event_bytes=1024 * 1024,
                    max_pending_stream_bytes=1024 * 1024,
                ),
            ),
            checker=checker,
            dispatch=reject_buffered,
            render_outcome=OPENAI_ERROR_MAPPING.renderer,
        )
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=application), base_url="http://proxy.test"
    ) as client:
        result = await client.post("/v1/responses", json={"model": "test-model", "input": "question", "stream": True})
    return result, source, dispatches


def checker(*, input_decision=ContentAllowed(), output_decisions=()):
    return StaticChecker(
        ContentInspectionPolicy(True, True, StreamBufferingPolicy(8, 0)), input_decision, output_decisions
    )


def good_stream():
    return b"".join(
        [
            event_bytes("response.created", response={"output": []}),
            event_bytes("response.output_text.delta", delta="an", item_id="msg_test", output_index=1),
            event_bytes("response.output_text.delta", delta="swer", item_id="msg_test", output_index=1),
            event_bytes("response.output_text.done", text="answer"),
            event_bytes("response.completed", response=response_payload()),
        ]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("fragment_size", [1, 17, 10000])
async def test_allowed_stream_preserves_exact_bytes_and_checks_original_deltas(fragment_size):
    wire = good_stream()
    inspector = checker()
    response, source, dispatches = await post_stream(
        [wire[index : index + fragment_size] for index in range(0, len(wire), fragment_size)], inspector
    )
    assert response.status_code == 200
    assert response.content == wire
    assert len(dispatches) == 1 and source.closed
    assert [kind for kind, _ in inspector.calls] == ["input", "output"]
    assert inspector.calls[0][1].message.content == "question"
    assert inspector.calls[1][1].output_content == "answer"


@pytest.mark.asyncio
async def test_input_block_prevents_any_upstream_dispatch():
    response, _, dispatches = await post_stream([good_stream()], checker(input_decision=ContentBlocked("blocked")))
    assert response.status_code == 400
    assert dispatches == []


@pytest.mark.asyncio
async def test_output_block_withholds_deltas_and_all_text_snapshots():
    response, source, _ = await post_stream([good_stream()], checker(output_decisions=[ContentBlocked("blocked")]))
    assert response.status_code == 200
    assert b'"code":"content_blocked"' in response.content
    assert b"event: error" in response.content
    assert b"response.output_text" not in response.content
    assert b"answer" not in response.content
    assert b"[DONE]" not in response.content
    assert source.closed


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tail",
    [
        b"",
        b"data: [DONE]\n\n",
        event_bytes("response.output_text.done", text="different"),
        event_bytes("response.refusal.delta", delta="hidden"),
        event_bytes("response.completed", response=response_payload("different")),
        event_bytes("response.completed", response=response_payload())
        + event_bytes("response.output_text.delta", delta="late"),
    ],
)
async def test_invalid_completion_never_releases_pending_text(tail):
    wire = event_bytes("response.output_text.delta", delta="answer") + tail
    response, source, _ = await post_stream([wire], checker())
    assert b'"code":"unsupported_responses_response_shape"' in response.content
    assert b"answer" not in response.content
    assert b"hidden" not in response.content
    assert source.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal", ["response.failed", "response.incomplete", "error"])
async def test_provider_terminal_failures_are_relayed_after_checked_text(terminal):
    fields = (
        {"message": "provider failure", "code": "provider_error"}
        if terminal == "error"
        else {"response": response_payload()}
    )
    wire = event_bytes("response.output_text.delta", delta="answer") + event_bytes(terminal, **fields)
    response, source, _ = await post_stream([wire], checker())
    assert response.content == wire
    assert source.closed


@pytest.mark.asyncio
async def test_upstream_failure_closes_owner_and_withholds_pending_text():
    response, source, _ = await post_stream(
        [event_bytes("response.output_text.delta", delta="answer")],
        checker(),
        failure=httpx.ReadError("private detail"),
    )
    assert b"answer" not in response.content
    assert b"private detail" not in response.content
    assert b"upstream_stream_failed" in response.content
    assert source.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("storage_fields", [{}, {"store": True}], ids=["default-storage", "explicit-storage"])
async def test_stored_response_retrieval_passes_through_after_later_stream_output_is_blocked(storage_fields):
    response_id = "resp_stored"
    accepted_prefix = event_bytes("response.created", response={"id": response_id, "output": []}) + event_bytes(
        "response.output_text.delta", delta="safe prefix "
    )
    stored_response = {**response_payload("safe prefix BLOCK_OUTPUT"), "id": response_id, "store": True}
    stored_wire = json.dumps(stored_response, indent=2).encode()
    retrieval_stream = ResponseBody(stored_wire)
    stream = ResponseBody(
        accepted_prefix
        + event_bytes("response.output_text.delta", delta="BLOCK_OUTPUT")
        + event_bytes("response.completed", response=stored_response)
    )
    inspector = StaticChecker(
        ContentInspectionPolicy(True, True, StreamBufferingPolicy(1, 0)),
        output_decisions=[ContentAllowed(), ContentBlocked("blocked")],
    )
    requests = []
    resolutions = []

    async def handler(request):
        requests.append(request)
        if request.method == "POST":
            assert request.url.path == "/v1/responses"
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)
        assert request.method == "GET"
        assert request.url.path == f"/v1/responses/{response_id}"
        return httpx.Response(200, headers={"content-type": "application/json"}, stream=retrieval_stream)

    async def resolve_checker(request, provider_type, provider_instance_id, operation_name):
        resolutions.append((request.method, operation_name))
        return inspector

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as upstream:
        application = FastAPI()
        include_provider_routes(
            application,
            (create_openai_integration(base_url="https://provider.test/v1", operations=("responses",)),),
            resolve_checker=resolve_checker,
            http_client=lambda: upstream,
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=application), base_url="http://proxy.test"
        ) as client:
            request_wire = json.dumps({"input": "question", "stream": True, **storage_fields}, indent=2).encode()
            response = await client.post(
                "/v1/responses", content=request_wire, headers={"content-type": "application/json"}
            )
            assert response.status_code == 200
            assert response.content.startswith(accepted_prefix)
            assert response_id.encode() in response.content
            assert b'"code":"content_blocked"' in response.content
            assert b"BLOCK_OUTPUT" not in response.content
            assert b"response.completed" not in response.content
            assert stream.closed
            assert requests[0].content == request_wire
            assert inspector.text_calls == [
                ("input", "question"),
                ("output", "safe prefix "),
                ("output", "BLOCK_OUTPUT"),
            ]
            calls_before_retrieval = list(inspector.calls)
            policy_reads_before_retrieval = inspector.policy_reads

            retrieved = await client.get(f"/v1/responses/{response_id}")

    assert retrieved.status_code == 200
    assert retrieved.content == stored_wire
    assert retrieval_stream.closed
    assert b"BLOCK_OUTPUT" in retrieved.content
    assert [request.method for request in requests] == ["POST", "GET"]
    assert resolutions == [("POST", "responses.create")]
    assert inspector.calls == calls_before_retrieval
    assert inspector.policy_reads == policy_reads_before_retrieval
