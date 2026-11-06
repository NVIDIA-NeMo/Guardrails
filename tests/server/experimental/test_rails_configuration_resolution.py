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

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import anyio
import httpx
import pytest
from fastapi import Request
from fastapi.testclient import TestClient

from nemoguardrails import RailsConfig
from nemoguardrails.server.experimental._application import create_proxy_application
from nemoguardrails.server.experimental._content_checker import (
    ContentAllowed,
    ContentInspectionPolicy,
    StreamBufferingPolicy,
    UnsupportedContentCheckerConfiguration,
)
from nemoguardrails.server.experimental._runtime import ServerRuntime
from nemoguardrails.server.experimental.providers.openai.integration import create_openai_chat_integration
from nemoguardrails.server.experimental.rails import ConfigEntry
from nemoguardrails.server.experimental.rails._adapter import LegacyGuardrailsChecker
from nemoguardrails.server.experimental.rails._repository import GuardrailsRepository


class StaticStream(httpx.AsyncByteStream):
    def __init__(self, body):
        self.body = body

    async def __aiter__(self):
        yield self.body


class RecordingTransport(httpx.AsyncBaseTransport):
    def __init__(self):
        self.requests = []

    async def handle_async_request(self, request):
        self.requests.append(request)
        request_body = await request.aread()
        payload = json.loads(request_body) if request_body else {}
        if payload.get("stream"):
            body = (
                b'data: {"id":"chatcmpl-stream","object":"chat.completion.chunk","choices":'
                b'[{"index":0,"delta":{"content":"safe"},"finish_reason":null}]}\n\ndata: [DONE]\n\n'
            )
            content_type = "text/event-stream"
        else:
            body = b'{"choices":[{"message":{"role":"assistant","content":"safe"},"finish_reason":"stop"}]}'
            content_type = "application/json"
        return httpx.Response(
            200,
            headers={"content-type": content_type},
            stream=StaticStream(body),
            request=request,
        )


def config_entry(tmp_path, config_id="policy"):
    path = tmp_path / config_id
    path.mkdir()
    (path / "config.yml").write_text("models: []\n", encoding="utf-8")
    return ConfigEntry(config_id, path)


def checker(*, inspect_output=False, streaming=False):
    value = MagicMock()
    stream_buffering = StreamBufferingPolicy(1, 0) if streaming and inspect_output else None
    value.inspection_policy.return_value = ContentInspectionPolicy(False, inspect_output, stream_buffering)
    value.check_input = AsyncMock(return_value=ContentAllowed())
    value.check_output = AsyncMock(return_value=ContentAllowed())
    return value


def chat_request():
    return {
        "model": "provider-model",
        "messages": [{"role": "user", "content": "hello"}],
    }


def test_trusted_resolver_receives_request_and_provider_context(tmp_path, monkeypatch):
    contexts = []
    resolved_checker = checker()
    get_checker = MagicMock(return_value=resolved_checker)
    monkeypatch.setattr(GuardrailsRepository, "get", get_checker)

    async def resolve_config(context):
        contexts.append(context)
        return "tenant-policy"

    transport = RecordingTransport()
    application = create_proxy_application(
        providers=(create_openai_chat_integration(instance_id="production"),),
        rails_configs=(config_entry(tmp_path, "tenant-policy"),),
        config_resolver=resolve_config,
        http_transport_factory=lambda: transport,
    )

    @application.middleware("http")
    async def authenticate(request, call_next):
        request.state.tenant = "tenant-a"
        return await call_next(request)

    with TestClient(application) as client:
        response = client.post(
            "/v1/chat/completions",
            json=chat_request(),
            headers={"x-nemo-guardrails-config": "ignored-by-guardrails"},
        )

    assert response.status_code == 200
    assert len(contexts) == 1
    assert contexts[0].provider_type == "openai"
    assert contexts[0].provider_instance_id == "production"
    assert contexts[0].operation_name == "chat_completions.create"
    assert contexts[0].request.state.tenant == "tenant-a"
    get_checker.assert_called_once_with("tenant-policy")
    assert transport.requests[0].headers["x-nemo-guardrails-config"] == "ignored-by-guardrails"


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("body_reader", ["json", "body", "stream"])
def test_resolver_reads_buffered_body_and_preserves_forwarded_bytes(tmp_path, monkeypatch, stream, body_reader):
    entries = tuple(config_entry(tmp_path, name) for name in ("first", "second"))
    checkers = {entry.id: checker(inspect_output=True, streaming=stream) for entry in entries}
    for value in checkers.values():
        policy = value.inspection_policy.return_value
        value.inspection_policy.return_value = ContentInspectionPolicy(True, True, policy.stream_buffering)
    get_checker = MagicMock(side_effect=checkers.__getitem__)
    monkeypatch.setattr(GuardrailsRepository, "get", get_checker)
    resolved_bodies = []

    async def resolve_config(context):
        request = context.request
        if body_reader == "json":
            payload = await request.json()
        elif body_reader == "body":
            payload = json.loads(await request.body())
        else:
            payload = json.loads(b"".join([chunk async for chunk in request.stream()]))
        resolved_bodies.append(await request.body())
        assert await request.json() == payload
        return {"first-model": "first", "second-model": "second"}[payload["model"]]

    transport = RecordingTransport()
    application = create_proxy_application(
        providers=(create_openai_chat_integration(),),
        rails_configs=entries,
        config_resolver=resolve_config,
        http_transport_factory=lambda: transport,
    )
    bodies = []
    with TestClient(application) as client:
        for name in checkers:
            body = json.dumps({**chat_request(), "model": f"{name}-model", "stream": stream}, indent=2).encode()
            bodies.append(body)
            response = client.post("/v1/chat/completions", content=body, headers={"content-type": "application/json"})
            assert response.status_code == 200
            if stream:
                assert response.content.endswith(b"data: [DONE]\n\n")

    assert resolved_bodies == bodies
    assert [request.content for request in transport.requests] == bodies
    assert [call.args[0] for call in get_checker.call_args_list] == ["first", "second"]
    for value in checkers.values():
        value.check_input.assert_awaited_once()
        value.check_output.assert_awaited()


@pytest.mark.parametrize("chunked", [False, True])
def test_oversized_request_is_rejected_before_body_based_resolution(tmp_path, chunked):
    resolver = AsyncMock(return_value="policy")
    transport = RecordingTransport()
    body = json.dumps(chat_request()).encode()
    application = create_proxy_application(
        providers=(create_openai_chat_integration(),),
        rails_configs=(config_entry(tmp_path),),
        config_resolver=resolver,
        http_transport_factory=lambda: transport,
        max_request_body_bytes=len(body) - 1,
    )
    with TestClient(application) as client:
        request = client.build_request(
            "POST",
            "/v1/chat/completions",
            content=iter((body[:10], body[10:])) if chunked else body,
            headers={"content-type": "application/json"},
        )
        assert ("content-length" not in request.headers) == chunked
        response = client.send(request)

    assert response.status_code == 413
    resolver.assert_not_called()
    assert transport.requests == []


def test_unknown_resolved_config_fails_closed_before_upstream(tmp_path, caplog):
    transport = RecordingTransport()
    application = create_proxy_application(
        providers=(create_openai_chat_integration(),),
        rails_configs=(config_entry(tmp_path, "known"),),
        config_resolver=lambda _context: "unknown",
        http_transport_factory=lambda: transport,
    )

    with TestClient(application) as client:
        response = client.post("/v1/chat/completions", json=chat_request())

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "config_resolution_failed"
    assert transport.requests == []
    assert "Rails config resolution failed for openai.chat_completions.create (openai): ValueError" in caplog.text
    assert "Unknown Rails config" not in caplog.text


@pytest.mark.parametrize("stream", [False, True], ids=["buffered", "streaming"])
@pytest.mark.parametrize("sync_factory", [False, True], ids=["async", "sync-returning-awaitable"])
def test_resolver_deadline_cancels_selection_before_upstream(tmp_path, caplog, stream, sync_factory):
    cancelled = []

    async def resolve(_context):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.append(True)

    transport = RecordingTransport()
    application = create_proxy_application(
        providers=(create_openai_chat_integration(),),
        rails_configs=(config_entry(tmp_path),),
        config_resolver=(lambda context: resolve(context)) if sync_factory else resolve,
        config_resolution_timeout=0.01,
        http_transport_factory=lambda: transport,
    )

    with TestClient(application) as client:
        response = client.post("/v1/chat/completions", json={**chat_request(), "stream": stream})

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "config_resolution_failed"
    assert cancelled == [True]
    assert transport.requests == []
    assert "TimeoutError" in caplog.text


@pytest.mark.asyncio
async def test_cancelling_request_cancels_resolver_without_rewriting_failure(tmp_path):
    entered = asyncio.Event()
    cancelled = asyncio.Event()

    async def resolve(_context):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    runtime = ServerRuntime.create((config_entry(tmp_path),), resolve)
    task = asyncio.create_task(runtime.resolve_content_checker(Request({"type": "http"}), "openai", "test", "chat"))
    try:
        with anyio.fail_after(1):
            await entered.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert cancelled.is_set()
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await runtime.close()


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan"), True, None, "5"])
def test_invalid_config_resolution_deadline_rejected_during_assembly(timeout):
    with pytest.raises(ValueError, match="config resolution timeout"):
        create_proxy_application(providers=(create_openai_chat_integration(),), config_resolution_timeout=timeout)


@pytest.mark.parametrize("stream", [False, True], ids=["buffered", "streaming"])
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("invalid_result", [False, True], ids=["exception", "invalid-result"])
def test_resolver_failure_logs_identity_without_sensitive_details(
    tmp_path, caplog, stream, asynchronous, invalid_result
):
    marker = "resolver-sensitive-marker"

    def resolve(_context):
        if invalid_result:
            return {"private": marker}
        raise RuntimeError(marker)

    async def resolve_async(context):
        return resolve(context)

    transport = RecordingTransport()
    application = create_proxy_application(
        providers=(create_openai_chat_integration(instance_id="production"),),
        rails_configs=(config_entry(tmp_path),),
        config_resolver=resolve_async if asynchronous else resolve,
        http_transport_factory=lambda: transport,
    )

    with TestClient(application) as client:
        response = client.post(
            "/v1/chat/completions",
            json={**chat_request(), "stream": stream},
            headers={"authorization": f"Bearer {marker}"},
        )

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "config_resolution_failed"
    assert transport.requests == []
    records = [record for record in caplog.records if record.name == "nemoguardrails.server.experimental._http_kernel"]
    assert len(records) == 1
    record = records[0]
    error_type = "TypeError" if invalid_result else "RuntimeError"
    assert record.levelname == "WARNING"
    assert record.getMessage() == (
        f"Rails config resolution failed for production.chat_completions.create (openai): {error_type}"
    )
    assert record.exc_info is None
    assert marker not in caplog.text
    assert marker not in response.text


@pytest.mark.parametrize(
    "rail_type,flow",
    [
        ("tool_call", "tool call validation"),
        ("tool_result", "tool result validation"),
        ("tool_output", "tool call validation"),
        ("tool_input", "tool result validation"),
    ],
)
@pytest.mark.parametrize("per_tool", [False, True], ids=["global", "per-tool"])
@pytest.mark.parametrize("stream", [False, True], ids=["buffered", "streaming"])
def test_tool_rails_fail_before_upstream(tmp_path, monkeypatch, rail_type, flow, per_tool, stream):
    section = {"per_tool": {"unused": [], "lookup": [flow]}} if per_tool else {"flows": [flow]}
    config = RailsConfig.from_content(config={"models": [], "rails": {rail_type: section}})
    resolved_checker = LegacyGuardrailsChecker(MagicMock(config=config))
    monkeypatch.setattr(GuardrailsRepository, "get", MagicMock(return_value=resolved_checker))
    transport = RecordingTransport()
    application = create_proxy_application(
        providers=(create_openai_chat_integration(),),
        rails_configs=(config_entry(tmp_path),),
        http_transport_factory=lambda: transport,
    )

    with TestClient(application) as client:
        response = client.post("/v1/chat/completions", json={**chat_request(), "stream": stream})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "unsupported_rail_types"
    assert transport.requests == []


def test_unsupported_resolved_checker_configuration_fails_before_upstream(tmp_path, monkeypatch):
    resolved_checker = checker()
    resolved_checker.inspection_policy.side_effect = UnsupportedContentCheckerConfiguration("retrieval rails")
    monkeypatch.setattr(GuardrailsRepository, "get", MagicMock(return_value=resolved_checker))
    transport = RecordingTransport()
    application = create_proxy_application(
        providers=(create_openai_chat_integration(),),
        rails_configs=(config_entry(tmp_path),),
        http_transport_factory=lambda: transport,
    )

    with TestClient(application) as client:
        response = client.post("/v1/chat/completions", json=chat_request())

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "unsupported_rail_types"
    assert transport.requests == []


@pytest.mark.parametrize("rail_type", ["tool_call", "tool_result", "tool_input", "tool_output"])
def test_empty_per_tool_flows_do_not_enable_unsupported_rails(rail_type):
    """Treat a tool entry without flows as unconfigured, as the core does."""

    config = RailsConfig.from_content(config={"models": [], "rails": {rail_type: {"per_tool": {"lookup": []}}}})
    resolved_checker = LegacyGuardrailsChecker(MagicMock(config=config))

    assert resolved_checker.inspection_policy() == ContentInspectionPolicy(False, False)


def test_resolved_checker_policy_controls_each_request_representation(tmp_path, monkeypatch):
    entries = (
        config_entry(tmp_path, "inspect"),
        config_entry(tmp_path, "passthrough"),
    )
    checkers = {
        "inspect": checker(inspect_output=True),
        "passthrough": checker(inspect_output=False),
    }

    def get_checker(_repository, config_id):
        return checkers[config_id]

    monkeypatch.setattr(GuardrailsRepository, "get", get_checker)
    transport = RecordingTransport()
    application = create_proxy_application(
        providers=(create_openai_chat_integration(),),
        rails_configs=entries,
        config_resolver=lambda context: context.request.headers["x-policy"],
        http_transport_factory=lambda: transport,
    )

    with TestClient(application) as client:
        for policy in ("passthrough", "inspect"):
            response = client.post(
                "/v1/chat/completions",
                json=chat_request(),
                headers={"x-policy": policy, "accept-encoding": "br"},
            )
            assert response.status_code == 200

    assert transport.requests[0].headers["accept-encoding"] == "br"
    assert transport.requests[1].headers["accept-encoding"] == "identity"


def test_resolved_checker_policy_controls_each_stream_request_representation(tmp_path, monkeypatch):
    entries = (
        config_entry(tmp_path, "inspect-stream"),
        config_entry(tmp_path, "passthrough-stream"),
    )
    checkers = {
        "inspect-stream": checker(inspect_output=True, streaming=True),
        "passthrough-stream": checker(inspect_output=False, streaming=True),
    }

    def get_checker(_repository, config_id):
        return checkers[config_id]

    monkeypatch.setattr(GuardrailsRepository, "get", get_checker)
    transport = RecordingTransport()
    application = create_proxy_application(
        providers=(create_openai_chat_integration(),),
        rails_configs=entries,
        config_resolver=lambda context: context.request.headers["x-policy"],
        http_transport_factory=lambda: transport,
    )

    with TestClient(application) as client:
        for policy in ("passthrough-stream", "inspect-stream"):
            payload = chat_request()
            payload["stream"] = True
            response = client.post(
                "/v1/chat/completions",
                json=payload,
                headers={"x-policy": policy, "accept-encoding": "br"},
            )
            assert response.status_code == 200
            assert response.content.endswith(b"data: [DONE]\n\n")

    assert transport.requests[0].headers["accept-encoding"] == "br"
    assert transport.requests[1].headers["accept-encoding"] == "identity"
