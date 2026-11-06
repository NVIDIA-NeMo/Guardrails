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
from dataclasses import replace
from unittest.mock import MagicMock

import httpcore
import httpx
import pytest
from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.testclient import TestClient

from nemoguardrails.server.experimental import _application, _runtime
from nemoguardrails.server.experimental._application import create_proxy_application
from nemoguardrails.server.experimental._content_checker import (
    ContentAllowed,
    ContentInspectionPolicy,
)
from nemoguardrails.server.experimental._runtime import get_runtime
from nemoguardrails.server.experimental.providers.openai.integration import create_openai_chat_integration
from nemoguardrails.server.experimental.rails import ConfigEntry
from nemoguardrails.server.experimental.rails._repository import GuardrailsRepository


class StaticStream(httpx.AsyncByteStream):
    def __init__(self, chunks):
        self.chunks = chunks
        self.closed = False

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk

    async def aclose(self):
        self.closed = True


class TrackingTransport(httpx.AsyncBaseTransport):
    def __init__(self, events=None):
        self.events = events
        self.requests = []
        self.request_bodies = []
        self.response_streams = []
        self.close_calls = 0

    async def handle_async_request(self, request):
        if self.events is not None:
            self.events.append("dispatch")
        self.requests.append(request)
        self.request_bodies.append(await request.aread())
        if request.url.path.endswith("/chat/completions"):
            body = b'{"choices":[{"message":{"role":"assistant","content":"safe"},"finish_reason":"stop"}]}'
        else:
            body = b"unbounded provider response"
        stream = StaticStream([body])
        self.response_streams.append(stream)
        return httpx.Response(
            200,
            headers={"content-type": "application/json", "content-length": str(len(body))},
            stream=stream,
            request=request,
        )

    async def aclose(self):
        self.close_calls += 1


class StaticChecker:
    def __init__(self, *, inspect_input=False, inspect_output=False, events=None):
        self.policy = ContentInspectionPolicy(inspect_input, inspect_output)
        self.events = events

    def inspection_policy(self):
        return self.policy

    async def check_input(self, _check):
        if self.events is not None:
            self.events.append("input")
        return ContentAllowed()

    async def check_output(self, _check):
        if self.events is not None:
            self.events.append("output")
        return ContentAllowed()


def chat_request(**updates):
    payload = {
        "model": "provider-model",
        "messages": [{"role": "user", "content": "hello"}],
    }
    payload.update(updates)
    return payload


@pytest.mark.asyncio
async def test_shared_client_does_not_persist_or_replay_caller_cookies():
    observed = []

    async def respond(request):
        observed.append((request.headers["authorization"], request.headers.get("cookie")))
        return httpx.Response(200, headers={"set-cookie": "session=upstream; Path=/"}, stream=StaticStream([b"ok"]))

    application = create_proxy_application(
        providers=(create_openai_chat_integration(base_url="https://provider.test/v1"),),
        http_transport_factory=lambda: httpx.MockTransport(respond),
    )

    async def caller(identity, cookie=None):
        headers = {"authorization": identity}
        if cookie is not None:
            headers["cookie"] = cookie
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=application), base_url="http://proxy.test"
        ) as client:
            response = await client.get("/v1/models", headers=headers)
            assert response.status_code == 200
            assert response.headers["set-cookie"] == "session=upstream; Path=/"

    async with application.router.lifespan_context(application):
        await caller("synthetic-a")
        assert not get_runtime(application).http_client.cookies
        await asyncio.gather(caller("synthetic-b", "session=caller-b"), caller("synthetic-c"))
        assert not get_runtime(application).http_client.cookies
    assert sorted(observed) == [("synthetic-a", None), ("synthetic-b", "session=caller-b"), ("synthetic-c", None)]


def create_test_application(tmp_path, monkeypatch, *, checker, **kwargs):
    config_path = tmp_path / "rails"
    config_path.mkdir(exist_ok=True)
    (config_path / "config.yml").write_text("models: []\n", encoding="utf-8")

    def get_checker(_repository, config_id):
        assert config_id == "test"
        return checker

    monkeypatch.setattr(GuardrailsRepository, "get", get_checker)
    return create_proxy_application(
        providers=(create_openai_chat_integration(),),
        rails_configs=(ConfigEntry("test", config_path),),
        **kwargs,
    )


def test_application_creates_one_shared_transport_per_lifespan_and_closes_it_once(tmp_path, monkeypatch):
    created = []

    def create_transport():
        transport = TrackingTransport()
        created.append(transport)
        return transport

    application = create_test_application(
        tmp_path,
        monkeypatch,
        checker=StaticChecker(),
        http_transport_factory=create_transport,
    )

    for expected_count in (1, 2):
        with TestClient(application) as client:
            runtime = get_runtime(application)
            assert len(created) == expected_count
            assert not runtime.http_client.is_closed
            assert client.get("/v1/models").status_code == 200
            assert client.post("/v1/chat/completions", json=chat_request()).status_code == 200
            assert len(created[-1].requests) == 2
        assert runtime.http_client.is_closed
        assert created[-1].close_calls == 1

    assert created[0] is not created[1]


def test_startup_closes_transport_if_runtime_construction_fails(tmp_path, monkeypatch):
    transport = TrackingTransport()
    application = create_test_application(
        tmp_path,
        monkeypatch,
        checker=StaticChecker(),
        http_transport_factory=lambda: transport,
    )
    monkeypatch.setattr(
        _application.ServerRuntime,
        "create",
        MagicMock(side_effect=RuntimeError("runtime construction failed")),
    )

    with pytest.raises(RuntimeError, match="runtime construction failed"):
        with TestClient(application):
            pass

    assert transport.close_calls == 1
    assert not hasattr(application.state, "proxy_runtime")


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["hang", "raise"])
async def test_startup_rollback_is_bounded_and_preserves_construction_failure(monkeypatch, caplog, failure):
    monkeypatch.setattr(_runtime, "_HTTP_SHUTDOWN_TIMEOUT_SECONDS", 0.01)

    class FailingCloseTransport(TrackingTransport):
        async def aclose(self):
            self.close_calls += 1
            if failure == "hang":
                await asyncio.Event().wait()
            raise RuntimeError("private rollback detail")

    transport = FailingCloseTransport()
    application = create_proxy_application(
        providers=(create_openai_chat_integration(),),
        http_transport_factory=lambda: transport,
    )
    monkeypatch.setattr(_application.ServerRuntime, "create", MagicMock(side_effect=ValueError("runtime failed")))
    with pytest.raises(ValueError, match="runtime failed"):
        async with application.router.lifespan_context(application):
            pytest.fail("startup must not complete")
    assert transport.close_calls == 1
    assert "HTTP client cleanup" in caplog.text
    assert "private rollback detail" not in caplog.text


def test_transport_factory_must_return_an_async_transport(tmp_path, monkeypatch):
    application = create_test_application(
        tmp_path,
        monkeypatch,
        checker=StaticChecker(),
        http_transport_factory=MagicMock(return_value=None),
    )

    with pytest.raises(TypeError, match="must return an httpx.AsyncBaseTransport"):
        with TestClient(application):
            pass


def test_http_transport_closes_when_checker_repository_shutdown_fails(tmp_path, monkeypatch):
    transport = TrackingTransport()
    application = create_test_application(
        tmp_path,
        monkeypatch,
        checker=StaticChecker(),
        http_transport_factory=lambda: transport,
    )

    async def fail_close(_repository):
        raise RuntimeError("checker shutdown failed")

    monkeypatch.setattr(GuardrailsRepository, "close", fail_close)

    with pytest.raises(RuntimeError, match="checker shutdown failed"):
        with TestClient(application):
            pass

    assert transport.close_calls == 1


def test_guarded_dispatch_occurs_between_input_and_output_inspection(tmp_path, monkeypatch):
    events = []
    transport = TrackingTransport(events)
    application = create_test_application(
        tmp_path,
        monkeypatch,
        checker=StaticChecker(inspect_input=True, inspect_output=True, events=events),
        http_transport_factory=lambda: transport,
    )

    with TestClient(application) as client:
        response = client.post("/v1/chat/completions", json=chat_request())

    assert response.status_code == 200
    assert events == ["input", "dispatch", "output"]


def test_only_guarded_output_inspection_forces_identity_encoding(tmp_path, monkeypatch):
    transports = []
    for inspect_output in (False, True):
        transport = TrackingTransport()
        transports.append(transport)
        application = create_test_application(
            tmp_path,
            monkeypatch,
            checker=StaticChecker(inspect_output=inspect_output),
            http_transport_factory=lambda transport=transport: transport,
        )
        with TestClient(application) as client:
            response = client.post(
                "/v1/chat/completions",
                json=chat_request(),
                headers={"accept-encoding": "br"},
            )
        assert response.status_code == 200

    assert transports[0].requests[0].headers["accept-encoding"] == "br"
    assert transports[1].requests[0].headers["accept-encoding"] == "identity"


def test_passthrough_streams_bodies_outside_guarded_limits_and_preserves_representation_headers(tmp_path, monkeypatch):
    monkeypatch.setenv("NEMO_GUARDRAILS_OPENAI_PROXY_UPSTREAM_BASE_URL", "https://api.openai.com/v1")
    transport = TrackingTransport()
    application = create_test_application(
        tmp_path,
        monkeypatch,
        checker=StaticChecker(),
        http_transport_factory=lambda: transport,
        max_request_body_bytes=1,
        max_response_body_bytes=1,
    )
    request_body = b"unbounded provider request"

    with TestClient(application) as client:
        response = client.post(
            "/v1/embeddings?mode=opaque%2Fprovider",
            content=request_body,
            headers={
                "accept-encoding": "br",
                "authorization": "Bearer provider-key",
                "content-type": "application/provider+json",
            },
        )

    assert response.status_code == 200
    assert response.content == b"unbounded provider response"
    assert transport.request_bodies == [request_body]
    assert str(transport.requests[0].url) == "https://api.openai.com/v1/embeddings?mode=opaque%2Fprovider"
    assert transport.requests[0].headers["accept-encoding"] == "br"
    assert transport.requests[0].headers["authorization"] == "Bearer provider-key"
    assert "content-length" not in response.headers
    assert transport.response_streams[0].closed is True


@pytest.mark.parametrize("length", ["invalid", "-1", "3, 3"])
def test_passthrough_renders_invalid_content_length_without_dispatch(tmp_path, monkeypatch, length):
    transport = TrackingTransport()
    application = create_test_application(
        tmp_path,
        monkeypatch,
        checker=StaticChecker(),
        http_transport_factory=lambda: transport,
    )
    with TestClient(application) as client:
        response = client.post("/v1/files", content=b"abc", headers={"content-length": length})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_content_length"
    assert not transport.requests


@pytest.mark.asyncio
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("outer_mount", ["", "/gateway"])
async def test_local_routers_precede_provider_catchalls_and_reserve_paths(tmp_path, monkeypatch, nested, outer_mount):
    dependency_calls = []

    def observe():
        dependency_calls.append("local")

    local = APIRouter(prefix="/ops", dependencies=[Depends(observe)])

    @local.get("/health")
    async def health():
        return {"status": "ok"}

    if nested:
        parent = APIRouter()
        parent.include_router(local, prefix="/internal")
        local = parent
    local_path = ("/internal" if nested else "") + "/ops/health"
    transport = TrackingTransport()
    application = create_test_application(
        tmp_path,
        monkeypatch,
        checker=StaticChecker(),
        local_routers=(local,),
        http_transport_factory=lambda: transport,
    )
    assert local_path in application.openapi()["paths"]
    if outer_mount:
        outer = FastAPI()
        outer.mount(outer_mount, application)
    else:
        outer = application

    async with application.router.lifespan_context(application):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=outer), base_url="http://proxy.test") as client:
            response = await client.get(outer_mount + local_path)
            assert response.status_code == 200
            assert response.json() == {"status": "ok"}
            assert dependency_calls == ["local"]
            rejected = await client.post(outer_mount + local_path)
            assert rejected.status_code == 405
            assert rejected.headers["allow"] == "GET"
            assert not transport.requests
            assert (await client.get(outer_mount + "/v1/models")).status_code == 200
            assert (await client.post(outer_mount + "/v1/chat/completions", json=chat_request())).status_code == 200
    assert len(transport.requests) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("maximum", [1, 2])
async def test_transport_factory_pool_limits_apply_across_origins_until_stream_closes(tmp_path, monkeypatch, maximum):
    application = create_test_application(
        tmp_path,
        monkeypatch,
        checker=StaticChecker(),
        http_transport_factory=lambda: httpx.AsyncHTTPTransport(
            limits=httpx.Limits(max_connections=maximum, max_keepalive_connections=1),
            trust_env=False,
        ),
    )
    async with application.router.lifespan_context(application):
        client = get_runtime(application).http_client
        pool = client._transport._pool
        monkeypatch.setattr(
            pool,
            "_network_backend",
            httpcore.AsyncMockBackend([b"HTTP/1.1 200 OK\r\nContent-Length: 3\r\n\r\nabc"]),
        )
        async with client.stream("GET", "http://first.test/files") as first:
            assert first.status_code == 200
            if maximum == 1:
                with pytest.raises(httpx.PoolTimeout):
                    await client.get("http://second.test/models", timeout=0.01)
            else:
                second = await client.get("http://second.test/models", timeout=0.01)
                assert second.content == b"abc"
        assert (await client.get("http://third.test/models", timeout=0.01)).content == b"abc"
    assert client.is_closed


def test_application_instances_own_independent_runtime_resources(tmp_path, monkeypatch):
    first_transport = TrackingTransport()
    second_transport = TrackingTransport()
    first = create_test_application(
        tmp_path,
        monkeypatch,
        checker=StaticChecker(),
        http_transport_factory=lambda: first_transport,
    )
    second = create_test_application(
        tmp_path,
        monkeypatch,
        checker=StaticChecker(),
        http_transport_factory=lambda: second_transport,
    )

    with TestClient(first) as first_client:
        first_runtime = get_runtime(first)
        with TestClient(second) as second_client:
            second_runtime = get_runtime(second)
            assert first_client.get("/v1/models").status_code == 200
            assert second_client.get("/v1/models").status_code == 200
            assert first_runtime is not second_runtime
            assert first_runtime.http_client is not second_runtime.http_client
            assert first_transport.close_calls == 0
            assert second_transport.close_calls == 0
        assert first_transport.close_calls == 0
        assert second_transport.close_calls == 1
    assert first_transport.close_calls == 1


def test_all_provider_instances_share_the_application_transport(tmp_path, monkeypatch):
    transport = TrackingTransport()
    providers = (
        create_openai_chat_integration(
            instance_id="first",
            base_url="https://first.example/v1",
        ),
        create_openai_chat_integration(
            instance_id="second",
            base_url="https://second.example/v1",
            mount_path="/providers/second",
        ),
    )
    config_path = tmp_path / "rails"
    config_path.mkdir()
    (config_path / "config.yml").write_text("models: []\n", encoding="utf-8")

    def get_checker(_repository, _config_id):
        return StaticChecker()

    monkeypatch.setattr(GuardrailsRepository, "get", get_checker)
    application = create_proxy_application(
        providers=providers,
        rails_configs=(ConfigEntry("test", config_path),),
        http_transport_factory=lambda: transport,
    )

    with TestClient(application) as client:
        assert client.get("/v1/models").status_code == 200
        assert client.get("/providers/second/v1/models").status_code == 200

    assert [request.url.host for request in transport.requests] == [
        "first.example",
        "second.example",
    ]
    assert transport.close_calls == 1


def test_application_can_dispatch_through_an_in_process_asgi_component(tmp_path, monkeypatch):
    forwarding = FastAPI()
    received = []

    @forwarding.get("/v1/models")
    async def models(request: Request):
        received.append(
            (
                request.url.path,
                request.scope["query_string"],
                request.headers["x-routing-context"],
                request.headers.get("x-routing-secret"),
            )
        )
        return {"object": "list", "data": []}

    provider = replace(
        create_openai_chat_integration(base_url="http://forwarding.internal/v1"),
        control_headers=frozenset({"x-routing-secret"}),
    )
    config_path = tmp_path / "rails"
    config_path.mkdir()
    (config_path / "config.yml").write_text("models: []\n", encoding="utf-8")

    def get_checker(_repository, _config_id):
        return StaticChecker()

    monkeypatch.setattr(GuardrailsRepository, "get", get_checker)
    application = create_proxy_application(
        providers=(provider,),
        rails_configs=(ConfigEntry("test", config_path),),
        http_transport_factory=lambda: httpx.ASGITransport(app=forwarding),
    )

    with TestClient(application) as client:
        response = client.get(
            "/v1/models?limit=2",
            headers={
                "x-routing-context": "tenant-a",
                "x-routing-secret": "do-not-forward",
            },
        )

    assert response.status_code == 200
    assert response.json() == {"object": "list", "data": []}
    assert received == [("/v1/models", b"limit=2", "tenant-a", None)]
