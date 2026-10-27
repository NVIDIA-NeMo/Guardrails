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

from nemoguardrails.server.experimental._application import create_proxy_application
from nemoguardrails.server.experimental.examples.model_routing_transport import ModelRoute, ModelRoutingTransport
from nemoguardrails.server.experimental.providers.openai.integration import create_openai_chat_integration
from nemoguardrails.server.experimental.rails import ConfigEntry
from tests.server.experimental.chat_fixtures import stream_event


class RecordingTransport(httpx.AsyncBaseTransport):
    def __init__(self):
        self.requests = []
        self.close_calls = 0

    async def handle_async_request(self, request):
        self.requests.append(request)
        return httpx.Response(200, json={"ok": True}, request=request)

    async def aclose(self):
        self.close_calls += 1


def request(model_marker=...):
    payload = {"messages": [{"role": "user", "content": "hello"}]}
    if model_marker is not ...:
        payload["model"] = model_marker
    return httpx.Request(
        "POST",
        "http://model-router.internal/v1/files/a%2Fb?trace=enabled%2Fverbose",
        headers={
            "authorization": "Bearer caller-credential",
            "content-type": "application/json",
            "content-digest": "caller-signature",
        },
        json=payload,
    )


def routes():
    return {
        "canonical-a": ModelRoute(
            httpx.URL("https://provider-a.example/v1"),
            "deployment-a",
            "PROVIDER_A_KEY",
        ),
        "canonical-b": ModelRoute(
            httpx.URL("https://provider-b.example/openai/v1"),
            "vendor/deployment-b",
            "PROVIDER_B_KEY",
        ),
    }


@pytest.fixture
def rails_configs(tmp_path):
    (tmp_path / "config.yml").write_text(
        "models: []\nrails:\n  config:\n    regex_detection:\n"
        "      input:\n        patterns: [BLOCK_INPUT]\n"
        "      output:\n        patterns: [BLOCK_OUTPUT]\n"
        "  input:\n    flows: [regex check input]\n"
        "  output:\n    flows: [regex check output]\n"
        "    streaming:\n      enabled: true\n      chunk_size: 2\n      context_size: 2\n      stream_first: false\n",
        encoding="utf-8",
    )
    return (ConfigEntry("test", tmp_path),)


@pytest.mark.asyncio
async def test_alias_selects_trusted_origin_and_rewrites_model_and_credential(monkeypatch):
    monkeypatch.setenv("PROVIDER_A_KEY", "destination-a")
    network = RecordingTransport()
    transport = ModelRoutingTransport(
        routes(),
        {"public-a": "canonical-a"},
        default_model="canonical-a",
        network_transport=network,
    )
    routed = request("public-a")
    for name in ("api-key", "x-api-key", "cookie"):
        routed.headers[name] = "caller-secret"
    payload = json.loads(routed.content)
    payload["base_url"] = "http://client-selected.invalid"
    routed = httpx.Request(routed.method, routed.url, headers=routed.headers, json=payload)

    response = await transport.handle_async_request(routed)

    assert response.status_code == 200
    assert len(network.requests) == 1
    upstream = network.requests[0]
    assert str(upstream.url) == "https://provider-a.example/v1/files/a%2Fb?trace=enabled%2Fverbose"
    assert json.loads(upstream.content) == {
        "messages": [{"role": "user", "content": "hello"}],
        "model": "deployment-a",
        "base_url": "http://client-selected.invalid",
    }
    assert upstream.headers["authorization"] == "Bearer destination-a"
    assert all(name not in upstream.headers for name in ("api-key", "x-api-key", "cookie"))
    assert "content-digest" not in upstream.headers
    assert upstream.headers["content-length"] == str(len(upstream.content))
    assert "transfer-encoding" not in upstream.headers


@pytest.mark.asyncio
async def test_each_destination_receives_only_its_own_credential(monkeypatch):
    monkeypatch.setenv("PROVIDER_A_KEY", "destination-a")
    monkeypatch.setenv("PROVIDER_B_KEY", "destination-b")
    network = RecordingTransport()
    transport = ModelRoutingTransport(
        routes(),
        {"public-a": "canonical-a", "public-b": "canonical-b"},
        default_model="canonical-a",
        network_transport=network,
    )

    await transport.handle_async_request(request("public-a"))
    await transport.handle_async_request(request("public-b"))

    assert [item.url.host for item in network.requests] == ["provider-a.example", "provider-b.example"]
    assert [str(item.url) for item in network.requests] == [
        "https://provider-a.example/v1/files/a%2Fb?trace=enabled%2Fverbose",
        "https://provider-b.example/openai/v1/files/a%2Fb?trace=enabled%2Fverbose",
    ]
    assert [item.headers["authorization"] for item in network.requests] == [
        "Bearer destination-a",
        "Bearer destination-b",
    ]
    assert all(item.headers["authorization"] != "Bearer caller-credential" for item in network.requests)


@pytest.mark.asyncio
async def test_model_less_request_uses_explicit_default_route(monkeypatch):
    monkeypatch.setenv("PROVIDER_B_KEY", "destination-b")
    network = RecordingTransport()
    transport = ModelRoutingTransport(
        routes(),
        {},
        default_model="canonical-b",
        network_transport=network,
    )

    response = await transport.handle_async_request(request())

    assert response.status_code == 200
    assert network.requests[0].url.host == "provider-b.example"
    assert "model" not in json.loads(network.requests[0].content)
    assert network.requests[0].headers["authorization"] == "Bearer destination-b"


@pytest.mark.asyncio
async def test_unknown_model_and_missing_route_credential_stop_before_network(monkeypatch):
    monkeypatch.delenv("PROVIDER_A_KEY", raising=False)
    network = RecordingTransport()
    transport = ModelRoutingTransport(
        routes(),
        {},
        default_model="canonical-a",
        network_transport=network,
    )

    unknown = await transport.handle_async_request(request("client-selected-model"))
    missing_key = await transport.handle_async_request(request("canonical-a"))

    assert unknown.status_code == 404
    assert not unknown.is_stream_consumed
    assert json.loads(await unknown.aread())["error"]["type"] == "model_routing_error"
    assert missing_key.status_code == 502
    assert "PROVIDER_A_KEY" in json.loads(await missing_key.aread())["error"]["message"]
    assert network.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("model,status", [("unknown", 404), ("canonical-a", 502)])
async def test_routing_errors_are_readable_through_the_application(monkeypatch, rails_configs, model, status, stream):
    monkeypatch.delenv("PROVIDER_A_KEY", raising=False)
    network = RecordingTransport()
    transport = ModelRoutingTransport(routes(), {}, default_model="canonical-a", network_transport=network)
    application = create_proxy_application(
        providers=(create_openai_chat_integration(base_url="http://model-router.internal/v1"),),
        rails_configs=rails_configs,
        http_transport_factory=lambda: transport,
    )
    async with application.router.lifespan_context(application):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=application), base_url="http://proxy") as client:
            response = await client.post(
                "/v1/chat/completions",
                json={"model": model, "messages": [{"role": "user", "content": "hello"}], "stream": stream},
            )
    assert response.status_code == status, response.text
    assert response.json()["error"]["type"] == "model_routing_error"
    assert network.requests == []
    assert network.close_calls == 1


@pytest.mark.asyncio
async def test_non_json_upload_reaches_network_before_its_body_is_consumed(monkeypatch):
    monkeypatch.setenv("PROVIDER_B_KEY", "destination-b")
    events = []
    bodies = []

    class UploadTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            events.append("network")
            assert request.url.raw_path == b"/openai/v1/files/a%2Fb?trace=%252F&trace=two"
            async for chunk in request.stream:
                if chunk:
                    events.append("received")
                    bodies.append(chunk)
            return httpx.Response(200, stream=httpx.ByteStream(b"uploaded"))

    async def upload():
        assert events == ["network"]
        events.append("first")
        yield b"first chunk"
        assert events == ["network", "first", "received"]
        events.append("second")
        yield b"second chunk"

    transport = ModelRoutingTransport(routes(), {}, default_model="canonical-b", network_transport=UploadTransport())
    application = create_proxy_application(
        providers=(create_openai_chat_integration(base_url="http://model-router.internal/v1"),),
        http_transport_factory=lambda: transport,
    )
    async with application.router.lifespan_context(application):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=application), base_url="http://proxy") as client:
            response = await client.post(
                "/v1/files/a%2Fb?trace=%252F&trace=two",
                content=upload(),
                headers={"content-type": "application/octet-stream"},
            )
    assert response.status_code == 200
    assert response.content == b"uploaded"
    assert bodies == [b"first chunk", b"second chunk"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "method,headers,chunks,expected_length,expected_encoding",
    [
        ("POST", {"Content-Length": "6"}, (b"abc", b"def"), "6", None),
        ("POST", {"Content-Length": "0"}, (), "0", None),
        ("GET", {"Content-Length": "0"}, (), "0", None),
        ("GET", {}, (), None, None),
        ("HEAD", {}, (), None, None),
        ("POST", {}, (b"abc", b"def"), None, "chunked"),
    ],
)
async def test_routed_passthrough_preserves_request_framing(
    monkeypatch, method, headers, chunks, expected_length, expected_encoding
):
    monkeypatch.setenv("PROVIDER_B_KEY", "destination-b")
    received = []
    reads = 0
    calls = []

    async def upload():
        nonlocal reads
        for chunk in chunks:
            reads += 1
            yield chunk

    class UploadTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            calls.append(request)
            assert reads == 0
            assert request.url.raw_path == b"/openai/v1/files"
            assert request.headers["host"] == "provider-b.example"
            assert request.headers.get("content-length") == expected_length
            assert request.headers.get("transfer-encoding") == expected_encoding
            assert isinstance(request.stream, httpx.AsyncByteStream)
            async for chunk in request.stream:
                if chunk:
                    received.append(chunk)
            return httpx.Response(204, stream=httpx.ByteStream(b""))

    transport = ModelRoutingTransport(routes(), {}, default_model="canonical-b", network_transport=UploadTransport())
    application = create_proxy_application(
        providers=(create_openai_chat_integration(base_url="http://model-router.internal/v1"),),
        http_transport_factory=lambda: transport,
    )
    async with application.router.lifespan_context(application):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=application), base_url="http://proxy") as client:
            response = await client.request(
                method,
                "/v1/files",
                content=upload() if chunks else b"",
                headers={"content-type": "application/octet-stream", **headers},
            )

    assert response.status_code == 204
    assert len(calls) == 1
    assert received == list(chunks)


@pytest.mark.asyncio
async def test_json_model_selection_has_a_bound_and_stops_reading_at_overflow(monkeypatch):
    monkeypatch.setenv("PROVIDER_A_KEY", "destination-a")
    network = RecordingTransport()
    transport = ModelRoutingTransport(
        routes(), {}, default_model="canonical-a", network_transport=network, max_json_body_bytes=8
    )

    async def upload():
        yield b'{"model"'
        yield b':"too large"}'
        pytest.fail("the router must stop reading once its JSON limit is exceeded")

    response = await transport.handle_async_request(
        httpx.Request(
            "POST",
            "http://model-router.internal/v1/embeddings",
            content=upload(),
            headers={"content-type": "application/json"},
        )
    )
    assert response.status_code == 413
    assert json.loads(await response.aread())["error"]["type"] == "model_routing_error"
    assert network.requests == []
    await transport.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("suffix", [b"/chat/completions", b"/files/a%2Fb", b"//files/a%252Fb"])
async def test_destination_replaces_the_logical_base_and_preserves_raw_suffix(monkeypatch, suffix):
    monkeypatch.setenv("PROVIDER_B_KEY", "destination-b")
    network = RecordingTransport()
    transport = ModelRoutingTransport(routes(), {}, default_model="canonical-b", network_transport=network)
    original_url = httpx.URL("http://model-router.internal").copy_with(raw_path=b"/v1" + suffix + b"?x=%252F&x=%2f")
    await transport.handle_async_request(httpx.Request("POST", original_url, content=b"opaque"))
    assert network.requests[0].url.raw_path == b"/openai/v1" + suffix + b"?x=%252F&x=%2f"


@pytest.mark.asyncio
async def test_json_at_the_limit_is_reframed_and_forwarded(monkeypatch):
    monkeypatch.setenv("PROVIDER_A_KEY", "destination-a")
    network = RecordingTransport()
    body = b'{"model":"canonical-a"}'
    transport = ModelRoutingTransport(
        routes(), {}, default_model="canonical-a", network_transport=network, max_json_body_bytes=len(body)
    )

    async def upload():
        yield body[:10]
        yield body[10:]

    response = await transport.handle_async_request(
        httpx.Request(
            "POST",
            "http://model-router.internal/v1/chat/completions",
            content=upload(),
            headers={"content-type": "application/json; charset=utf-8"},
        )
    )
    assert response.status_code == 200
    upstream = network.requests[0]
    assert json.loads(upstream.content) == {"model": "deployment-a"}
    assert upstream.headers["content-length"] == str(len(upstream.content))
    assert "transfer-encoding" not in upstream.headers


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_json_limit_must_be_a_positive_integer(limit):
    with pytest.raises(ValueError, match="JSON body limit"):
        ModelRoutingTransport(routes(), {}, default_model="canonical-a", max_json_body_bytes=limit)


@pytest.mark.asyncio
async def test_transport_closes_its_owned_network_transport():
    network = RecordingTransport()
    transport = ModelRoutingTransport(
        routes(),
        {},
        default_model="canonical-a",
        network_transport=network,
    )

    await transport.aclose()

    assert network.close_calls == 1


@pytest.mark.parametrize(
    "url",
    [
        "provider.example/v1",
        "ftp://provider.example/v1",
        "https://user:secret@provider.example/v1",
        "https://provider.example/v1?destination=other",
        "https://provider.example/v1#fragment",
    ],
)
def test_route_destinations_must_be_fixed_safe_http_origins(url):
    with pytest.raises(ValueError, match="base URL"):
        ModelRoute(httpx.URL(url), "deployment")


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("decision", ["allow", "input-block", "output-block"])
async def test_routed_generation_enforces_real_rails(monkeypatch, rails_configs, stream, decision):
    monkeypatch.setenv("PROVIDER_A_KEY", "destination-a")
    calls = []
    text = "BLOCK_OUTPUT" if decision == "output-block" else "hello"
    wire = (
        stream_event(text) + b"data: [DONE]\n\n"
        if stream
        else json.dumps({"choices": [{"message": {"role": "assistant", "content": text}}]}).encode()
    )

    def upstream(request):
        calls.append(request)
        assert request.url.host == "provider-a.example"
        assert json.loads(request.content)["model"] == "deployment-a"
        assert request.headers["authorization"] == "Bearer destination-a"
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream" if stream else "application/json"},
            stream=httpx.ByteStream(wire),
        )

    def transport():
        return ModelRoutingTransport(
            routes(),
            {"public-a": "canonical-a"},
            default_model="canonical-a",
            network_transport=httpx.MockTransport(upstream),
        )

    application = create_proxy_application(
        providers=(create_openai_chat_integration(base_url="http://model-router.internal/v1"),),
        rails_configs=rails_configs,
        http_transport_factory=transport,
    )
    async with application.router.lifespan_context(application):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=application), base_url="http://proxy") as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "public-a",
                    "messages": [{"role": "user", "content": "BLOCK_INPUT" if decision == "input-block" else "hello"}],
                    "stream": stream,
                },
                headers={"authorization": "Bearer caller-credential"},
            )

    assert len(calls) == (0 if decision == "input-block" else 1)
    if decision == "allow":
        assert response.status_code == 200
        assert response.content == wire
    else:
        assert response.status_code == (200 if stream and decision == "output-block" else 400)
        assert "content_blocked" in response.text
        assert "BLOCK_OUTPUT" not in response.text
