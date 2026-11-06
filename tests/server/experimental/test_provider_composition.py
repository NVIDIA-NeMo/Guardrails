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
from nemoguardrails.server.experimental._http_kernel import GuardedOperationPath
from nemoguardrails.server.experimental._provider_routes import include_provider_routes
from nemoguardrails.server.experimental.provider.integration import UpstreamSettings
from nemoguardrails.server.experimental.providers.openai.dialects.litellm.chat_completions import (
    LITELLM_CHAT_COMPLETIONS_ENDPOINT,
)
from nemoguardrails.server.experimental.providers.openai.integration import create_openai_chat_integration
from tests.server.experimental.chat_fixtures import stream_event
from tests.server.experimental.checkers import StaticChecker


class PassthroughChecker:
    def inspection_policy(self):
        return ContentInspectionPolicy(False, False)

    async def check_input(self, _check):
        raise AssertionError("input inspection is disabled")

    async def check_output(self, _check):
        raise AssertionError("output inspection is disabled")


async def resolve_passthrough_checker(_request, _provider_type, _provider_instance_id, _operation_name):
    return PassthroughChecker()


class StaticResponseStream(httpx.AsyncByteStream):
    def __init__(self, body):
        self.body = body

    async def __aiter__(self):
        yield self.body


def provider_response():
    return httpx.Response(
        200,
        headers={"content-type": "application/json"},
        stream=StaticResponseStream(b'{"object":"list","data":[]}'),
    )


def create_composed_app(providers, handler, *, checker=None):
    upstream_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    application = FastAPI()

    async def resolve_checker(_request, _provider_type, _provider_instance_id, _operation_name):
        return PassthroughChecker() if checker is None else checker

    include_provider_routes(
        application,
        providers,
        resolve_checker=resolve_checker,
        http_client=upstream_client,
    )
    proxy_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=application),
        base_url="http://proxy.test",
    )
    return application, upstream_client, proxy_client


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("mount_path", ["", "/providers/named"])
@pytest.mark.parametrize(
    ("base_path", "alias"),
    [
        ("/v1", "/chat/completions"),
        ("/v1", "/chat%2Fcompletions"),
        ("/v1", "/chat//completions"),
        ("/v1", "/chat/completions/"),
        ("/v1", "/chat/completions%3Fignored"),
        ("/v1", "/chat/completions%23ignored"),
        ("/%761", "/chat/completions"),
        ("/proxy/v1", "/proxy/v1/v1/chat/completions"),
        ("/v1/chat", "/completions"),
    ],
)
async def test_composition_rejects_upstream_aliases_before_checks_or_dispatch(streaming, mount_path, base_path, alias):
    """Keep alternate spellings of a guarded destination out of the catch-all."""

    provider = create_openai_chat_integration(base_url=f"https://provider.example{base_path}", mount_path=mount_path)
    checker = StaticChecker(
        ContentInspectionPolicy(True, True, StreamBufferingPolicy(1, 0)), ContentBlocked("blocked by review checker")
    )
    requests = []

    async def respond(request):
        """Record any request that incorrectly escapes the guarded boundary."""

        requests.append(request)
        return provider_response()

    _, upstream, client = create_composed_app((provider,), respond, checker=checker)
    try:
        payload = {"messages": [{"role": "user", "content": "must be blocked"}], "stream": streaming}
        response = await client.post(mount_path + alias, json=payload)
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "non_canonical_path"
        assert checker.calls == []
        assert requests == []

        canonical = await client.post(mount_path + "/v1/chat/completions", json=payload)
        assert canonical.status_code == 400
        assert checker.text_calls == [("input", "must be blocked")]
        assert requests == []
    finally:
        await client.aclose()
        await upstream.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("reserve_alias", [False, True])
async def test_composition_respects_declared_aliases_without_upstream_rewriting(reserve_alias):
    """Use explicit ownership when the upstream mapping does not add a prefix."""

    provider = create_openai_chat_integration(base_url="https://provider.example")
    if reserve_alias:
        endpoint = provider.endpoints[0]
        provider = replace(
            provider,
            endpoints=(
                replace(
                    endpoint,
                    operation_paths=(*endpoint.operation_paths, GuardedOperationPath("/chat/completions")),
                ),
            ),
        )
    requests = []

    async def respond(request):
        """Retain the actual provider path for the pass-through assertion."""

        requests.append(request)
        return provider_response()

    _, upstream, client = create_composed_app((provider,), respond)
    try:
        response = await client.post("/chat/completions?trace=%2Fopaque", content=b"unconfigured operation")
    finally:
        await client.aclose()
        await upstream.aclose()

    if reserve_alias:
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "non_canonical_path"
        assert requests == []
    else:
        assert response.status_code == 200
        assert requests[0].url.raw_path == b"/chat/completions?trace=%2Fopaque"
        assert requests[0].content == b"unconfigured operation"


@pytest.mark.asyncio
@pytest.mark.parametrize("mount_path", ["", "/providers/named"])
@pytest.mark.parametrize(
    ("method", "path", "status", "code"),
    [
        ("POST", "/chat/completions", 422, "non_canonical_path"),
        ("GET", "/chat/completions", 422, "non_canonical_path"),
        ("PUT", "/chat/completions", 405, "method_not_allowed"),
        ("PUT", "/v1/chat/completions", 405, "method_not_allowed"),
        ("POST", "/v2/chat/completions", 422, "non_canonical_path"),
        ("GET", "/v1/chat/completions/", 405, "method_not_allowed"),
        ("POST", "/v1/chat/completions%3Fignored", 422, "non_canonical_path"),
        ("GET", "/v1/chat/completions%23ignored", 405, "method_not_allowed"),
    ],
)
async def test_composed_path_rejections_do_not_read_the_body(mount_path, method, path, status, code):
    class UnreadableBody(httpx.AsyncByteStream):
        async def __aiter__(self):
            raise AssertionError("Rejected paths must not consume the request body.")
            yield

    async def respond(_request):
        raise AssertionError("Rejected paths must not reach the upstream.")

    checker = StaticChecker(ContentInspectionPolicy(True, True))
    provider = create_openai_chat_integration(base_url="https://provider.example/v1", mount_path=mount_path)
    _, upstream, client = create_composed_app((provider,), respond, checker=checker)
    try:
        response = await client.request(method, mount_path + path, content=UnreadableBody())
    finally:
        await client.aclose()
        await upstream.aclose()

    assert response.status_code == status
    assert response.json()["error"]["code"] == code
    assert checker.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("litellm", [False, True])
async def test_composed_instance_uses_its_declared_dialect_under_inspection(streaming, litellm):
    outputs = []

    class Checker:
        def inspection_policy(self):
            return ContentInspectionPolicy(True, True, StreamBufferingPolicy(1, 0))

        async def check_input(self, check):
            assert check.message.content == "hello"
            return ContentAllowed()

        async def check_output(self, check):
            outputs.append(check.output_content)
            return ContentAllowed()

    provider = create_openai_chat_integration(
        instance_id="selected",
        base_url="https://provider.example/v1",
        mount_path="/selected",
    )
    if litellm:
        provider = replace(provider, endpoints=(LITELLM_CHAT_COMPLETIONS_ENDPOINT,))
    if streaming:
        body = (
            b'data: {"object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"answer","provider_specific_fields":{}},'
            b'"finish_reason":null}]}\n\n'
            b'data: {"object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
            b"data: [DONE]\n\n"
        )
    else:
        body = json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "answer",
                            "provider_specific_fields": {"refusal": None},
                        }
                    }
                ]
            }
        ).encode()

    async def respond(request):
        assert request.headers["accept-encoding"] == "identity"
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream" if streaming else "application/json"},
            stream=StaticResponseStream(body),
        )

    _, upstream_client, proxy_client = create_composed_app((provider,), respond, checker=Checker())
    try:
        response = await proxy_client.post(
            "/selected/v1/chat/completions",
            json={"model": "provider-model", "messages": [{"role": "user", "content": "hello"}], "stream": streaming},
        )
    finally:
        await proxy_client.aclose()
        await upstream_client.aclose()

    if litellm:
        assert response.status_code == 200
        assert response.content == body
        assert outputs == ["answer"]
    else:
        assert response.status_code == (200 if streaming else 502)
        assert b'"error"' in response.content
        assert b"answer" not in response.content
        assert outputs == []


@pytest.mark.asyncio
async def test_multiple_instances_share_openai_definitions_and_route_to_distinct_upstreams():
    providers = (
        create_openai_chat_integration(instance_id="first", base_url="https://first.example/v1"),
        create_openai_chat_integration(
            instance_id="second",
            base_url="https://second.example/v1",
            mount_path="/providers/second",
        ),
    )
    requests = []

    async def respond(request):
        requests.append(request)
        return provider_response()

    application, upstream_client, proxy_client = create_composed_app(providers, respond)
    try:
        first_response = await proxy_client.get("/v1/models")
        second_response = await proxy_client.get("/providers/second/v1/models")
        first_chat_list = await proxy_client.get("/v1/chat/completions?limit=2")
        second_chat_list = await proxy_client.get("/providers/second/v1/chat/completions?limit=3")
        guarded_response = await proxy_client.post(
            "/providers/second/v1/chat/completions",
            json={
                "model": "provider-model",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
    finally:
        await proxy_client.aclose()
        await upstream_client.aclose()

    assert first_response.status_code == 200
    assert second_response.status_code == 200
    assert first_chat_list.status_code == 200
    assert second_chat_list.status_code == 200
    assert guarded_response.status_code == 200
    assert [str(request.url) for request in requests] == [
        "https://first.example/v1/models",
        "https://second.example/v1/models",
        "https://first.example/v1/chat/completions?limit=2",
        "https://second.example/v1/chat/completions?limit=3",
        "https://second.example/v1/chat/completions",
    ]
    assert json.loads(requests[-1].content) == {
        "model": "provider-model",
        "messages": [{"role": "user", "content": "hello"}],
    }
    assert providers[0].provider_type == providers[1].provider_type == "openai"
    assert providers[0].endpoints == providers[1].endpoints
    paths = application.openapi()["paths"]
    assert paths["/v1/chat/completions"]["post"]["operationId"] == "first.chat_completions.create"
    assert paths["/providers/second/v1/chat/completions"]["post"]["operationId"] == "second.chat_completions.create"


@pytest.mark.asyncio
async def test_buffered_only_endpoint_is_composed_without_a_stream_dispatcher():
    provider = create_openai_chat_integration(base_url="https://provider.example/v1")
    provider = replace(provider, endpoints=(replace(provider.endpoints[0], stream=None),))
    requests = []

    async def respond(request):
        requests.append(request)
        return provider_response()

    _, upstream, client = create_composed_app((provider,), respond)
    try:
        payload = {"messages": [{"role": "user", "content": "hello"}]}
        buffered = await client.post("/v1/chat/completions", json=payload)
        streaming = await client.post("/v1/chat/completions", json={**payload, "stream": True})
    finally:
        await client.aclose()
        await upstream.aclose()
    assert buffered.status_code == 200
    assert streaming.status_code == 422
    assert len(requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_dialects_of_one_protocol_can_coexist_without_widening_each_other(streaming):
    class Checker:
        def inspection_policy(self):
            return ContentInspectionPolicy(True, True, StreamBufferingPolicy(1, 0))

        async def check_input(self, _check):
            return ContentAllowed()

        async def check_output(self, _check):
            return ContentAllowed()

    standard = create_openai_chat_integration(instance_id="standard", base_url="https://standard.example/v1")
    dialect = create_openai_chat_integration(
        instance_id="litellm",
        base_url="https://litellm.example/v1",
        mount_path="/providers/litellm",
        endpoint=LITELLM_CHAT_COMPLETIONS_ENDPOINT,
    )
    if streaming:
        body = (
            b'data: {"object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"answer","provider_specific_fields":{}},"finish_reason":null}]}\n\n'
            b'data: {"object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
            b"data: [DONE]\n\n"
        )
    else:
        body = b'{"choices":[{"message":{"role":"assistant","content":"answer","provider_specific_fields":{}}}]}'

    async def respond(_request):
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream" if streaming else "application/json"},
            stream=StaticResponseStream(body),
        )

    _, upstream, client = create_composed_app((standard, dialect), respond, checker=Checker())
    try:
        payload = {"messages": [{"role": "user", "content": "hello"}], "stream": streaming}
        strict = await client.post("/v1/chat/completions", json=payload)
        compatible = await client.post("/providers/litellm/v1/chat/completions", json=payload)
    finally:
        await client.aclose()
        await upstream.aclose()
    assert standard.provider_type == dialect.provider_type == "openai"
    assert b'"error"' in strict.content
    assert b"answer" not in strict.content
    assert compatible.status_code == 200
    assert compatible.content == body


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_composed_guarded_routes_map_connection_failures_to_native_errors(streaming):
    async def fail(request):
        raise httpx.ConnectError("private upstream detail", request=request)

    provider = create_openai_chat_integration(base_url="https://provider.example/v1")
    _, upstream, client = create_composed_app((provider,), fail)
    try:
        response = await client.post(
            "/v1/chat/completions", json={"messages": [{"role": "user", "content": "hello"}], "stream": streaming}
        )
    finally:
        await client.aclose()
        await upstream.aclose()
    assert response.status_code == 502
    assert response.json()["error"]["code"] == ("upstream_stream_failed" if streaming else "upstream_request_failed")
    assert "private upstream detail" not in response.text


@pytest.mark.asyncio
async def test_mounted_instance_streams_through_its_selected_upstream():
    provider = create_openai_chat_integration(
        instance_id="streaming",
        base_url="https://streaming.example/v1",
        mount_path="/providers/streaming",
    )
    requests = []

    async def respond(request):
        requests.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=StaticResponseStream(b'data: {"choices":[]}\n\ndata: [DONE]\n\n'),
        )

    _, upstream_client, proxy_client = create_composed_app((provider,), respond)
    try:
        response = await proxy_client.post(
            "/providers/streaming/v1/chat/completions",
            json={
                "model": "provider-model",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
            },
        )
    finally:
        await proxy_client.aclose()
        await upstream_client.aclose()

    assert response.status_code == 200
    assert response.content == b'data: {"choices":[]}\n\ndata: [DONE]\n\n'
    assert str(requests[0].url) == "https://streaming.example/v1/chat/completions"


@pytest.mark.asyncio
async def test_nested_provider_catchalls_route_most_specific_mount_first():
    providers = (
        create_openai_chat_integration(
            instance_id="parent",
            base_url="https://parent.example/v1",
            mount_path="/providers",
        ),
        create_openai_chat_integration(
            instance_id="nested",
            base_url="https://nested.example/v1",
            mount_path="/providers/instances/nested",
        ),
    )
    requests = []

    async def respond(request):
        requests.append(request)
        return provider_response()

    _, upstream_client, proxy_client = create_composed_app(providers, respond)
    try:
        response = await proxy_client.get("/providers/instances/nested/v1/models")
    finally:
        await proxy_client.aclose()
        await upstream_client.aclose()

    assert response.status_code == 200
    assert str(requests[0].url) == "https://nested.example/v1/models"


@pytest.mark.asyncio
async def test_shared_application_prefix_does_not_change_upstream_paths():
    provider = create_openai_chat_integration(base_url="https://provider.example/v1")
    requests = []

    async def respond(request):
        requests.append(request)
        return provider_response()

    upstream_client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    application = FastAPI()
    include_provider_routes(
        application,
        (provider,),
        resolve_checker=resolve_passthrough_checker,
        http_client=upstream_client,
        path_prefix="/guardrails",
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=application),
        base_url="http://proxy.test",
    ) as proxy_client:
        response = await proxy_client.get("/guardrails/v1/models")
    await upstream_client.aclose()

    assert response.status_code == 200
    assert str(requests[0].url) == "https://provider.example/v1/models"


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("provider_mount", ["", "/providers/named"])
@pytest.mark.parametrize("path_prefix", ["", "/guardrails"])
@pytest.mark.parametrize(("outer_mount", "outer_url"), [("/outer", "/%6Futer"), ("/café", "/caf%C3%A9")])
async def test_outer_asgi_mount_preserves_guards_and_provider_paths(
    streaming, provider_mount, path_prefix, outer_mount, outer_url
):
    """Strip the ASGI root once while retaining provider ownership and raw suffixes."""

    provider = create_openai_chat_integration(base_url="https://provider.example/v1", mount_path=provider_mount)
    checker = StaticChecker(ContentInspectionPolicy(True, True, StreamBufferingPolicy(1, 0)))

    async def resolve_checker(_request, _provider_type, _provider_instance_id, _operation_name):
        return checker

    requests = []
    body = (
        stream_event("answer")
        + b'data: {"object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'
        if streaming
        else b'{"choices":[{"index":0,"message":{"role":"assistant","content":"answer"}}]}'
    )

    async def respond(request):
        """Capture the wire URL and return a complete guarded response."""

        requests.append(request)
        if request.method == "GET":
            return provider_response()
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream" if streaming else "application/json"},
            stream=StaticResponseStream(body),
        )

    application = FastAPI()

    @application.get("/health")
    async def health():
        """Reserve an application-owned route beneath the outer mount."""

        return {"status": "ok"}

    parent = FastAPI()
    parent.mount(outer_mount, application)
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as upstream:
        include_provider_routes(
            application,
            (provider,),
            resolve_checker=resolve_checker,
            http_client=upstream,
            path_prefix=path_prefix,
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=parent, raise_app_exceptions=False), base_url="http://proxy.test"
        ) as client:
            prefix = outer_url + path_prefix + provider_mount
            query = "?trace=%2Fopaque&trace=%252Fopaque"
            response = await client.post(
                prefix + "/v1/chat/completions" + query,
                json={"messages": [{"role": "user", "content": "hello"}], "stream": streaming},
            )
            assert response.status_code == 200
            assert response.content == body
            assert checker.text_calls == [("input", "hello"), ("output", "answer")]
            assert requests[0].url.raw_path == b"/v1/chat/completions?trace=%2Fopaque&trace=%252Fopaque"

            relay = await client.get(prefix + "/v1/mo%64els/caf%C3%A9" + query)
            assert relay.status_code == 200
            assert requests[1].url.raw_path == b"/v1/mo%64els/caf%C3%A9?trace=%2Fopaque&trace=%252Fopaque"

            chat_list = await client.get(prefix + "/v1/chat/completions?limit=2")
            assert chat_list.status_code == 200
            assert requests[2].url.raw_path == b"/v1/chat/completions?limit=2"

            alias = await client.post(prefix + "/chat/completions", content=b"must not reach upstream")
            assert alias.status_code == 422
            assert alias.json()["error"]["code"] == "non_canonical_path"
            wrong_method = await client.put(prefix + "/v1/chat/completions")
            assert wrong_method.status_code == 405
            reserved = await client.post(outer_url + "/health")
            assert reserved.status_code == 405
            assert len(requests) == 3
            assert checker.text_calls == [("input", "hello"), ("output", "answer")]


@pytest.mark.asyncio
async def test_existing_application_routes_are_reserved_from_root_provider_catchall():
    provider = create_openai_chat_integration(base_url="https://provider.example/v1")
    requests = []

    async def respond(request):
        requests.append(request)
        return provider_response()

    upstream_client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    application = FastAPI()

    @application.get("/health")
    async def health():
        return {"status": "ok"}

    include_provider_routes(
        application,
        (provider,),
        resolve_checker=resolve_passthrough_checker,
        http_client=upstream_client,
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=application),
        base_url="http://proxy.test",
    ) as proxy_client:
        get_response = await proxy_client.get("/health")
        post_response = await proxy_client.post("/health")
    await upstream_client.aclose()

    assert get_response.status_code == 200
    assert post_response.status_code == 405
    assert requests == []


@pytest.mark.asyncio
async def test_mounted_guarded_path_rejects_undeclared_method_without_dispatch():
    provider = create_openai_chat_integration(
        base_url="https://provider.example/v1",
        mount_path="/providers/openai",
    )
    requests = []

    async def respond(request):
        requests.append(request)
        return provider_response()

    _, upstream_client, proxy_client = create_composed_app((provider,), respond)
    try:
        response = await proxy_client.put("/providers/openai/v1/chat/completions")
    finally:
        await proxy_client.aclose()
        await upstream_client.aclose()

    assert response.status_code == 405
    assert response.headers["allow"] == "GET, POST"
    assert requests == []


@pytest.mark.asyncio
async def test_mounted_future_api_version_cannot_bypass_guarded_operation():
    provider = create_openai_chat_integration(
        base_url="https://provider.example/v1",
        mount_path="/providers/openai",
    )
    requests = []

    async def respond(request):
        requests.append(request)
        return provider_response()

    _, upstream_client, proxy_client = create_composed_app((provider,), respond)
    try:
        response = await proxy_client.post(
            "/providers/openai/v2/chat/completions",
            json={"model": "provider-model", "messages": [{"role": "user", "content": "hello"}]},
        )
    finally:
        await proxy_client.aclose()
        await upstream_client.aclose()

    assert response.status_code == 422
    assert requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/v1/chat/completions/", "/v1//chat/completions"])
async def test_declared_passthrough_method_requires_the_canonical_guarded_path(path):
    provider = create_openai_chat_integration(base_url="https://provider.example/v1")
    requests = []

    async def respond(request):
        requests.append(request)
        return provider_response()

    _, upstream_client, proxy_client = create_composed_app((provider,), respond)
    try:
        response = await proxy_client.get(path)
    finally:
        await proxy_client.aclose()
        await upstream_client.aclose()

    assert response.status_code == 405
    assert response.headers["allow"] == "GET, POST"
    assert requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "method,path,status,code",
    [
        ("PUT", "/v1/chat/completions", 405, "method_not_allowed"),
        ("HEAD", "/v1/chat/completions", 405, "method_not_allowed"),
        ("POST", "/v1/%2e/chat/completions", 422, "non_canonical_path"),
        ("POST", "/v1/chat/completions%3F", 422, "non_canonical_path"),
        ("POST", "/v1/chat/completions%23", 422, "non_canonical_path"),
        ("GET", "/v1%5Cmodels", 400, "invalid_request_path"),
        ("POST", "/openapi.json%3F", 405, "method_not_allowed"),
    ],
)
async def test_composed_catchall_preserves_native_route_rejections(method, path, status, code):
    provider = create_openai_chat_integration(base_url="https://provider.example/v1")
    requests = []

    async def respond(request):
        requests.append(request)
        return provider_response()

    _, upstream_client, proxy_client = create_composed_app((provider,), respond)
    try:
        response = await proxy_client.request(method, path)
    finally:
        await proxy_client.aclose()
        await upstream_client.aclose()

    assert response.status_code == status
    assert requests == []
    if method == "HEAD":
        assert response.content == b""
    else:
        assert response.json()["error"]["code"] == code


@pytest.mark.asyncio
async def test_provider_control_headers_stop_before_outbound_dispatch():
    provider = replace(
        create_openai_chat_integration(base_url="https://provider.example/v1"),
        control_headers=frozenset({"x-proxy-control"}),
    )
    requests = []

    async def respond(request):
        requests.append(request)
        return provider_response()

    _, upstream_client, proxy_client = create_composed_app((provider,), respond)
    try:
        response = await proxy_client.get(
            "/v1/models",
            headers={
                "authorization": "Bearer provider-key",
                "x-proxy-control": "local-only",
            },
        )
    finally:
        await proxy_client.aclose()
        await upstream_client.aclose()

    assert response.status_code == 200
    assert requests[0].headers["authorization"] == "Bearer provider-key"
    assert "x-proxy-control" not in requests[0].headers


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("providers", "message"),
    [
        (
            (
                create_openai_chat_integration(instance_id="shared"),
                create_openai_chat_integration(instance_id="shared", mount_path="/providers/second"),
            ),
            "instance ids must be unique",
        ),
        (
            (
                create_openai_chat_integration(instance_id="first"),
                create_openai_chat_integration(instance_id="second"),
            ),
            "mount paths must be unique",
        ),
    ],
)
async def test_provider_composition_rejects_ambiguous_instance_identity(providers, message):
    async with httpx.AsyncClient() as upstream_client:
        with pytest.raises(ValueError, match=message):
            include_provider_routes(
                FastAPI(),
                providers,
                resolve_checker=resolve_passthrough_checker,
                http_client=upstream_client,
            )


@pytest.mark.asyncio
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("path_prefix", ["", "/guardrails"])
@pytest.mark.parametrize("parent_mount", ["", "/providers/root"])
@pytest.mark.parametrize(
    "mount_path",
    ["/v1", "/v1/chat", "/v1/chat/completions", "/v2", "/v2/chat", "/v2/chat/completions", "/litellm"],
)
async def test_composition_rejects_mounts_that_shadow_guarded_routes(reverse, path_prefix, parent_mount, mount_path):
    """Reject a more specific catch-all before it can hide another instance's guard."""

    root = create_openai_chat_integration(instance_id="root", mount_path=parent_mount)
    shadow = create_openai_chat_integration(
        instance_id="shadow",
        mount_path=parent_mount + mount_path,
        endpoint=replace(root.endpoints[0], route_path="/v1/other", operation_paths=()),
    )
    providers = (root, shadow)
    application = FastAPI()
    original_routes = tuple(application.routes)
    async with httpx.AsyncClient() as upstream:
        with pytest.raises(ValueError, match="would shadow a guarded route"):
            include_provider_routes(
                application,
                providers[::-1] if reverse else providers,
                resolve_checker=resolve_passthrough_checker,
                http_client=upstream,
                path_prefix=path_prefix,
            )
        assert tuple(application.routes) == original_routes


@pytest.mark.asyncio
@pytest.mark.parametrize("parent_mount", ["", "/providers/root"])
@pytest.mark.parametrize("mount_path", ["/providers/named", "/v2/models", "/v2/chatty", "/v2/chat/completions/history"])
async def test_composition_accepts_mounts_disjoint_from_guarded_patterns(parent_mount, mount_path):
    root = create_openai_chat_integration(instance_id="root", mount_path=parent_mount)
    child = create_openai_chat_integration(instance_id="child", mount_path=parent_mount + mount_path)
    requests = []

    async def respond(request):
        requests.append(request)
        return provider_response()

    _, upstream, client = create_composed_app((root, child), respond)
    try:
        response = await client.get(parent_mount + mount_path + "/v1/models")
    finally:
        await client.aclose()
        await upstream.aclose()

    assert response.status_code == 200
    assert len(requests) == 1
    assert requests[0].url.path == "/v1/models"


@pytest.mark.asyncio
@pytest.mark.parametrize(("mount_path", "conflicts"), [("/2", True), ("/2/chat", True), ("/named", False)])
async def test_mount_conflicts_respect_guarded_parameter_converters(mount_path, conflicts):
    root = create_openai_chat_integration(instance_id="root")
    root = replace(
        root,
        endpoints=(
            replace(
                root.endpoints[0],
                route_path="/1/chat/completions",
                operation_paths=(GuardedOperationPath("/{version:int}/chat/completions"),),
            ),
        ),
    )
    child = create_openai_chat_integration(instance_id="child", mount_path=mount_path)
    application = FastAPI()
    original_routes = tuple(application.routes)
    async with httpx.AsyncClient() as upstream:
        if conflicts:
            with pytest.raises(ValueError, match="would shadow a guarded route"):
                include_provider_routes(
                    application, (root, child), resolve_checker=resolve_passthrough_checker, http_client=upstream
                )
            assert tuple(application.routes) == original_routes
        else:
            include_provider_routes(
                application, (root, child), resolve_checker=resolve_passthrough_checker, http_client=upstream
            )


@pytest.mark.asyncio
async def test_composition_does_not_install_routers_before_every_plan_is_validated():
    root = create_openai_chat_integration(instance_id="root")
    root = replace(
        root,
        endpoints=(
            replace(root.endpoints[0], operation_paths=(GuardedOperationPath("/v1/chat/completions"),)),
            replace(
                root.endpoints[0],
                route_path="/{version}/chat/completions",
                operation_name="other.create",
                operation_paths=(),
            ),
        ),
    )
    child = create_openai_chat_integration(instance_id="child", mount_path="/providers/named")
    application = FastAPI()
    original_routes = tuple(application.routes)
    async with httpx.AsyncClient() as upstream:
        with pytest.raises(ValueError, match="must not overlap"):
            include_provider_routes(
                application, (root, child), resolve_checker=resolve_passthrough_checker, http_client=upstream
            )
    assert tuple(application.routes) == original_routes


def test_provider_mounts_have_one_canonical_static_form():
    for mount_path in ("providers/example", "/providers/example/", "/providers/{instance}", "/providers//example"):
        with pytest.raises(ValueError, match="mount path"):
            create_openai_chat_integration(mount_path=mount_path)


@pytest.mark.asyncio
async def test_provider_composition_rejects_guarded_routes_that_collide_after_mounting():
    template = create_openai_chat_integration(instance_id="root")
    endpoint = template.endpoints[0]
    colliding_path = "/providers/second/v1/chat/completions"
    colliding_endpoint = replace(
        endpoint,
        route_path=colliding_path,
        operation_paths=(GuardedOperationPath(colliding_path, frozenset({"POST"})),),
    )
    root_provider = replace(template, provider_type="synthetic", endpoints=(colliding_endpoint,))
    second_provider = create_openai_chat_integration(
        instance_id="second",
        mount_path="/providers/second",
    )

    async with httpx.AsyncClient() as upstream_client:
        with pytest.raises(ValueError, match="globally unique after provider mounting"):
            include_provider_routes(
                FastAPI(),
                (root_provider, second_provider),
                resolve_checker=resolve_passthrough_checker,
                http_client=upstream_client,
            )


def test_openai_instance_defaults_are_resolved_when_the_factory_is_called(monkeypatch):
    monkeypatch.setenv("NEMO_GUARDRAILS_OPENAI_PROXY_UPSTREAM_BASE_URL", "https://openai.example/v1")

    provider = create_openai_chat_integration()

    assert provider.provider_type == "openai"
    assert provider.instance_id == "openai"
    assert provider.upstream == UpstreamSettings(base_url="https://openai.example/v1")


def test_shared_provider_definition_creates_request_local_stream_state():
    first = create_openai_chat_integration(instance_id="first")
    second = create_openai_chat_integration(instance_id="second", mount_path="/providers/second")
    first_binding = first.endpoints[0].stream
    second_binding = second.endpoints[0].stream
    assert first_binding is not None
    assert second_binding is not None

    first_stream = first_binding.create_adapter()
    second_stream = second_binding.create_adapter()

    assert first.endpoints == second.endpoints
    assert first_stream is not second_stream
    assert first_stream.hooks is not second_stream.hooks
    assert first_stream.classifier is second_stream.classifier
