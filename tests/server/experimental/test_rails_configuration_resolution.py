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
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi.testclient import TestClient

from nemoguardrails import RailsConfig
from nemoguardrails.server.experimental._application import create_proxy_application
from nemoguardrails.server.experimental._content_checker import (
    ContentAllowed,
    ContentInspectionPolicy,
    StreamBufferingPolicy,
    UnsupportedContentCheckerConfiguration,
)
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


def test_unknown_resolved_config_fails_closed_before_upstream(tmp_path):
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


@pytest.mark.parametrize("rail_type", ["tool_input", "tool_output"])
@pytest.mark.parametrize("stream", [False, True], ids=["buffered", "streaming"])
def test_per_tool_rails_fail_before_upstream(tmp_path, monkeypatch, rail_type, stream):
    """Reject configured tool flows even when the global flow list is empty."""

    config = RailsConfig.from_content(
        config={"models": [], "rails": {rail_type: {"per_tool": {"unused": [], "lookup": ["tool check"]}}}}
    )
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


@pytest.mark.parametrize("rail_type", ["tool_input", "tool_output"])
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
