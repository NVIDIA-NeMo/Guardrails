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

"""Exercise policy differences and production defaults through public proxy configuration."""

import asyncio
import json

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from nemoguardrails.server.experimental import _configuration
from nemoguardrails.server.experimental.application import ConfigEntry, ProviderConfig, create_proxy_application
from nemoguardrails.server.experimental.provider.projection_policy import EXTENSION
from nemoguardrails.server.experimental.provider.types import UnknownContentFieldPolicy
from nemoguardrails.server.experimental.providers.openai.chat_completions.endpoint import CHAT_COMPLETIONS_ENDPOINT
from nemoguardrails.server.experimental.providers.openai.dialects.litellm.responses import LITELLM_RESPONSES_ENDPOINT
from nemoguardrails.server.experimental.providers.openai.integration import (
    create_openai_chat_integration,
    create_openai_chat_router,
    create_openai_integration,
)
from nemoguardrails.server.experimental.providers.openai.responses.endpoint import RESPONSES_ENDPOINT
from tests.server.experimental import test_application_configuration as configuration_tests
from tests.server.experimental.test_application_configuration import Upstream, config, provider, request_payload, route
from tests.server.experimental.test_openai_chat_integration import StaticChecker, _request_body
from tests.server.experimental.test_openai_responses_projections import response_payload
from tests.server.experimental.test_openai_responses_streaming_integration import checker, post_stream
from tests.server.experimental.test_openai_responses_streaming_projections import event_bytes

entries = configuration_tests.entries


class MetadataUpstream(Upstream):
    """Add synthetic future fields without changing the supported text subject."""

    def __init__(self, location, text="answer"):
        super().__init__(text)
        self.location = location

    async def handle(self, request):
        response = await super().handle(request)
        body = self.bodies[-1]
        streaming = json.loads(request.content).get("stream", False)
        chat = request.url.path.endswith("/chat/completions")

        def extend(document):
            if self.location == "root":
                target = document
            elif chat:
                target = document["choices"][0]["delta" if streaming else "message"]
            elif streaming:
                target = document.get("response")
            else:
                target = document["output"][1]["content"][0]
            if target is not None:
                target["future_metadata"] = {"counter": 1}
            return document

        if streaming:
            body.body = b"".join(
                b"data: " + json.dumps(extend(json.loads(line[6:]))).encode() + b"\n"
                if line.startswith(b"data: ") and line.strip() != b"data: [DONE]"
                else line
                for line in body.body.splitlines(keepends=True)
            )
        else:
            body.body = json.dumps(extend(json.loads(body.body))).encode()
        return response


@pytest.mark.parametrize("value", ["allow", "forbid", *UnknownContentFieldPolicy])
def test_unknown_fields_selection_roundtrips_and_reaches_the_integration(value):
    declaration = provider(unknown_fields=value)
    expected = UnknownContentFieldPolicy(value)
    assert declaration.unknown_fields is expected
    assert _configuration.resolve_provider(declaration).unknown_fields is expected
    assert ProviderConfig.model_validate_json(declaration.model_dump_json()) == declaration
    with pytest.raises(ValidationError):
        declaration.unknown_fields = UnknownContentFieldPolicy.FORBID


def test_unknown_fields_default_allows_provider_additions():
    declaration = provider()
    assert declaration.unknown_fields is UnknownContentFieldPolicy.ALLOW
    assert _configuration.resolve_provider(declaration).unknown_fields is UnknownContentFieldPolicy.ALLOW


@pytest.mark.parametrize("factory", [create_openai_integration, create_openai_chat_integration])
def test_python_factory_default_allows_provider_additions(factory):
    assert factory().unknown_fields is UnknownContentFieldPolicy.ALLOW


@pytest.mark.parametrize("unknown_fields", [None, "allow", "forbid"], ids=["default", "allow", "forbid"])
@pytest.mark.parametrize("dialect", [None, "litellm"])
@pytest.mark.parametrize("operation", ["chat_completions", "responses"])
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("stage", ["request_root", "request_nested", "response_root", "response_nested"])
def test_unknown_field_policy_applies_to_every_provider_payload(
    entries, unknown_fields, dialect, operation, streaming, stage
):
    upstream = MetadataUpstream(stage.removeprefix("response_")) if stage.startswith("response_") else Upstream()
    options = {} if unknown_fields is None else {"unknown_fields": unknown_fields}
    app = create_proxy_application(
        config(entries, provider(dialect=dialect, operations=(operation,), **options)),
        http_transport_factory=upstream.factory,
    )
    payload = request_payload(operation, streaming=streaming)
    if stage == "request_root":
        payload["future_option"] = {"counter": 1}
    elif stage == "request_nested":
        target = payload["messages"][0] if operation == "chat_completions" else payload.setdefault("reasoning", {})
        target["future_option"] = {"counter": 1}
    original = json.dumps(payload, indent=2).encode()
    with TestClient(app) as client:
        response = client.post(route(operation), content=original, headers={"content-type": "application/json"})
    if unknown_fields != "forbid":
        assert response.status_code == 200, response.text
        assert len(upstream.requests) == 1
        assert upstream.requests[0].content == original
        assert response.content == upstream.bodies[0].body
    elif stage.startswith("request_"):
        assert response.status_code == 422
        assert response.json()["error"]["code"] == f"unsupported_{operation}_shape"
        assert not upstream.requests
    else:
        assert response.status_code == (200 if streaming else 502)
        assert f"unsupported_{operation}_response_shape" in response.text
        assert b"answer" not in response.content
        assert len(upstream.requests) == 1
        assert upstream.requests[0].content == original
    assert all(body.closed for body in upstream.bodies)


@pytest.mark.parametrize("dialect", [None, "litellm"])
@pytest.mark.parametrize("operation", ["chat_completions", "responses"])
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("blocked_stage", ["input", "output"])
def test_permissive_metadata_preserves_input_and_output_enforcement(
    entries, dialect, operation, streaming, blocked_stage
):
    upstream = MetadataUpstream("nested", "BLOCK_OUTPUT" if blocked_stage == "output" else "answer")
    app = create_proxy_application(
        config(entries, provider(dialect=dialect, operations=(operation,))),
        http_transport_factory=upstream.factory,
    )
    payload = request_payload(operation, "BLOCK_INPUT" if blocked_stage == "input" else "question", streaming)
    payload["future_option"] = {"counter": 1}
    with TestClient(app) as client:
        response = client.post(route(operation), json=payload)
    assert response.status_code == (200 if streaming and blocked_stage == "output" else 400)
    assert "content_blocked" in response.text
    assert "BLOCK_OUTPUT" not in response.text
    assert len(upstream.requests) == (1 if blocked_stage == "output" else 0)
    assert all(body.closed for body in upstream.bodies)


def test_unknown_cache_option_follows_selected_policy(unknown_fields):
    payload = {"input": "question", "store": False, "prompt_cache_options": {"future_control": True}}
    if unknown_fields is UnknownContentFieldPolicy.FORBID:
        with pytest.raises(ValidationError):
            RESPONSES_ENDPOINT.guarded_request_model.validate_payload(payload, unknown_content_fields=unknown_fields)
    else:
        projection = RESPONSES_ENDPOINT.guarded_request_model.validate_payload(
            payload, unknown_content_fields=unknown_fields
        )
        assert projection.locate_guarded_message(payload).message.content == "question"


@pytest.mark.parametrize("unknown_fields", ["allow", "forbid"])
@pytest.mark.parametrize("prewarm", [True, None, 0, 1, "false"])
def test_unknown_field_policy_keeps_cache_only_generation_disabled(entries, unknown_fields, prewarm):
    upstream = Upstream()
    app = create_proxy_application(
        config(entries, provider(unknown_fields=unknown_fields)), http_transport_factory=upstream.factory
    )
    payload = request_payload("responses")
    payload["prompt_cache_options"] = {"prewarm": prewarm}
    with TestClient(app) as client:
        response = client.post(route("responses"), json=payload)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "unsupported_responses_shape"
    assert not upstream.requests


@pytest.mark.parametrize("unknown_fields", ["allow", "forbid"])
@pytest.mark.parametrize("operation", ["chat_completions", "responses"])
@pytest.mark.parametrize("field", ["tools", "Tools", "stream"])
def test_unknown_field_policy_does_not_relax_declared_request_constraints(entries, unknown_fields, operation, field):
    upstream = Upstream()
    app = create_proxy_application(
        config(entries, provider(unknown_fields=unknown_fields)), http_transport_factory=upstream.factory
    )
    payload = request_payload(operation)
    payload[field] = "true" if field == "stream" else [{"type": "function", "name": "hidden"}]
    with TestClient(app) as client:
        response = client.post(route(operation), json=payload)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == f"unsupported_{operation}_shape"
    assert not upstream.requests


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["chat_completions", "responses"])
@pytest.mark.parametrize("streaming", [False, True])
async def test_instances_keep_independent_unknown_field_policies(entries, operation, streaming):
    upstream = MetadataUpstream("nested")
    app = create_proxy_application(
        config(
            entries,
            provider(instance_id="compatible", mount_path="/compatible", unknown_fields="allow"),
            provider(instance_id="strict", mount_path="/strict", unknown_fields="forbid"),
        ),
        http_transport_factory=upstream.factory,
    )
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            for _ in range(2):
                compatible, strict = await asyncio.gather(
                    client.post(
                        route(operation, mount="/compatible"), json=request_payload(operation, streaming=streaming)
                    ),
                    client.post(
                        route(operation, mount="/strict"), json=request_payload(operation, streaming=streaming)
                    ),
                )
                assert compatible.status_code == 200
                assert b"future_metadata" in compatible.content
                assert b"answer" in compatible.content
                assert strict.status_code == (200 if streaming else 502)
                assert f"unsupported_{operation}_response_shape" in strict.text
                assert b"answer" not in strict.content
    assert len(upstream.requests) == 4
    assert all(body.closed for body in upstream.bodies)


@pytest.mark.parametrize("inspect_input", [False, True], ids=["relay", "input-only"])
@pytest.mark.parametrize("operation", ["chat_completions", "responses"])
@pytest.mark.parametrize("streaming", [False, True])
def test_strict_requests_and_uninspected_responses_keep_separate_boundaries(
    tmp_path, inspect_input, operation, streaming
):
    directory = tmp_path / "rails"
    directory.mkdir()
    source = "models: []\n"
    if inspect_input:
        source += """rails:
  config:
    regex_detection:
      input:
        patterns: [BLOCK_INPUT]
  input:
    flows: [regex check input]
"""
    (directory / "config.yml").write_text(source, encoding="utf-8")
    upstream = MetadataUpstream("nested")
    app = create_proxy_application(
        config((ConfigEntry("standard", directory),), provider(unknown_fields="forbid")),
        http_transport_factory=upstream.factory,
    )
    payload = request_payload(operation, streaming=streaming)
    with TestClient(app) as client:
        response = client.post(route(operation), json=payload)
        assert response.status_code == 200
        assert response.content == upstream.bodies[0].body
        assert b"future_metadata" in response.content
        rejected = client.post(route(operation), json={**payload, "future_option": True})
        assert rejected.status_code == 422
        assert rejected.json()["error"]["code"] == f"unsupported_{operation}_shape"
        if inspect_input:
            blocked = client.post(route(operation), json=request_payload(operation, "BLOCK_INPUT", streaming))
            assert blocked.status_code == 400
            assert blocked.json()["error"]["code"] == "content_blocked"
    assert len(upstream.requests) == 1
    assert all(body.closed for body in upstream.bodies)


@pytest.mark.parametrize(
    ("location", "member", "accepted"),
    [
        ("root", "model", True),
        ("root", "metadata", True),
        ("root", "future_field", False),
        ("root", "Model", False),
        ("message", "future_field", False),
        ("message", "Content", False),
    ],
)
@pytest.mark.parametrize("unknown_fields", [None, *UnknownContentFieldPolicy], ids=["default", "allow", "forbid"])
def test_openai_request_openapi_matches_selected_object_policy(
    location, member, accepted, unknown_fields, export_policy_validator
):
    """Published schemas and runtime apply the selected inventory and alias policies."""
    app = FastAPI()

    async def dispatch(_request):
        raise AssertionError("OpenAPI generation must not dispatch")

    options = {} if unknown_fields is None else {"unknown_fields": unknown_fields}
    selected = UnknownContentFieldPolicy.ALLOW if unknown_fields is None else unknown_fields
    app.include_router(create_openai_chat_router(checker=StaticChecker(), dispatch=dispatch, **options))
    document = app.openapi()
    schema = document["paths"]["/v1/chat/completions"]["post"]["requestBody"]["content"]["application/json"]["schema"]
    message_schema = schema["properties"]["messages"]["items"]
    assert schema["additionalProperties"] is (selected == UnknownContentFieldPolicy.ALLOW)
    assert schema[EXTENSION]["unknown_fields"] == "configurable"
    assert message_schema["additionalProperties"] is (selected == UnknownContentFieldPolicy.ALLOW)
    assert message_schema[EXTENSION]["unknown_fields"] == "configurable"
    assert message_schema[EXTENSION]["reject_case_aliases"] is True
    assert schema["properties"]["model"][EXTENSION]["classification"] == "opaque"

    payload = json.loads(_request_body())
    target = payload if location == "root" else payload["messages"][0]
    target[member] = {"arbitrary": [1, "opaque"]} if member == "metadata" else "value"
    accepted = accepted or (member == "future_field" and selected == UnknownContentFieldPolicy.ALLOW)
    assert export_policy_validator(schema, selected).is_valid(payload) is accepted
    if accepted:
        CHAT_COMPLETIONS_ENDPOINT.guarded_request_model.validate_payload(payload, unknown_content_fields=selected)
    else:
        with pytest.raises(ValidationError):
            CHAT_COMPLETIONS_ENDPOINT.guarded_request_model.validate_payload(payload, unknown_content_fields=selected)


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", [RESPONSES_ENDPOINT, LITELLM_RESPONSES_ENDPOINT], ids=["native", "litellm"])
@pytest.mark.parametrize("unknown_fields", ["allow", "forbid"])
@pytest.mark.parametrize("kind", ["reasoning", "error"])
@pytest.mark.parametrize(
    "metadata", [None, {"id": "metadata-only"}, {"output": [{"type": "message", "id": "unrelated"}]}]
)
async def test_response_member_on_another_event_follows_unknown_field_policy(endpoint, unknown_fields, kind, metadata):
    if kind == "reasoning":
        extra = event_bytes("response.output_item.done", item={"type": "reasoning", "summary": []}, response=metadata)
        terminal = event_bytes("response.completed", response=response_payload())
    else:
        extra = event_bytes("error", message="provider diagnostic", code="provider_error", response=metadata)
        terminal = b""
    wire = (
        event_bytes("response.output_text.delta", delta="answer", item_id="msg_test", output_index=1) + extra + terminal
    )
    inspector = checker()
    response, source, dispatches = await post_stream(
        [wire], inspector, endpoint=endpoint, unknown_fields=unknown_fields
    )
    assert response.status_code == 200
    assert len(dispatches) == 1 and source.closed
    if unknown_fields == "allow":
        assert response.content == wire
        assert [(kind, check.output_content) for kind, check in inspector.calls if kind == "output"] == [
            ("output", "answer")
        ]
    else:
        assert b"unsupported_responses_response_shape" in response.content
        assert b"answer" not in response.content
        assert [kind for kind, _ in inspector.calls] == ["input"]


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", [RESPONSES_ENDPOINT, LITELLM_RESPONSES_ENDPOINT], ids=["native", "litellm"])
async def test_permissive_metadata_does_not_relax_lifecycle_identity_checks(endpoint):
    payload = response_payload()
    payload["output"][1]["id"] = "different-item"
    payload["future_metadata"] = {"counter": 1}
    wire = event_bytes("response.output_text.delta", delta="answer", item_id="msg_test", output_index=1) + event_bytes(
        "response.completed", response=payload
    )
    response, source, _ = await post_stream([wire], checker(), endpoint=endpoint, unknown_fields="allow")
    assert b"unsupported_responses_response_shape" in response.content
    assert b"answer" not in response.content
    assert source.closed
