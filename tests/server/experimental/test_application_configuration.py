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

"""Exercise declarative selection through the existing proxy assembly."""

import json
from dataclasses import replace
from pathlib import Path

import httpx
import pytest
from fastapi import Request
from fastapi.testclient import TestClient
from pydantic import ValidationError

from nemoguardrails.server.experimental import _application, _configuration, application
from nemoguardrails.server.experimental.application import (
    ConfigEntry,
    ProviderConfig,
    ProxyConfig,
    create_proxy_application,
)
from nemoguardrails.server.experimental.providers.openai.chat_completions.endpoint import CHAT_COMPLETIONS_ENDPOINT
from nemoguardrails.server.experimental.providers.openai.dialects.litellm.chat_completions import (
    LITELLM_CHAT_COMPLETIONS_ENDPOINT,
)
from nemoguardrails.server.experimental.providers.openai.dialects.litellm.responses import LITELLM_RESPONSES_ENDPOINT
from nemoguardrails.server.experimental.providers.openai.integration import create_openai_integration
from nemoguardrails.server.experimental.providers.openai.responses.endpoint import RESPONSES_ENDPOINT
from nemoguardrails.server.experimental.rails import ConfigContext
from nemoguardrails.server.experimental.rails._repository import GuardrailsRepository
from tests.server.experimental.test_openai_responses_integration import ResponseBody
from tests.server.experimental.test_openai_responses_projections import response_payload
from tests.server.experimental.test_openai_responses_streaming_projections import event_bytes


def provider(**overrides):
    return ProviderConfig(
        **{
            "provider": "openai",
            "instance_id": "gateway",
            "base_url": "https://provider.test/v1",
            "mount_path": "/gateway",
            "operations": ("chat_completions", "responses"),
            "rails_config_id": "standard",
            **overrides,
        }
    )


@pytest.fixture
def entries(tmp_path):
    result = []
    for name, pattern in (("standard", "BLOCK_INPUT"), ("strict", "BLOCK_INPUT|STRICT_ONLY")):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "config.yml").write_text(
            f"""models: []
rails:
  config:
    regex_detection:
      input:
        patterns: ["{pattern}"]
      output:
        patterns: [BLOCK_OUTPUT]
  input:
    flows: [regex check input]
  output:
    flows: [regex check output]
    streaming:
      enabled: true
      chunk_size: 8
      context_size: 0
      stream_first: false
""",
            encoding="utf-8",
        )
        result.append(ConfigEntry(name, directory))
    return tuple(result)


def config(entries, *providers, **overrides):
    return ProxyConfig(providers=providers or (provider(),), rails_configs=entries, **overrides)


def request_payload(operation, text="question", streaming=False):
    subject = {"messages": [{"role": "user", "content": text}]} if operation == "chat_completions" else {"input": text}
    return {"model": "test-model", "stream": streaming, **subject}


def route(operation, mount="/gateway"):
    return mount + "/v1/" + ("chat/completions" if operation == "chat_completions" else "responses")


class Upstream:
    """Provide synthetic protocol bytes through a real HTTPX client owner."""

    def __init__(self, text="answer"):
        self.text = text
        self.requests = []
        self.bodies = []
        self.factory_calls = 0

    def factory(self):
        self.factory_calls += 1
        return httpx.MockTransport(self.handle)

    async def handle(self, request):
        self.requests.append(request)
        payload = json.loads(await request.aread())
        if request.url.path.endswith("/chat/completions"):
            if payload.get("stream"):
                wire = (
                    b"data: "
                    + json.dumps(
                        {
                            "object": "chat.completion.chunk",
                            "choices": [{"index": 0, "delta": {"content": self.text}}],
                        }
                    ).encode()
                    + b"\n\ndata: [DONE]\n\n"
                )
            else:
                wire = json.dumps({"choices": [{"message": {"role": "assistant", "content": self.text}}]}).encode()
        elif payload.get("stream"):
            wire = b"".join(
                (
                    event_bytes("response.created", response={"output": []}, sequence_number=0),
                    event_bytes(
                        "response.output_text.delta",
                        delta=self.text,
                        item_id="msg_test",
                        output_index=1,
                        content_index=0,
                        sequence_number=1,
                    ),
                    event_bytes("response.completed", response=response_payload(self.text), sequence_number=2),
                )
            )
        else:
            wire = json.dumps(response_payload(self.text)).encode()
        body = ResponseBody(wire)
        self.bodies.append(body)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream" if payload.get("stream") else "application/json"},
            stream=body,
        )


@pytest.mark.parametrize("dialect", [None, "litellm"])
@pytest.mark.parametrize(
    "operations",
    [("chat_completions",), ("responses",), ("chat_completions", "responses"), ("responses", "chat_completions")],
)
def test_exact_binding_selection_and_direct_assembly_equivalence(entries, dialect, operations):
    declaration = provider(dialect=dialect, operations=operations)
    actual = _configuration.resolve_provider(declaration)
    native = {"chat_completions": CHAT_COMPLETIONS_ENDPOINT, "responses": RESPONSES_ENDPOINT}
    alternative = {"chat_completions": LITELLM_CHAT_COMPLETIONS_ENDPOINT, "responses": LITELLM_RESPONSES_ENDPOINT}
    expected_endpoints = tuple((native if dialect is None else alternative)[operation] for operation in operations)
    expected = replace(
        create_openai_integration(instance_id="gateway", base_url="https://provider.test/v1", mount_path="/gateway"),
        endpoints=expected_endpoints,
    )
    assert actual == expected
    assert all(left is right for left, right in zip(actual.endpoints, expected_endpoints))
    declared_app = create_proxy_application(config(entries, declaration))
    direct_app = create_proxy_application(providers=(expected,), rails_configs=(entries[0],))
    assert declared_app.openapi() == direct_app.openapi()


@pytest.mark.parametrize(
    "field,value",
    [
        ("provider", "anthropic"),
        ("provider", "unknown"),
        ("dialect", "openai"),
        ("dialect", "unknown"),
        ("operations", ()),
        ("operations", ("responses", "responses")),
        ("operations", ("unknown",)),
        ("operations", "responses"),
        ("operations", {"responses": True}),
        ("operations", {"responses"}),
        ("operations", iter(["responses"])),
        ("rails_config_id", ""),
        ("rails_config_id", " standard "),
        ("rails_config_ids", ["standard"]),
        ("extra", True),
    ],
)
def test_invalid_provider_declarations_are_rejected(field, value):
    with pytest.raises(ValidationError):
        provider(**{field: value})


def test_native_binding_is_explicitly_provider_relative():
    declaration = provider().model_copy(update={"provider": "anthropic"})
    with pytest.raises(ValueError, match="Unsupported"):
        _configuration.resolve_provider(declaration)


@pytest.mark.parametrize(
    "override",
    [
        {"providers": ()},
        {"rails_configs": ()},
        {"providers": "openai"},
        {"rails_configs": {}},
        {"max_request_body_bytes": True},
        {"max_request_body_bytes": 0},
        {"max_response_body_bytes": "1024"},
        {"timeout": float("inf")},
        {"timeout": True},
        {"timeout": 0},
        {"extra": "unknown"},
    ],
)
def test_invalid_application_data_is_rejected(entries, override):
    with pytest.raises(ValidationError):
        ProxyConfig(**{"providers": (provider(),), "rails_configs": entries, **override})


def test_catalogue_references_and_nested_unknown_keys_are_validated(entries):
    with pytest.raises(ValidationError, match="unknown Rails"):
        config(entries, provider(rails_config_id="missing"))
    with pytest.raises(ValidationError, match="unique"):
        config((entries[0], entries[0]))
    data = config(entries).model_dump(mode="json")
    data["rails_configs"][0]["unexpected"] = True
    with pytest.raises(ValidationError):
        ProxyConfig.model_validate(data)


def test_schema_roundtrip_and_immutable_selection_without_file_loading(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Data validation must not load Rails or create runtime resources")

    monkeypatch.setattr(_application.ServerRuntime, "create", forbidden)
    monkeypatch.setattr(GuardrailsRepository, "startup", forbidden)
    operations = ["responses"]
    declarations = [provider(operations=operations)]
    catalogue = [ConfigEntry("standard", Path("/nonexistent/rails"))]
    value = ProxyConfig(providers=declarations, rails_configs=catalogue)
    operations.append("chat_completions")
    declarations.clear()
    catalogue.clear()
    assert value.providers[0].operations == ("responses",)
    assert len(value.rails_configs) == 1
    with pytest.raises(ValidationError):
        value.providers[0].rails_config_id = "other"
    assert ProxyConfig.model_validate_json(value.model_dump_json()) == value
    assert ProxyConfig.model_json_schema()["$defs"]["ConfigEntry"]["additionalProperties"] is False
    assert application.__all__ == ["ConfigEntry", "ProviderConfig", "ProxyConfig", "create_proxy_application"]
    assert application.create_proxy_application is _application.create_proxy_application


@pytest.mark.parametrize(
    "setting,value",
    [
        ("providers", ()),
        ("rails_configs", ()),
        ("path_prefix", ""),
        ("max_request_body_bytes", 1024),
        ("max_response_body_bytes", 1024),
        ("timeout", 30.0),
    ],
)
def test_config_rejects_direct_settings_even_when_they_equal_defaults(entries, setting, value):
    with pytest.raises(TypeError, match="direct assembly"):
        create_proxy_application(config(entries), **{setting: value})


@pytest.mark.parametrize(
    "changes",
    [
        {"base_url": "relative"},
        {"base_url": "https://user:secret@provider.test"},
        {"base_url": "https://provider.test?secret=value"},
        {"mount_path": "/trailing/"},
        {"instance_id": "invalid id"},
    ],
)
def test_invalid_routes_and_upstreams_fail_before_resource_creation(entries, monkeypatch, changes):
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid declarations must fail before runtime ownership begins")

    monkeypatch.setattr(_application.ServerRuntime, "create", forbidden)
    monkeypatch.setattr(GuardrailsRepository, "startup", forbidden)
    with pytest.raises(ValueError):
        create_proxy_application(config(entries, provider(**changes)), http_transport_factory=forbidden)


@pytest.mark.parametrize("duplicate", ["instance", "mount"])
def test_provider_collisions_reuse_assembly_validation(entries, duplicate):
    other = provider(
        instance_id="gateway" if duplicate == "instance" else "other",
        mount_path="/gateway" if duplicate == "mount" else "/other",
    )
    with pytest.raises(ValueError, match="unique"):
        create_proxy_application(config(entries, provider(), other))


def test_dynamic_selection_requires_a_callable_even_with_one_entry(entries):
    declaration = config((entries[0],), provider(rails_config_id=None))
    with pytest.raises(ValueError, match="require a resolver"):
        create_proxy_application(declaration)
    with pytest.raises(TypeError, match="callable"):
        create_proxy_application(declaration, config_resolver="not-a-callback")


def test_factory_revalidates_copied_models_and_requires_typed_configuration(entries):
    original = config(entries)
    with pytest.raises(TypeError, match="ProxyConfig"):
        create_proxy_application(original.model_dump())
    with pytest.raises(TypeError, match="Supply"):
        create_proxy_application()
    changed = original.model_copy(update={"providers": (provider().model_copy(update={"dialect": "unknown"}),)})
    with pytest.raises(ValidationError):
        create_proxy_application(changed)


def test_application_settings_and_explicit_upstream_are_preserved(entries, monkeypatch):
    monkeypatch.setenv("NEMO_GUARDRAILS_OPENAI_PROXY_UPSTREAM_BASE_URL", "https://unselected.test")
    original = _application.include_provider_routes
    observed = {}

    def capture(*args, **kwargs):
        observed.update(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(_application, "include_provider_routes", capture)
    upstream = Upstream()
    app = create_proxy_application(
        config(
            entries,
            path_prefix="/proxy",
            max_request_body_bytes=512,
            max_response_body_bytes=2048,
            timeout=7.0,
        ),
        http_transport_factory=upstream.factory,
    )
    with TestClient(app) as client:
        assert client.post("/proxy" + route("responses"), json=request_payload("responses")).status_code == 200
        assert (
            client.post("/proxy" + route("responses"), json=request_payload("responses", "a" * 600)).status_code == 413
        )
    assert observed["max_request_body_bytes"] == 512
    assert observed["max_response_body_bytes"] == 2048
    assert observed["timeout"] == 7.0
    assert observed["path_prefix"] == "/proxy"
    assert len(upstream.requests) == 1
    assert str(upstream.requests[0].url) == "https://provider.test/v1/responses"


def test_large_catalogue_needs_no_provider_level_enumeration(entries):
    catalogue = tuple(ConfigEntry(f"tenant-{index}", entries[0].config_path) for index in range(48))
    fixed = provider(instance_id="fixed", mount_path="/fixed", rails_config_id="tenant-47")
    dynamic = provider(instance_id="dynamic", mount_path="/dynamic", rails_config_id=None)
    app = create_proxy_application(config(catalogue, fixed, dynamic), config_resolver=lambda context: "tenant-46")
    choose = app.state.experimental_config_resolver
    request = Request({"type": "http"})
    assert choose(ConfigContext(request, "openai", "fixed", "responses.create")) == "tenant-47"
    assert choose(ConfigContext(request, "openai", "dynamic", "responses.create")) == "tenant-46"
    assert len(app.state.experimental_rails_configs) == 48


@pytest.mark.parametrize("dialect", [None, "litellm"])
@pytest.mark.parametrize("operation", ["chat_completions", "responses"])
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("decision", ["allow", "input", "output"])
def test_declarations_use_real_rails_for_buffered_and_streaming_checks(
    entries, dialect, operation, streaming, decision
):
    upstream = Upstream("BLOCK_OUTPUT" if decision == "output" else "answer")
    declaration = config(entries, provider(dialect=dialect, operations=(operation,)))
    app = create_proxy_application(declaration, http_transport_factory=upstream.factory)
    assert upstream.factory_calls == 0
    with TestClient(app) as client:
        response = client.post(
            route(operation),
            json=request_payload(operation, "BLOCK_INPUT" if decision == "input" else "question", streaming),
        )
    assert upstream.factory_calls == 1
    assert all(body.closed for body in upstream.bodies)
    if decision == "allow":
        assert response.status_code == 200
        assert response.content == upstream.bodies[0].body
    elif decision == "input":
        assert response.status_code == 400
        assert not upstream.requests
        assert response.json()["error"]["code"] == "content_blocked"
    else:
        assert response.status_code == (200 if streaming else 400)
        assert b"content_blocked" in response.content
        assert b"BLOCK_OUTPUT" not in response.content


@pytest.mark.parametrize("async_resolver", [False, True])
def test_mixed_fixed_dynamic_instances_select_from_shared_catalogue(entries, async_resolver):
    contexts = []

    def choose(context):
        contexts.append(context)
        assert context.request.state.tenant == "trusted"
        return "standard"

    async def choose_async(context):
        return choose(context)

    upstream = Upstream()
    declaration = config(
        entries,
        provider(instance_id="public", mount_path="/public", rails_config_id="strict"),
        provider(
            instance_id="internal", mount_path="/internal", base_url="https://internal.test/v1", rails_config_id=None
        ),
    )
    app = create_proxy_application(
        declaration, config_resolver=choose_async if async_resolver else choose, http_transport_factory=upstream.factory
    )

    @app.middleware("http")
    async def authenticate(request, call_next):
        request.state.tenant = "trusted"
        return await call_next(request)

    with TestClient(app) as client:
        for operation in ("chat_completions", "responses"):
            assert (
                client.post(route(operation, "/public"), json=request_payload(operation, "STRICT_ONLY")).status_code
                == 400
            )
            assert (
                client.post(route(operation, "/internal"), json=request_payload(operation, "STRICT_ONLY")).status_code
                == 200
            )
    assert [item.provider_instance_id for item in contexts] == ["internal", "internal"]
    assert [item.operation_name for item in contexts] == ["chat_completions.create", "responses.create"]
    assert all(item.provider_type == "openai" for item in contexts)
    assert all(request.url.host == "internal.test" for request in upstream.requests)
    assert upstream.factory_calls == 1


def test_all_fixed_providers_need_no_resolver_for_multiple_entries(entries):
    upstream = Upstream()
    app = create_proxy_application(
        config(
            entries,
            provider(instance_id="public", mount_path="/public", rails_config_id="strict"),
            provider(instance_id="internal", mount_path="/internal", rails_config_id="standard"),
        ),
        http_transport_factory=upstream.factory,
    )
    with TestClient(app) as client:
        assert (
            client.post(route("responses", "/public"), json=request_payload("responses", "STRICT_ONLY")).status_code
            == 400
        )
        assert (
            client.post(route("responses", "/internal"), json=request_payload("responses", "STRICT_ONLY")).status_code
            == 200
        )


def test_fixed_provider_never_invokes_explicit_user_resolver(entries):
    def forbidden(context):
        pytest.fail("A fixed provider must bypass the user resolver")

    upstream = Upstream()
    app = create_proxy_application(config(entries), config_resolver=forbidden, http_transport_factory=upstream.factory)
    with TestClient(app) as client:
        assert client.post(route("responses"), json=request_payload("responses")).status_code == 200


@pytest.mark.parametrize("result", ["missing", "", None, 1, RuntimeError("private resolver failure")])
@pytest.mark.parametrize("async_resolver", [False, True])
def test_dynamic_selection_failure_is_closed_without_fallback(entries, result, async_resolver):
    def choose(context):
        if isinstance(result, Exception):
            raise result
        return result

    async def choose_async(context):
        return choose(context)

    upstream = Upstream()
    app = create_proxy_application(
        config(entries, provider(rails_config_id=None)),
        config_resolver=choose_async if async_resolver else choose,
        http_transport_factory=upstream.factory,
    )
    with TestClient(app) as client:
        response = client.post(route("responses"), json=request_payload("responses"))
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "config_resolution_failed"
    assert "private resolver failure" not in response.text
    assert not upstream.requests


@pytest.mark.parametrize("selected,unselected", [("responses", "chat_completions"), ("chat_completions", "responses")])
@pytest.mark.parametrize("streaming", [False, True])
def test_unselected_routes_remain_passthrough_without_rails_resolution(entries, selected, unselected, streaming):
    def forbidden(context):
        pytest.fail("Passthrough must not resolve Rails")

    upstream = Upstream("BLOCK_OUTPUT")
    app = create_proxy_application(
        config(entries, provider(operations=(selected,), rails_config_id=None)),
        config_resolver=forbidden,
        http_transport_factory=upstream.factory,
    )
    with TestClient(app) as client:
        response = client.post(route(unselected), json=request_payload(unselected, "BLOCK_INPUT", streaming))
    assert response.status_code == 200
    assert response.content == upstream.bodies[0].body
    assert upstream.bodies[0].closed


def test_direct_mode_retains_explicit_single_entry_resolver_and_custom_binding(entries):
    upstream = Upstream()
    contexts = []
    custom = replace(
        create_openai_integration(base_url="https://provider.test/v1"),
        endpoints=(LITELLM_RESPONSES_ENDPOINT,),
    )

    def choose(context):
        contexts.append(context)
        return entries[0].id

    with pytest.raises(ValueError, match="Multiple Rails"):
        create_proxy_application(providers=(custom,), rails_configs=entries)
    app = create_proxy_application(
        providers=(custom,),
        rails_configs=(entries[0],),
        config_resolver=choose,
        http_transport_factory=upstream.factory,
    )
    with TestClient(app) as client:
        assert client.post("/v1/responses", json=request_payload("responses")).status_code == 200
    assert len(contexts) == 1
    assert isinstance(contexts[0], ConfigContext)
