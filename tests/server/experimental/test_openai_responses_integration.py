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

"""Exercise explicit OpenAI composition through real Rails and HTTPX ownership."""

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from nemoguardrails.server.experimental._application import create_proxy_application
from nemoguardrails.server.experimental.providers.openai.chat_completions.endpoint import CHAT_COMPLETIONS_ENDPOINT
from nemoguardrails.server.experimental.providers.openai.dialects.litellm.chat_completions import (
    LITELLM_CHAT_COMPLETIONS_ENDPOINT,
)
from nemoguardrails.server.experimental.providers.openai.integration import (
    create_openai_chat_integration,
    create_openai_integration,
)
from nemoguardrails.server.experimental.providers.openai.responses.endpoint import RESPONSES_ENDPOINT
from nemoguardrails.server.experimental.rails import ConfigEntry
from tests.server.experimental.test_openai_responses_projections import response_payload
from tests.server.experimental.test_openai_responses_streaming_projections import event_bytes


class ResponseBody(httpx.AsyncByteStream):
    """Record ownership of synthetic provider bytes without a network service."""

    def __init__(self, body):
        self.body = body
        self.closed = False

    async def __aiter__(self):
        for index in range(0, len(self.body), 19):
            yield self.body[index : index + 19]

    async def aclose(self):
        self.closed = True


@pytest.fixture
def proxy(tmp_path, request):
    config = tmp_path / "rails"
    config.mkdir()
    (config / "config.yml").write_text("""models: []
rails:
  config:
    regex_detection:
      input:
        patterns: [BLOCK_INPUT]
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
""")
    requests = []
    bodies = []
    settings = {"output": "answer", "status": 200}

    async def handler(request):
        requests.append(request)
        payload = json.loads(await request.aread())
        text = settings["output"]
        streaming = payload.get("stream") is True and settings["status"] == 200
        if settings["status"] != 200:
            wire = b'{ "error": { "message": "native upstream failure", "code": "rate_limit" } }'
        elif request.url.path.endswith("/chat/completions"):
            if streaming:
                wire = (
                    b"data: "
                    + json.dumps(
                        {"object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": text}}]}
                    ).encode()
                    + b"\n\ndata: [DONE]\n\n"
                )
            else:
                wire = json.dumps({"choices": [{"message": {"role": "assistant", "content": text}}]}).encode()
        elif streaming:
            split = max(1, len(text) // 2)
            wire = b"".join(
                [
                    event_bytes("response.created", response={"output": []}, sequence_number=0),
                    event_bytes(
                        "response.output_text.delta",
                        delta=text[:split],
                        item_id="msg_test",
                        output_index=1,
                        content_index=0,
                        sequence_number=1,
                    ),
                    event_bytes(
                        "response.output_text.delta",
                        delta=text[split:],
                        item_id="msg_test",
                        output_index=1,
                        content_index=0,
                        sequence_number=2,
                    ),
                    event_bytes("response.completed", response=response_payload(text), sequence_number=3),
                ]
            )
        else:
            wire = json.dumps(response_payload(text), indent=2).encode()
        body = ResponseBody(wire)
        bodies.append(body)
        return httpx.Response(
            settings["status"],
            headers={
                "content-type": "text/event-stream" if streaming else "application/json",
                "x-request-id": "synthetic",
            },
            stream=body,
        )

    provider = create_openai_integration(
        instance_id="test",
        base_url="https://provider.test/v1",
        mount_path="/openai",
        operations=getattr(request, "param", ("chat_completions", "responses")),
    )
    app = create_proxy_application(
        providers=(provider,),
        rails_configs=(ConfigEntry("regex", config),),
        http_transport_factory=lambda: httpx.MockTransport(handler),
    )
    with TestClient(app) as client:
        yield client, requests, bodies, settings
    assert all(body.closed for body in bodies)


def test_factories_keep_chat_only_and_multi_operation_policies_distinct():
    chat = create_openai_chat_integration(base_url="https://provider.test/v1")
    both = create_openai_integration(base_url="https://provider.test/v1")
    assert chat.endpoints == (CHAT_COMPLETIONS_ENDPOINT,)
    assert both.endpoints == (CHAT_COMPLETIONS_ENDPOINT, RESPONSES_ENDPOINT)
    assert chat.upstream == both.upstream
    assert chat.errors is both.errors


@pytest.mark.parametrize(
    ("operations", "endpoints"),
    [
        (("chat_completions",), (CHAT_COMPLETIONS_ENDPOINT,)),
        (("responses",), (RESPONSES_ENDPOINT,)),
        (("chat_completions", "responses"), (CHAT_COMPLETIONS_ENDPOINT, RESPONSES_ENDPOINT)),
        (["responses", "chat_completions"], (RESPONSES_ENDPOINT, CHAT_COMPLETIONS_ENDPOINT)),
    ],
)
def test_factory_selects_exact_bindings_in_declared_order(operations, endpoints):
    integration = create_openai_integration(
        operations=operations, instance_id="selected", base_url="https://provider.test/v1", mount_path="/selected"
    )
    assert integration.endpoints == endpoints
    assert integration.instance_id == "selected"
    assert integration.upstream.base_url == "https://provider.test/v1"
    assert integration.upstream.mount_path == "/selected"


@pytest.mark.parametrize(
    "operations",
    [(), [], ("unknown",), ("responses", "unknown"), ("responses", "responses"), (None,), (["responses"],)],
)
def test_factory_rejects_invalid_operation_names_and_selections(operations):
    with pytest.raises(ValueError):
        create_openai_integration(operations=operations)


@pytest.mark.parametrize("operations", ["responses", b"responses", None, {"responses"}, {"responses": True}])
def test_factory_requires_a_sequence_not_a_bare_name_or_mapping(operations):
    with pytest.raises(TypeError, match="operations must be a sequence"):
        create_openai_integration(operations=operations)


def test_factory_snapshots_the_operation_selection():
    operations = ["responses"]
    integration = create_openai_integration(operations=operations)
    operations.append("chat_completions")
    assert integration.endpoints == (RESPONSES_ENDPOINT,)


def test_chat_wrapper_preserves_settings_and_explicit_dialect(monkeypatch):
    monkeypatch.setenv("NEMO_GUARDRAILS_OPENAI_PROXY_UPSTREAM_BASE_URL", "https://gateway.test/v1")
    chat = create_openai_chat_integration(instance_id="chat", mount_path="/chat")
    selected = create_openai_integration(instance_id="chat", mount_path="/chat", operations=("chat_completions",))
    dialect = create_openai_chat_integration(
        instance_id="chat", mount_path="/chat", endpoint=LITELLM_CHAT_COMPLETIONS_ENDPOINT
    )
    assert chat == selected
    assert chat.upstream.base_url == "https://gateway.test/v1"
    assert dialect.endpoints == (LITELLM_CHAT_COMPLETIONS_ENDPOINT,)
    assert dialect.upstream == chat.upstream
    assert dialect.errors is chat.errors
    assert create_openai_integration().endpoints == (CHAT_COMPLETIONS_ENDPOINT, RESPONSES_ENDPOINT)


@pytest.mark.parametrize(
    ("proxy", "selected"),
    [(("responses",), "responses"), (("chat_completions",), "chat/completions")],
    indirect=["proxy"],
)
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("operation", ["responses", "chat/completions"])
@pytest.mark.parametrize("blocked_stage", ["input", "output"])
def test_operation_selection_guards_selected_routes_and_preserves_other_passthrough(
    proxy, selected, streaming, operation, blocked_stage
):
    client, requests, bodies, settings = proxy
    guarded = operation == selected
    text = "BLOCK_INPUT" if blocked_stage == "input" else "question"
    if blocked_stage == "output":
        settings["output"] = "BLOCK_OUTPUT"
    subject = {"input": text} if operation == "responses" else {"messages": [{"role": "user", "content": text}]}
    response = client.post("/openai/v1/" + operation, json={"stream": streaming, **subject})
    if guarded:
        assert response.status_code == (200 if streaming and blocked_stage == "output" else 400)
        assert b"content_blocked" in response.content
        assert b"BLOCK_OUTPUT" not in response.content
        assert len(requests) == (0 if blocked_stage == "input" else 1)
    else:
        assert response.status_code == 200
        assert len(requests) == 1
        assert response.content == bodies[0].body
        if blocked_stage == "output":
            assert b"BLOCK_OUTPUT" in response.content


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("operation", ["responses", "chat/completions"])
def test_both_operations_work_through_the_existing_application(proxy, streaming, operation):
    client, requests, bodies, _ = proxy
    subject = (
        {"input": "question"} if operation == "responses" else {"messages": [{"role": "user", "content": "question"}]}
    )
    body = json.dumps({"model": "test-model", "stream": streaming, **subject}, indent=2).encode()
    response = client.post(
        "/openai/v1/" + operation,
        content=body,
        headers={"content-type": "application/json", "authorization": "Bearer synthetic"},
    )
    assert response.status_code == 200
    assert response.content == bodies[0].body
    assert str(requests[0].url) == "https://provider.test/v1/" + operation
    assert requests[0].content == body
    assert requests[0].headers["authorization"] == "Bearer synthetic"


@pytest.mark.parametrize("streaming", [False, True])
def test_real_input_rail_blocks_before_dispatch(proxy, streaming):
    client, requests, _, _ = proxy
    response = client.post("/openai/v1/responses", json={"input": "BLOCK_INPUT", "stream": streaming})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "content_blocked"
    assert not requests


@pytest.mark.parametrize("streaming", [False, True])
def test_real_output_rail_withholds_blocked_response_and_stream_snapshots(proxy, streaming):
    client, _, bodies, settings = proxy
    settings["output"] = "BLOCK_OUTPUT"
    response = client.post("/openai/v1/responses", json={"input": "question", "stream": streaming})
    assert response.status_code == (200 if streaming else 400)
    assert b"content_blocked" in response.content
    assert b"BLOCK_OUTPUT" not in response.content
    if streaming:
        assert b"response.output_text.delta" not in response.content
        assert b"event: error" in response.content
    assert bodies[0].closed


@pytest.mark.parametrize("streaming", [False, True])
def test_provider_failure_bytes_and_headers_are_not_rewritten(proxy, streaming):
    client, _, bodies, settings = proxy
    settings["status"] = 429
    response = client.post("/openai/v1/responses", json={"input": "question", "stream": streaming})
    assert response.status_code == 429
    assert response.content == bodies[0].body
    assert response.headers["x-request-id"] == "synthetic"


@pytest.mark.parametrize("path", ["/openai/v2/responses", "/openai/v1/responses/"])
def test_noncanonical_responses_routes_cannot_bypass_inspection(proxy, path):
    client, requests, _, _ = proxy
    response = client.post(path, json={"input": "BLOCK_INPUT"}, follow_redirects=False)
    assert response.status_code in {400, 404, 405, 422}
    assert not requests


def test_application_openapi_documents_both_operations_without_loading_export(proxy):
    client, _, _, _ = proxy
    paths = client.get("/openapi.json").json()["paths"]
    assert "/openai/v1/responses" in paths
    assert "/openai/v1/chat/completions" in paths


@pytest.mark.parametrize("blocked", [False, True])
def test_openai_sdk_decodes_native_success_and_proxy_error_events(proxy, blocked):
    from openai import OpenAI

    client, _, _, settings = proxy
    if blocked:
        settings["output"] = "BLOCK_OUTPUT"

    def forward(request):
        response = client.post(
            "/openai/v1/responses", content=request.content, headers={"content-type": "application/json"}
        )
        return httpx.Response(response.status_code, headers=response.headers, content=response.content)

    with OpenAI(
        api_key="synthetic",
        base_url="https://sdk.test/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(forward)),
    ) as sdk:
        events = list(sdk.responses.create(model="test-model", input="question", stream=True))
    if blocked:
        assert events[-1].type == "error"
        assert events[-1].code == "content_blocked"
        assert not any(item.type == "response.output_text.delta" for item in events)
    else:
        assert events[-1].type == "response.completed"
        assert "".join(item.delta for item in events if item.type == "response.output_text.delta") == "answer"
