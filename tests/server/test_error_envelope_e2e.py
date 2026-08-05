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

"""End-to-end coverage for the OpenAI-compatible HTTP error envelope.

Unlike ``tests/test_http_error_handling.py``, which asserts handler-level
mappings against injected exceptions, every test here drives a real request
through the ASGI app against a real ``RailsConfig`` and lets the exception
originate at the transport boundary. That is the only way to catch the chains
that break in production: an upstream status flowing through the client layer
into the exception, ``APIEngineError`` carrying a non-error status, and the SSE
frames a streaming client actually receives.

Two transports are mocked because the stack uses two:

* the main model goes through the OpenAI-compatible client over **httpx**, so
  those cases use ``httpx_mock`` (with ``testserver`` excluded so ``TestClient``
  still reaches the app);
* an IORails rail reaches its model through ``ModelEngine``, which uses
  **aiohttp**, so those cases use ``aioresponses``.

The matrix mirrors the manual smoke harness (``smoke_client.py`` +
``fake_upstream.py``): upstream status passthrough, rail failures, protocol-level
responses, and streaming frames.
"""

import json

import pytest
from aioresponses import aioresponses

pytest.importorskip("openai", reason="openai is required for server tests")
from fastapi.testclient import TestClient
from openai import InternalServerError, OpenAI
from pytest_httpx import HTTPXMock

from nemoguardrails import Guardrails, RailsConfig
from nemoguardrails.server import api

MAIN_MODEL_URL = "http://upstream.invalid/v1/chat/completions"

MAIN_MODEL_CONFIG = """
models:
  - type: main
    engine: openai
    model: gpt-4o-mini
    parameters:
      base_url: http://upstream.invalid/v1
      api_key: sk-dummy
      max_retries: 0
"""

# A rail that calls the main model, so an upstream failure surfaces through
# /v1/checks. The jailbreak rail cannot be used here: it runs on an IORails
# engine, and /v1/checks requires Colang 1.0 via LLMRails.
CONTENT_SAFETY_CONFIG = """
models:
  - type: main
    engine: openai
    model: gpt-4o-mini
    parameters:
      base_url: http://upstream.invalid/v1
      api_key: sk-dummy
      max_retries: 0
  - type: content_safety
    engine: openai
    model: gpt-4o-mini
    parameters:
      base_url: http://upstream.invalid/v1
      api_key: sk-dummy
      max_retries: 0
rails:
  input:
    flows:
      - content safety check input $model=content_safety
prompts:
  - task: content_safety_check_input $model=content_safety
    content: |
      Is the following user message safe? Answer "safe" or "unsafe".
      User message: {{ user_input }}
    output_parser: is_content_safe
"""

# Output rails configured but streaming for them left disabled, which makes a
# streaming request unsatisfiable.
OUTPUT_RAILS_NO_STREAMING_CONFIG = """
models:
  - type: main
    engine: openai
    model: gpt-4o-mini
    parameters:
      base_url: http://upstream.invalid/v1
      api_key: sk-dummy
      max_retries: 0
  - type: content_safety
    engine: openai
    model: gpt-4o-mini
    parameters:
      base_url: http://upstream.invalid/v1
      api_key: sk-dummy
      max_retries: 0
rails:
  output:
    streaming:
      enabled: false
    flows:
      - content safety check output $model=content_safety
prompts:
  - task: content_safety_check_output $model=content_safety
    content: |
      Is the following bot message safe? Answer "safe" or "unsafe".
      Bot message: {{ bot_response }}
    output_parser: is_content_safe
"""


@pytest.fixture
def non_mocked_hosts():
    """Let TestClient reach the app; httpx_mock only intercepts the upstream provider."""
    return ["testserver"]


_active_client = None


@pytest.fixture(autouse=True)
def reset_server_state():
    """Clear the per-config rails cache and force multi-config mode around each test."""
    global _active_client
    original_single_config_mode = api.app.single_config_mode
    api.app.single_config_mode = False
    api.llm_rails_instances.clear()
    api.llm_rails_events_history_cache.clear()
    try:
        with TestClient(api.app, raise_server_exceptions=False) as client:
            _active_client = client
            try:
                yield
            finally:
                assert client.portal is not None
                for rails in api.llm_rails_instances.values():
                    if isinstance(rails, Guardrails):
                        client.portal.call(rails.shutdown)
    finally:
        _active_client = None
        api.llm_rails_instances.clear()
        api.llm_rails_events_history_cache.clear()
        api.app.single_config_mode = original_single_config_mode


@pytest.fixture
def serve_config(monkeypatch):
    """Serve a config built from YAML for any requested config_id."""

    def _serve(yaml_content: str, *, iorails: bool = False):
        config = RailsConfig.from_content(yaml_content=yaml_content)
        monkeypatch.setattr(api.RailsConfig, "from_path", staticmethod(lambda full_path: config))
        if iorails:
            # Mirrors NEMO_GUARDRAILS_IORAILS_ENGINE, the same aliasing used by
            # tests/server/test_iorails_engine_compat.py. Rail engines only run
            # on the IORails path.
            monkeypatch.setattr(api, "LLMRails", Guardrails)
        return config

    return _serve


def _client() -> TestClient:
    assert _active_client is not None
    return _active_client


def _chat(stream: bool = False, **body):
    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": stream,
        "guardrails": {"config_id": "test"},
        **body,
    }
    return _client().post("/v1/chat/completions", json=payload)


def _sse_payloads(response) -> list[dict]:
    """Parse the JSON objects out of an SSE response body."""
    payloads = []
    for line in response.text.splitlines():
        if not line.startswith("data: "):
            continue
        data = line[len("data: ") :].strip()
        if data and data != "[DONE]":
            payloads.append(json.loads(data))
    return payloads


class TestUpstreamStatusPassthrough:
    """An upstream provider status reaches the caller as our status and envelope.

    Exercises the full chain: httpx transport -> raise_for_status ->
    LLMClientError -> LLMCallException(status=...) -> llm_call_exception_handler.
    """

    @pytest.mark.parametrize(
        "upstream_status,expected_type",
        [
            (400, "invalid_request_error"),
            (401, "authentication_error"),
            (403, "permission_error"),
            (429, "rate_limit_error"),
            (500, "server_error"),
            (503, "server_error"),
        ],
    )
    def test_status_and_type(self, httpx_mock: HTTPXMock, serve_config, upstream_status, expected_type):
        serve_config(MAIN_MODEL_CONFIG)
        httpx_mock.add_response(
            url=MAIN_MODEL_URL,
            method="POST",
            status_code=upstream_status,
            json={"error": {"message": f"upstream returned {upstream_status}", "type": expected_type}},
            is_reusable=True,
        )

        response = _chat()

        assert response.status_code == upstream_status
        error = response.json()["error"]
        assert error["type"] == expected_type
        assert error["message"] == f"upstream returned {upstream_status}"

    def test_rate_limit_forwards_code_and_retry_after(self, httpx_mock: HTTPXMock, serve_config):
        """A 429 carries the provider's code and a Retry-After header.

        Without these an SDK's backoff is blind even though the provider
        supplied the value.
        """
        serve_config(MAIN_MODEL_CONFIG)
        httpx_mock.add_response(
            url=MAIN_MODEL_URL,
            method="POST",
            status_code=429,
            json={
                "error": {
                    "message": "slow down",
                    "type": "rate_limit_error",
                    "code": "rate_limit_exceeded",
                    "param": "messages",
                }
            },
            headers={"retry-after": "7"},
            is_reusable=True,
        )

        response = _chat()

        assert response.status_code == 429
        assert response.headers["retry-after"] == "7"
        assert response.json()["error"]["code"] == "rate_limit_exceeded"
        assert response.json()["error"]["param"] == "messages"

    def test_message_does_not_disclose_model_or_provider(self, httpx_mock: HTTPXMock, serve_config):
        """``str(LLMClientError)`` prefixes the internal model, provider, and endpoint.

        The client sees the provider's own message only.
        """
        serve_config(MAIN_MODEL_CONFIG)
        httpx_mock.add_response(
            url=MAIN_MODEL_URL,
            method="POST",
            status_code=401,
            json={"error": {"message": "bad key", "type": "authentication_error"}},
            is_reusable=True,
        )

        message = _chat().json()["error"]["message"]

        assert message == "bad key"
        assert "provider=" not in message
        assert "endpoint=" not in message
        assert "upstream.invalid" not in message


class TestRailEngineErrors:
    """Failures raised by an IORails rail, forwarded rather than reported as a verdict."""

    def test_rail_upstream_429_is_forwarded(self, serve_config):
        """A provider rate limit hit by a rail reaches the client as a 429, not a block.

        A rail that blocks and a provider that is rate-limiting mean very different things to
        a caller, and only one is worth retrying. ``aioresponses`` rather than ``httpx_mock``
        because the rail reaches its model through ``ModelEngine``, which talks aiohttp.
        """
        serve_config(CONTENT_SAFETY_CONFIG, iorails=True)

        with aioresponses() as mocked:
            mocked.post(MAIN_MODEL_URL, status=429, body="slow down", repeat=True)
            response = _chat()

        assert response.status_code == 429
        assert response.json()["error"]["type"] == "rate_limit_error"


class TestProtocolLevelResponses:
    """Responses produced before any rail runs."""

    def test_method_not_allowed_keeps_the_allow_header(self):
        """RFC 9110 requires ``Allow`` on a 405; replacing FastAPI's handler must not drop it."""
        response = _client().get("/v1/chat/completions")

        assert response.status_code == 405
        assert "POST" in response.headers["allow"]
        assert response.json()["error"]["message"] == "Method Not Allowed"

    def test_validation_error_does_not_echo_the_request_body(self, serve_config, caplog):
        """``str(RequestValidationError)`` embeds the raw body; the envelope must not.

        The body here carries a credential-shaped value and PII that would be
        disclosed to the client and written to the server log.
        """
        serve_config(MAIN_MODEL_CONFIG)

        response = _client().post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}], "user_token": "AKIAsecret", "ssn": "123-45-6789"},
        )

        assert response.status_code == 422
        error = response.json()["error"]
        assert error["type"] == "invalid_request_error"
        assert "AKIAsecret" not in error["message"]
        assert "123-45-6789" not in error["message"]
        # The failing field is still identified, so a client can act on it.
        assert "model" in error["message"]
        assert error["param"] == "model"
        validation_logs = [
            record.getMessage()
            for record in caplog.records
            if record.name == "nemoguardrails.server.exception_handlers"
            and record.getMessage().startswith("Request validation failed:")
        ]
        assert validation_logs
        assert "model" in validation_logs[0]
        assert "AKIAsecret" not in caplog.text
        assert "123-45-6789" not in caplog.text

    def test_unknown_config_id_is_a_client_error(self, monkeypatch):
        """A config that cannot be loaded is the caller's mistake, not a server fault."""

        def _raise(full_path):
            raise ValueError("no such config")

        monkeypatch.setattr(api.RailsConfig, "from_path", staticmethod(_raise))

        response = _chat()

        assert response.status_code == 400
        assert response.json()["error"]["type"] == "invalid_request_error"


class TestGuardrailCheckEndpoint:
    """``/v1/checks`` must reach the same handlers as ``/v1/chat/completions``.

    It used to keep a local ``except Exception -> HTTPException(500)`` that
    shadowed them, so an identical upstream failure reported differently on the
    two endpoints.
    """

    def test_upstream_rate_limit_is_forwarded(self, httpx_mock: HTTPXMock, serve_config):
        serve_config(CONTENT_SAFETY_CONFIG)
        httpx_mock.add_response(
            url=MAIN_MODEL_URL,
            method="POST",
            status_code=429,
            json={"error": {"message": "slow down", "type": "rate_limit_error"}},
            is_reusable=True,
        )

        response = _client().post(
            "/v1/checks",
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "hi"}],
                "guardrails": {"config_id": "test"},
            },
        )

        assert response.status_code == 429
        assert response.json()["error"]["type"] == "rate_limit_error"


class TestUnsupportedRequestCombinations:
    """Request/config combinations the caller can correct are 400, not 500.

    A 500 both hides the actionable message and invites an SDK retry of a
    request that can never succeed.
    """

    def test_streaming_without_streaming_output_rails_is_400(self, serve_config):
        serve_config(OUTPUT_RAILS_NO_STREAMING_CONFIG)

        response = _chat(stream=True)

        assert response.status_code == 400
        error = response.json()["error"]
        assert error["type"] == "invalid_request_error"
        # The actionable part of the message survives instead of being replaced
        # by "Internal server error".
        assert "streaming" in error["message"].lower()


class TestStreamingErrorFrames:
    def test_initial_downstream_failure_preserves_http_status(self, httpx_mock: HTTPXMock, serve_config):
        serve_config(MAIN_MODEL_CONFIG)
        httpx_mock.add_response(
            url=MAIN_MODEL_URL,
            method="POST",
            status_code=503,
            json={"error": {"message": "overloaded", "type": "server_error"}},
            is_reusable=True,
        )

        response = _chat(stream=True)

        assert response.status_code == 503
        assert response.json()["error"]["type"] == "server_error"
        assert response.json()["error"]["message"] == "overloaded"
        assert response.json()["error"]["code"] == 503

    def test_openai_client_retries_initial_streaming_failure(self, httpx_mock: HTTPXMock, serve_config):
        serve_config(MAIN_MODEL_CONFIG)
        httpx_mock.add_response(
            url=MAIN_MODEL_URL,
            method="POST",
            status_code=503,
            json={"error": {"message": "overloaded", "type": "server_error"}},
            is_reusable=True,
        )
        client = OpenAI(
            api_key="test-key",
            base_url="http://testserver/v1",
            http_client=_client(),
            max_retries=1,
        )

        with pytest.raises(InternalServerError) as exc_info:
            client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "hello"}],
                stream=True,
                extra_body={"guardrails": {"config_id": "test"}},
            )

        assert exc_info.value.status_code == 503
        assert exc_info.value.body["code"] == 503
        assert len(httpx_mock.get_requests(url=MAIN_MODEL_URL)) == 2

    def test_initial_error_does_not_disclose_model_provider_or_endpoint(self, httpx_mock: HTTPXMock, serve_config):
        serve_config(MAIN_MODEL_CONFIG)
        httpx_mock.add_response(
            url=MAIN_MODEL_URL,
            method="POST",
            status_code=401,
            json={"error": {"message": "bad key", "type": "authentication_error"}},
            is_reusable=True,
        )

        message = _chat(stream=True).json()["error"]["message"]

        assert "provider=" not in message
        assert "upstream.invalid" not in message
