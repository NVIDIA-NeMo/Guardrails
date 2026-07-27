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
* IORails rail engines (``model_engine`` / ``api_engine``) use **aiohttp**, so
  those cases use ``aioresponses``.

The matrix mirrors the manual smoke harness (``smoke_client.py`` +
``fake_upstream.py``): upstream status passthrough, non-error upstream status,
rail-engine failures, protocol-level responses, and streaming frames.
"""

import json

import pytest

pytest.importorskip("openai", reason="openai is required for server tests")
from aioresponses import aioresponses
from fastapi.testclient import TestClient
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

# The rail endpoint deliberately looks like an internal cluster address: the
# sanitization assertions below depend on it appearing in the failure reason.
RAIL_ENDPOINT = "http://jailbreak.internal.svc.cluster.local:8000/v1/classify"

RAIL_CONFIG = """
models:
  - type: main
    engine: openai
    model: gpt-4o-mini
    parameters:
      base_url: http://upstream.invalid/v1
      api_key: sk-dummy
rails:
  config:
    jailbreak_detection:
      nim_base_url: "http://jailbreak.internal.svc.cluster.local:8000/v1"
      nim_server_endpoint: "classify"
  input:
    flows:
      - jailbreak detection model
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
            yield
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
    """Failures raised by an IORails rail engine, which talks aiohttp, not httpx."""

    def test_non_error_upstream_status_is_clamped_to_500(self, serve_config):
        """A 2xx response with a non-JSON body must not become a 2xx error envelope.

        ``api_engine`` raises ``APIEngineError(status=exc.status)`` on a
        ``ContentTypeError``, and that status is 200 because the >=400 case was
        already handled. Returning it verbatim would make an OpenAI-compatible
        client parse the error body as a ChatCompletion.
        """
        serve_config(RAIL_CONFIG, iorails=True)

        with aioresponses() as mocked:
            mocked.post(RAIL_ENDPOINT, status=200, body="not json", content_type="text/plain", repeat=True)
            response = _chat()

        assert response.status_code == 500
        assert response.json()["error"]["type"] == "server_error"

    def test_rail_upstream_429_is_forwarded(self, serve_config):
        serve_config(RAIL_CONFIG, iorails=True)

        with aioresponses() as mocked:
            mocked.post(RAIL_ENDPOINT, status=429, body="slow down", repeat=True)
            response = _chat()

        assert response.status_code == 429
        assert response.json()["error"]["type"] == "rate_limit_error"

    def test_status_less_failure_blocks_without_leaking_the_endpoint(self, serve_config):
        """A rail that cannot connect blocks the request, and the reason reaches the client.

        The reason names the rail's endpoint, so it must be sanitized before it
        is streamed out.
        """
        serve_config(RAIL_CONFIG, iorails=True)

        with aioresponses() as mocked:
            mocked.post(RAIL_ENDPOINT, exception=OSError("connection refused"), repeat=True)
            response = _chat(stream=True)

        assert response.status_code == 200
        payload = _sse_payloads(response)[0]
        assert payload["error"]["type"] == "guardrails_violation"
        assert payload["error"]["param"] == "input_rails"
        assert "internal.svc.cluster.local" not in payload["error"]["message"]
        assert "[redacted-url]" in payload["error"]["message"]


class TestProtocolLevelResponses:
    """Responses produced before any rail runs."""

    def test_method_not_allowed_keeps_the_allow_header(self):
        """RFC 9110 requires ``Allow`` on a 405; replacing FastAPI's handler must not drop it."""
        response = _client().get("/v1/chat/completions")

        assert response.status_code == 405
        assert "POST" in response.headers["allow"]
        assert response.json()["error"]["message"] == "Method Not Allowed"

    def test_validation_error_does_not_echo_the_request_body(self, serve_config):
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
    """The SSE frames a streaming client actually receives.

    A streaming response is already committed at HTTP 200, so the status has to
    travel in the frame; these assert on the emitted ``data:`` lines rather than
    on a payload the test built itself.
    """

    def test_downstream_failure_emits_one_error_frame(self, httpx_mock: HTTPXMock, serve_config):
        serve_config(MAIN_MODEL_CONFIG)
        httpx_mock.add_response(
            url=MAIN_MODEL_URL,
            method="POST",
            status_code=503,
            json={"error": {"message": "overloaded", "type": "server_error"}},
            is_reusable=True,
        )

        response = _chat(stream=True)

        assert response.status_code == 200
        payloads = _sse_payloads(response)
        assert len(payloads) == 1
        error = payloads[0]["error"]
        assert error["type"] == "downstream_error"
        # The HTTP status has no status line to travel on, so it rides in `code`.
        assert error["code"] == 503
        assert error["message"] == "overloaded"

    def test_non_error_upstream_status_is_clamped_in_error_frame(self, serve_config):
        serve_config(RAIL_CONFIG, iorails=True)

        with aioresponses() as mocked:
            mocked.post(RAIL_ENDPOINT, status=200, body="not json", content_type="text/plain", repeat=True)
            response = _chat(stream=True)

        assert response.status_code == 200
        error = _sse_payloads(response)[0]["error"]
        assert error["type"] == "downstream_error"
        assert error["code"] == 500

    def test_error_frame_does_not_disclose_model_provider_or_endpoint(self, httpx_mock: HTTPXMock, serve_config):
        serve_config(MAIN_MODEL_CONFIG)
        httpx_mock.add_response(
            url=MAIN_MODEL_URL,
            method="POST",
            status_code=401,
            json={"error": {"message": "bad key", "type": "authentication_error"}},
            is_reusable=True,
        )

        message = _sse_payloads(_chat(stream=True))[0]["error"]["message"]

        assert "provider=" not in message
        assert "upstream.invalid" not in message
