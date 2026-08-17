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

"""End-to-end coverage for the request ``model`` field, through the ASGI app.

Every assertion is on the request that left the server rather than on the RailsConfig the
server built, because that is the only place this defect is visible: a request carrying
``model`` used to rebuild the configured main model from environment variables alone, and
the result was a call to the wrong provider, or one with no ``Authorization`` header.

``ENGINE_ONLY_CONFIG`` deliberately declares no ``base_url``. With one configured, the
engine-derived endpoint never applies and the wrong engine reaches the right URL anyway,
which is why the credential half of this defect was caught long before the engine half.
"""

import pytest

pytest.importorskip("openai", reason="openai is required for server tests")
from fastapi.testclient import TestClient
from pytest_httpx import HTTPXMock

from nemoguardrails import RailsConfig
from nemoguardrails.server import api

API_KEY_ENV_VAR = "MODEL_INJECTION_TEST_KEY"
API_KEY = "test-key"

REQUEST_MODEL = "meta/llama-3.1-70b-instruct"

NIM_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
RAIL_MODEL_URL = "http://rail.invalid/v1/chat/completions"

ENGINE_ONLY_CONFIG = """
models:
  - type: main
    engine: nim
    model: meta/llama-3.1-70b-instruct
    api_key_env_var: MODEL_INJECTION_TEST_KEY
    parameters:
      max_retries: 0
      default_headers:
        X-Tenant-Id: acme-corp
"""

MAIN_AND_RAIL_CONFIG = """
models:
  - type: main
    engine: nim
    model: meta/llama-3.1-70b-instruct
    api_key_env_var: MODEL_INJECTION_TEST_KEY
    parameters:
      max_retries: 0
  - type: content_safety
    engine: nim
    model: nvidia/llama-3.1-nemoguard-8b-content-safety
    api_key_env_var: MODEL_INJECTION_TEST_KEY
    parameters:
      base_url: http://rail.invalid/v1
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


@pytest.fixture
def non_mocked_hosts():
    """Let TestClient reach the app; httpx_mock only intercepts the upstream provider."""
    return ["testserver"]


@pytest.fixture(autouse=True)
def model_injection_env(monkeypatch):
    """Pin the credential and clear the env overrides so the config alone decides routing."""
    monkeypatch.setenv(API_KEY_ENV_VAR, API_KEY)
    monkeypatch.delenv("MAIN_MODEL_ENGINE", raising=False)
    monkeypatch.delenv("MAIN_MODEL_BASE_URL", raising=False)


@pytest.fixture
def client():
    """Serve the app with a clean rails cache, in multi-config mode."""
    original_single_config_mode = api.app.single_config_mode
    api.app.single_config_mode = False
    api.llm_rails_instances.clear()
    api.llm_rails_events_history_cache.clear()
    try:
        with TestClient(api.app, raise_server_exceptions=False) as test_client:
            yield test_client
    finally:
        api.llm_rails_instances.clear()
        api.llm_rails_events_history_cache.clear()
        api.app.single_config_mode = original_single_config_mode


@pytest.fixture
def serve_config(monkeypatch):
    """Serve a config built from YAML for any requested config_id."""

    def _serve(yaml_content: str):
        config = RailsConfig.from_content(yaml_content=yaml_content)
        monkeypatch.setattr(api.RailsConfig, "from_path", staticmethod(lambda full_path: config))
        return config

    return _serve


def _completion(content: str) -> dict:
    """Build a minimal OpenAI-compatible chat completion body."""
    return {
        "id": "chatcmpl-model-injection",
        "object": "chat.completion",
        "created": 1700000000,
        "model": REQUEST_MODEL,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def _chat(client: TestClient):
    """POST a chat completion that carries the request model, the way an OpenAI client does."""
    return client.post(
        "/v1/chat/completions",
        json={
            "model": REQUEST_MODEL,
            "messages": [{"role": "user", "content": "hi"}],
            "guardrails": {"config_id": "test"},
        },
    )


@pytest.fixture
def main_model_request(httpx_mock: HTTPXMock, serve_config, client):
    """Return the upstream request the server made for a config with no base_url."""
    serve_config(ENGINE_ONLY_CONFIG)
    httpx_mock.add_response(url=NIM_URL, method="POST", json=_completion("hello"))

    assert _chat(client).status_code == 200

    return httpx_mock.get_request()


def test_request_model_routes_to_the_configured_engine_endpoint(main_model_request):
    """Test the outbound call goes to the configured engine's endpoint, not the openai default."""
    assert str(main_model_request.url) == NIM_URL


def test_request_model_keeps_the_configured_credential(main_model_request):
    """Test the outbound call carries the key named by the configured api_key_env_var."""
    assert main_model_request.headers["Authorization"] == f"Bearer {API_KEY}"


def test_request_model_keeps_the_configured_default_headers(main_model_request):
    """Test the outbound call still carries the default headers from the configured parameters."""
    assert main_model_request.headers["X-Tenant-Id"] == "acme-corp"


def test_main_and_rail_models_both_carry_the_configured_credential(httpx_mock: HTTPXMock, serve_config, client):
    """Test a main model and a rail model configured alike are both authenticated."""
    serve_config(MAIN_AND_RAIL_CONFIG)
    httpx_mock.add_response(url=RAIL_MODEL_URL, method="POST", json=_completion("safe"))
    httpx_mock.add_response(url=NIM_URL, method="POST", json=_completion("hello"))

    assert _chat(client).status_code == 200

    authorization_by_host = {
        request.url.host: request.headers.get("Authorization") for request in httpx_mock.get_requests()
    }
    assert authorization_by_host == {
        "integrate.api.nvidia.com": f"Bearer {API_KEY}",
        "rail.invalid": f"Bearer {API_KEY}",
    }


def test_env_engine_redirects_the_main_model_but_keeps_its_credential(
    httpx_mock: HTTPXMock, serve_config, client, monkeypatch
):
    """Test MAIN_MODEL_ENGINE reroutes the main model without changing credential resolution."""
    monkeypatch.setenv("MAIN_MODEL_ENGINE", "openai")
    serve_config(ENGINE_ONLY_CONFIG)
    httpx_mock.add_response(url=OPENAI_URL, method="POST", json=_completion("hello"))

    assert _chat(client).status_code == 200

    request = httpx_mock.get_request()
    assert str(request.url) == OPENAI_URL
    assert request.headers["Authorization"] == f"Bearer {API_KEY}"
