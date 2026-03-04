# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import json
import os
import threading
import time
from typing import Dict, List

import pytest
from aiohttp import web

pytest.importorskip("openai", reason="openai is required for server tests")
pytest.importorskip("langchain_openai", reason="langchain-openai is required for server tests")
from fastapi.testclient import TestClient

from nemoguardrails.server import api

client = TestClient(api.app)

# Global storage for captured API keys from mock server
captured_requests: List[Dict] = []
mock_server_port = 8765
mock_server_url = f"http://localhost:{mock_server_port}"


@pytest.fixture(scope="function", autouse=True)
def set_rails_config_path():
    """Setup test environment with custom config path."""
    original_path = api.app.rails_config_path
    original_engine = os.environ.get("MAIN_MODEL_ENGINE")
    original_base_url = os.environ.get("MAIN_MODEL_BASE_URL")

    api.app.rails_config_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "test_configs"))
    os.environ["MAIN_MODEL_ENGINE"] = "openai"
    os.environ["MAIN_MODEL_BASE_URL"] = f"{mock_server_url}/v1"

    api.llm_rails_instances.clear()
    captured_requests.clear()

    yield

    api.app.rails_config_path = original_path
    api.llm_rails_instances.clear()

    if original_engine is not None:
        os.environ["MAIN_MODEL_ENGINE"] = original_engine
    else:
        os.environ.pop("MAIN_MODEL_ENGINE", None)

    if original_base_url is not None:
        os.environ["MAIN_MODEL_BASE_URL"] = original_base_url
    else:
        os.environ.pop("MAIN_MODEL_BASE_URL", None)


async def mock_openai_chat_handler(request):
    """Mock OpenAI /v1/chat/completions endpoint that captures headers."""
    # Capture the request details
    request_data = {
        "timestamp": time.time(),
        "headers": dict(request.headers),
        "authorization": request.headers.get("Authorization", None),
        "body": await request.json() if request.body_exists else {},
    }
    captured_requests.append(request_data)

    # Check if this is a streaming request
    body = request_data["body"]
    is_stream = body.get("stream", False)

    if is_stream:
        # Return streaming response
        async def stream_response():
            # Send a few chunks
            chunks = [
                {
                    "id": "chatcmpl-123",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": body.get("model", "gpt-3.5-turbo"),
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": "No "},
                            "finish_reason": None,
                        }
                    ],
                },
                {
                    "id": "chatcmpl-123",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": body.get("model", "gpt-3.5-turbo"),
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": ""},
                            "finish_reason": None,
                        }
                    ],
                },
                {
                    "id": "chatcmpl-123",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": body.get("model", "gpt-3.5-turbo"),
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop",
                        }
                    ],
                },
            ]

            for chunk in chunks:
                yield f"data: {json.dumps(chunk)}\n\n".encode()
            yield b"data: [DONE]\n\n"

        return web.Response(
            body=stream_response(),
            status=200,
            content_type="text/event-stream",
            headers={"Connection": "close"},
        )
    else:
        # Return non-streaming response
        response = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body.get("model", "gpt-3.5-turbo"),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "No",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }
        return web.json_response(response, headers={"Connection": "close"})


async def mock_openai_models_handler(request):
    """Mock OpenAI /v1/models endpoint."""
    return web.json_response(
        {
            "object": "list",
            "data": [
                {
                    "id": "gpt-3.5-turbo",
                    "object": "model",
                    "created": 1686935002,
                    "owned_by": "openai",
                }
            ],
        },
        headers={"Connection": "close"},
    )


async def create_mock_server():
    """Create and start the mock OpenAI server."""
    app = web.Application()
    app.router.add_post("/v1/chat/completions", mock_openai_chat_handler)
    app.router.add_get("/v1/models", mock_openai_models_handler)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "localhost", mock_server_port)
    await site.start()

    return runner


@pytest.fixture(scope="module")
def mock_openai_server():
    """Start mock OpenAI server for the test module."""
    loop = asyncio.new_event_loop()
    runner_ref = []

    def run_server():
        asyncio.set_event_loop(loop)
        runner = loop.run_until_complete(create_mock_server())
        runner_ref.append(runner)
        loop.run_forever()

    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()

    # Wait for server to start
    time.sleep(1.0)

    yield

    # Give time for any pending connections to close before fixture teardown
    time.sleep(1.0)


def test_header_api_key_injection_non_streaming(mock_openai_server):
    """Test that API key from header is correctly injected for non-streaming requests."""
    captured_requests.clear()

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Hello"}],
            "guardrails": {"config_id": "with_api_key_header"},
            "stream": False,
        },
        headers={"X-Gpt-3.5-Turbo-Authorization": "custom-api-key-123"},
    )

    assert response.status_code == 200

    # Verify that at least one request was made to the mock server
    assert len(captured_requests) > 0

    # Check that the custom API key was sent in the Authorization header
    auth_headers = [req["authorization"] for req in captured_requests if req["authorization"]]
    assert len(auth_headers) > 0, "No authorization headers found"
    assert any("custom-api-key-123" in auth for auth in auth_headers), f"custom-api-key-123 not found in {auth_headers}"


def test_header_api_key_injection_streaming(mock_openai_server):
    """Test that API key from header is correctly injected for streaming requests."""
    captured_requests.clear()

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Hello"}],
            "guardrails": {"config_id": "with_api_key_header"},
            "stream": True,
        },
        headers={"X-Gpt-3.5-Turbo-Authorization": "custom-stream-key-456"},
    )

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["Content-Type"]

    # Consume the stream
    chunks = list(response.iter_lines())
    assert len(chunks) > 0

    # Verify that at least one request was made to the mock server
    assert len(captured_requests) > 0

    # Check that the custom API key was sent
    auth_headers = [req["authorization"] for req in captured_requests if req["authorization"]]
    assert len(auth_headers) > 0
    assert any("custom-stream-key-456" in auth for auth in auth_headers)


def test_no_header_uses_default_api_key(mock_openai_server):
    """Test that default API key is used when no header is provided."""
    captured_requests.clear()

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Hello"}],
            "guardrails": {"config_id": "with_api_key_header"},
            "stream": False,
        },
    )

    assert response.status_code == 200

    # Verify request was made
    assert len(captured_requests) > 0

    # Check that the default API key was used
    auth_headers = [req["authorization"] for req in captured_requests if req["authorization"]]
    assert len(auth_headers) > 0
    assert any("default-api-key" in auth for auth in auth_headers)


def test_api_key_reset_after_non_streaming(mock_openai_server):
    """Test that API key is reset to original value after non-streaming generation.

    This test verifies that the same LLMRails instance properly resets API keys
    between requests, preventing leakage from one request to the next.
    """
    captured_requests.clear()

    # First request with custom header
    response1 = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "req1"}],
            "guardrails": {"config_id": "with_api_key_header"},
            "stream": False,
        },
        headers={"X-Gpt-3.5-Turbo-Authorization": "temp-key-1"},
    )
    assert response1.status_code == 200

    # DON'T clear cache - use the same LLMRails instance to verify reset works
    # DON'T clear captured_requests - we need both requests to verify isolation

    # Second request without header - should use default (tests that reset worked)
    response2 = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "req2"}],
            "guardrails": {"config_id": "with_api_key_header"},
            "stream": False,
        },
    )
    assert response2.status_code == 200

    # Match each request to its API key
    for req in captured_requests:
        content = req["body"]["messages"][-1]["content"]
        if content == "req1":
            assert "temp-key-1" in req["authorization"], f"Request 1 should use temp-key-1, got: {req['authorization']}"
        elif content == "req2":
            assert "default-api-key" in req["authorization"], (
                f"Request 2 should use default-api-key (reset failed!), got: {req['authorization']}"
            )


def test_api_key_reset_after_streaming(mock_openai_server):
    """Test that API key is reset to original value after streaming generation."""
    captured_requests.clear()

    # First request with custom header (streaming)
    response1 = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "stream1"}],
            "guardrails": {"config_id": "with_api_key_header"},
            "stream": True,
        },
        headers={"X-Gpt-3.5-Turbo-Authorization": "temp-stream-key"},
    )
    assert response1.status_code == 200
    list(response1.iter_lines())  # Consume the entire stream

    # DON'T clear cache - use the same LLMRails instance to verify reset works
    # DON'T clear captured_requests - we need both requests to verify isolation

    # Second request without header - should use default
    response2 = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "stream2"}],
            "guardrails": {"config_id": "with_api_key_header"},
            "stream": True,
        },
    )
    assert response2.status_code == 200
    list(response2.iter_lines())

    # Match each request to its API key
    for req in captured_requests:
        content = req["body"]["messages"][-1]["content"]
        if content == "stream1":
            assert "temp-stream-key" in req["authorization"], (
                f"Stream request 1 should use temp-stream-key, got: {req['authorization']}"
            )
        elif content == "stream2":
            assert "default-api-key" in req["authorization"], (
                f"Stream request 2 should use default-api-key (reset failed!), got: {req['authorization']}"
            )


def test_concurrent_requests_no_leakage(mock_openai_server):
    """Test that concurrent requests with different headers don't leak API keys."""
    import concurrent.futures

    captured_requests.clear()

    def make_request(request_num: int, api_key: str):
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": f"{request_num}"}],
                "guardrails": {"config_id": "with_api_key_header"},
                "stream": False,
            },
            headers={"X-Gpt-3.5-Turbo-Authorization": api_key},
        )
        return response

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(make_request, 1, "key-request-1"),
            executor.submit(make_request, 2, "key-request-2"),
            executor.submit(make_request, 3, "key-request-3"),
        ]
        results = [f.result() for f in futures]

    # All requests should succeed
    assert all(r.status_code == 200 for r in results)

    # Verify each request used the correct API key
    for req in captured_requests:
        request_idx = req["body"]["messages"][-1]["content"].strip()
        assert req["authorization"] == f"Bearer key-request-{request_idx}"


def test_case_insensitive_header_name(mock_openai_server):
    """Test that header names are case-insensitive."""
    test_cases = {
        "case1": "X-Gpt-3.5-Turbo-Authorization",
        "case2": "x-gpt-3.5-turbo-authorization",
        "case3": "X-GPT-3.5-TURBO-AUTHORIZATION",
    }

    captured_requests.clear()

    for request_id, header_name in test_cases.items():
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": request_id}],
                "guardrails": {"config_id": "with_api_key_header"},
                "stream": False,
            },
            headers={header_name: f"key-for-{header_name}"},
        )
        assert response.status_code == 200, f"Failed for header: {header_name}"

    # Verify each request used the correct key
    for req in captured_requests:
        content = req["body"]["messages"][-1]["content"]
        if content in test_cases:
            expected_header = test_cases[content]
            assert f"key-for-{expected_header}" in req["authorization"], (
                f"Request {content} should use key-for-{expected_header}, got: {req['authorization']}"
            )


def test_api_key_not_leaked_in_response(mock_openai_server):
    """Test that API keys are not leaked in the response."""
    captured_requests.clear()

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Check for leaks"}],
            "guardrails": {"config_id": "with_api_key_header"},
            "stream": False,
        },
        headers={"X-Gpt-3.5-Turbo-Authorization": "secret-key-should-not-leak"},
    )

    assert response.status_code == 200
    response_text = response.text.lower()

    # Verify the secret key is not in the response
    assert "secret-key-should-not-leak" not in response_text


def test_empty_header_value(mock_openai_server):
    """Test behavior when header is present but empty."""
    captured_requests.clear()

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Empty header test"}],
            "guardrails": {"config_id": "with_api_key_header"},
            "stream": False,
        },
        headers={"X-Gpt-3.5-Turbo-Authorization": ""},
    )

    # Should still work (uses empty string as the key)
    assert response.status_code == 200


def test_different_model_different_header(mock_openai_server):
    """Test that different models require different header keys."""
    # Create a config with a different model name
    config_with_different_model = """models:
  - type: main
    engine: openai
    model: gpt-4
    parameters:
      api_key: default-gpt4-key
"""
    # Write temporary config
    config_path = os.path.join(api.app.rails_config_path, "with_gpt4")
    os.makedirs(config_path, exist_ok=True)
    with open(os.path.join(config_path, "config.yml"), "w") as f:
        f.write(config_with_different_model)

    try:
        captured_requests.clear()

        # Test with gpt-4 header
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Test GPT-4"}],
                "guardrails": {"config_id": "with_gpt4"},
                "stream": False,
            },
            headers={"X-Gpt-4-Authorization": "gpt4-specific-key"},
        )

        assert response.status_code == 200

        # Verify the GPT-4 specific key was used
        auth_headers = [req["authorization"] for req in captured_requests if req["authorization"]]
        assert len(auth_headers) > 0
        assert any("gpt4-specific-key" in auth for auth in auth_headers)

        # Clean up
        api.llm_rails_instances.clear()
    finally:
        # Clean up the temporary config
        import shutil

        if os.path.exists(config_path):
            shutil.rmtree(config_path)


def test_multi_model_config_both_headers(mock_openai_server):
    """Test that multiple models in a config each receive their own API keys from headers."""
    captured_requests.clear()

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Hello multi-model"}],
            "guardrails": {"config_id": "with_multi_model_api_keys"},
            "stream": False,
        },
        headers={
            "X-Gpt-3.5-Turbo-Authorization": "custom-main-key-789",
            "X-Gpt-4-Authorization": "custom-self-check-key-012",
        },
    )

    assert response.status_code == 200

    # Verify that requests were made
    assert len(captured_requests) > 0, "No requests captured"

    # We expect at least 2 requests: one for self-check (gpt-4) and one for main (gpt-3.5-turbo)
    # Group requests by model
    requests_by_model = {}
    for req in captured_requests:
        model = req["body"].get("model", "unknown")
        if model not in requests_by_model:
            requests_by_model[model] = []
        requests_by_model[model].append(req)

    # Verify gpt-4 (self-check) used the custom self-check key
    if "gpt-4" in requests_by_model:
        gpt4_auths = [req["authorization"] for req in requests_by_model["gpt-4"]]
        assert any("custom-self-check-key-012" in auth for auth in gpt4_auths), (
            f"GPT-4 self-check model should use custom-self-check-key-012, got: {gpt4_auths}"
        )

    # Verify gpt-3.5-turbo (main) used the custom main key
    if "gpt-3.5-turbo" in requests_by_model:
        gpt35_auths = [req["authorization"] for req in requests_by_model["gpt-3.5-turbo"]]
        assert any("custom-main-key-789" in auth for auth in gpt35_auths), (
            f"GPT-3.5-turbo main model should use custom-main-key-789, got: {gpt35_auths}"
        )

    # At minimum, the main model should have been called
    assert "gpt-3.5-turbo" in requests_by_model, "Main model (gpt-3.5-turbo) was not called"


def test_multi_model_config_partial_headers(mock_openai_server):
    """Test multi-model config with only one header provided (partial override)."""
    captured_requests.clear()

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Partial override test"}],
            "guardrails": {"config_id": "with_multi_model_api_keys"},
            "stream": False,
        },
        headers={
            # Only provide main model header, self-check should use default
            "X-Gpt-3.5-Turbo-Authorization": "custom-main-only-key",
        },
    )

    assert response.status_code == 200

    # Debug: Print all captured requests
    requests_by_model = {}
    for req in captured_requests:
        model_name = req["body"].get("model", "unknown")
        if model_name not in requests_by_model:
            requests_by_model[model_name] = []
        requests_by_model[model_name].append(req)

    # Verify requests were made
    assert len(captured_requests) > 0

    # Main model should use custom key
    if "gpt-3.5-turbo" in requests_by_model:
        gpt35_auths = [req["authorization"] for req in requests_by_model["gpt-3.5-turbo"]]
        assert any("custom-main-only-key" in auth for auth in gpt35_auths), (
            f"Main model should use custom key, got: {gpt35_auths}"
        )

    # Self-check model should use default key if it was called
    if "gpt-4" in requests_by_model:
        gpt4_auths = [req["authorization"] for req in requests_by_model["gpt-4"]]
        assert any("default-self-check-key" in auth for auth in gpt4_auths), (
            f"Self-check model should use default key, got: {gpt4_auths}"
        )


def test_multi_model_config_no_headers(mock_openai_server):
    """Test multi-model config with no headers uses all default keys."""
    captured_requests.clear()

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Default keys test"}],
            "guardrails": {"config_id": "with_multi_model_api_keys"},
            "stream": False,
        },
    )

    assert response.status_code == 200

    # Verify requests were made
    assert len(captured_requests) > 0

    auth_headers = [req["authorization"] for req in captured_requests if req["authorization"]]
    assert len(auth_headers) > 0

    # Should use default keys
    default_main_used = any("default-main-key" in auth for auth in auth_headers)
    assert default_main_used, f"Default main key not found in {auth_headers}"
