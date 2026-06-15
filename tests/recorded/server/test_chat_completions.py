from __future__ import annotations

import pytest

from tests.recorded.server.assertions import normalize_chat_completion
from tests.recorded.snapshots import snapshot

pytestmark = [pytest.mark.recorded, pytest.mark.vcr]


def test_openai_chat_completion_public_contract(server_client, openai_api_key):
    response = server_client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-5.4-nano",
            "messages": [{"role": "user", "content": "Say a short safe greeting."}],
            "guardrails": {"config_id": "openai_baseline"},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["choices"][0]["message"]["content"].strip()
    assert normalize_chat_completion(body) == snapshot(
        {
            "id": "[RECORDED_RESPONSE_ID]",
            "choices": [
                {
                    "finish_reason": "stop",
                    "index": 0,
                    "message": {"content": "Hello! 😊 How can I help you today?", "role": "assistant"},
                }
            ],
            "created": 0,
            "model": "gpt-5.4-nano",
            "object": "chat.completion",
            "guardrails": {"config_id": "openai_baseline"},
        }
    )


def test_chat_completion_blocked_by_input_rail(server_client):
    response = server_client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-5.4-nano",
            "messages": [{"role": "user", "content": "block input"}],
            "guardrails": {"config_id": "blocking_input"},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["choices"][0]["message"]["content"] == "I'm sorry, I can't respond to that."
    assert normalize_chat_completion(body) == snapshot(
        {
            "id": "[RECORDED_RESPONSE_ID]",
            "choices": [
                {
                    "finish_reason": "stop",
                    "index": 0,
                    "message": {"content": "I'm sorry, I can't respond to that.", "role": "assistant"},
                }
            ],
            "created": 0,
            "model": "gpt-5.4-nano",
            "object": "chat.completion",
            "guardrails": {"config_id": "blocking_input"},
        }
    )


def test_chat_completion_invalid_config_id_returns_error(server_client):
    response = server_client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-5.4-nano",
            "messages": [{"role": "user", "content": "hi"}],
            "guardrails": {"config_id": "does_not_exist"},
        },
    )

    assert response.status_code == 200
    content = response.json()["choices"][0]["message"]["content"].lower()
    assert "could not load" in content or "error" in content
