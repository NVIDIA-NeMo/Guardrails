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
import os

import pytest

pytest.importorskip("openai", reason="openai is required for server tests")
from fastapi.testclient import TestClient

from nemoguardrails.server import api
from nemoguardrails.server.datastore.memory_store import MemoryStore

client = TestClient(api.app)


@pytest.fixture(scope="function", autouse=True)
def setup_test_env():
    original_engine = os.environ.get("MAIN_MODEL_ENGINE")
    original_base_url = os.environ.get("MAIN_MODEL_BASE_URL")
    original_datastore = api.datastore
    os.environ["MAIN_MODEL_ENGINE"] = "custom_llm"
    os.environ["MAIN_MODEL_BASE_URL"] = "http://localhost:8000"
    api.datastore = None
    api.llm_rails_instances.clear()
    yield
    api.llm_rails_instances.clear()
    api.datastore = original_datastore
    if original_engine is not None:
        os.environ["MAIN_MODEL_ENGINE"] = original_engine
    else:
        os.environ.pop("MAIN_MODEL_ENGINE", None)
    if original_base_url is not None:
        os.environ["MAIN_MODEL_BASE_URL"] = original_base_url
    else:
        os.environ.pop("MAIN_MODEL_BASE_URL", None)


def _chat_payload(config_id, state, stream=False):
    return {
        "model": "gpt-4o",
        "stream": stream,
        "messages": [
            {
                "content": "hi",
                "role": "user",
            }
        ],
        "guardrails": {
            "config_id": config_id,
            "state": state,
        },
    }


def _test_call(config_id):
    response = client.post(
        "/v1/chat/completions",
        json=_chat_payload(config_id, {}),
    )
    assert response.status_code == 200
    res = response.json()
    print(res)
    assert len(res["choices"][0]["message"]) == 2
    assert res["choices"][0]["message"]["content"] == "Hello!"
    assert res["guardrails"]["state"]

    response = client.post(
        "/v1/chat/completions",
        json=_chat_payload(config_id, res["guardrails"]["state"]),
    )
    assert response.status_code == 200
    res = response.json()
    assert res["choices"][0]["message"]["content"] == "Hello again!"


def test_1():
    api.app.rails_config_path = os.path.join(os.path.dirname(__file__), "..", "test_configs", "simple_server")
    _test_call("config_1")


def test_2_x_empty_state_runs_without_returning_state():
    api.app.rails_config_path = os.path.join(os.path.dirname(__file__), "..", "test_configs", "simple_server_2_x")
    response = client.post(
        "/v1/chat/completions",
        json=_chat_payload("config_2", {}),
    )

    assert response.status_code == 200
    res = response.json()
    assert res["choices"][0]["message"]["content"] == "Hello!"
    assert "state" not in res["guardrails"]


def test_2_x_raw_state_rejected_at_server():
    """Colang 2.0 stateful continuation over HTTP is no longer supported.

    The serialized 2.0 State carried trusted control-plane fields, so the server
    rejects any 2.0-shaped state with a 422 instead of forwarding it to the core.
    """
    api.app.rails_config_path = os.path.join(os.path.dirname(__file__), "..", "test_configs", "simple_server_2_x")
    response = client.post(
        "/v1/chat/completions",
        json=_chat_payload("config_2", {"version": "2.x", "state": "{}"}),
    )
    assert response.status_code == 422
    assert "Colang 2.0" in response.json()["detail"]


def test_2_x_raw_state_rejected_at_server_streaming():
    api.app.rails_config_path = os.path.join(os.path.dirname(__file__), "..", "test_configs", "simple_server_2_x")
    response = client.post(
        "/v1/chat/completions",
        json=_chat_payload("config_2", {"version": "2.x", "state": "{}"}, stream=True),
    )

    assert response.status_code == 422
    assert "Colang 2.0" in response.json()["detail"]


def test_2_x_events_state_rejected_after_config_load():
    api.app.rails_config_path = os.path.join(os.path.dirname(__file__), "..", "test_configs", "simple_server_2_x")
    response = client.post(
        "/v1/chat/completions",
        json=_chat_payload("config_2", {"events": []}),
    )

    assert response.status_code == 422
    assert "Colang 2.0" in response.json()["detail"]


def test_2_x_thread_id_rejected_after_config_load():
    api.app.rails_config_path = os.path.join(os.path.dirname(__file__), "..", "test_configs", "simple_server_2_x")
    api.datastore = MemoryStore()

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4o",
            "messages": [{"content": "hi", "role": "user"}],
            "guardrails": {
                "config_id": "config_2",
                "thread_id": "state-bug-thread-0001",
            },
        },
    )

    assert response.status_code == 422
    assert "thread_id" in response.json()["detail"]
    assert "Colang 2.0" in response.json()["detail"]


def test_invalid_state_shape_rejected_before_model_init():
    api.app.rails_config_path = os.path.join(os.path.dirname(__file__), "..", "test_configs", "simple_server_2_x")
    os.environ.pop("MAIN_MODEL_BASE_URL", None)

    response = client.post(
        "/v1/chat/completions",
        json=_chat_payload("config_2", {"unexpected": "value"}),
    )

    assert response.status_code == 422
    assert "events" in response.json()["detail"]


def test_invalid_events_state_type_rejected_before_model_init():
    api.app.rails_config_path = os.path.join(os.path.dirname(__file__), "..", "test_configs", "simple_server_2_x")
    os.environ.pop("MAIN_MODEL_BASE_URL", None)

    response = client.post(
        "/v1/chat/completions",
        json=_chat_payload("config_2", {"events": "not-a-list"}),
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "Invalid state format: 'events' must be a list."
