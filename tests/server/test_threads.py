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
from nemoguardrails.server.api import register_datastore
from nemoguardrails.server.datastore.memory_store import MemoryStore

register_datastore(MemoryStore())
client = TestClient(api.app)


@pytest.fixture(scope="function", autouse=True)
def setup_test_env():
    original_path = api.app.rails_config_path
    original_engine = os.environ.get("MAIN_MODEL_ENGINE")
    api.app.rails_config_path = os.path.join(os.path.dirname(__file__), "..", "test_configs", "simple_server")
    os.environ["MAIN_MODEL_ENGINE"] = "custom_llm"
    api.llm_rails_instances.clear()
    yield
    api.app.rails_config_path = original_path
    api.llm_rails_instances.clear()
    if original_engine is not None:
        os.environ["MAIN_MODEL_ENGINE"] = original_engine
    else:
        os.environ.pop("MAIN_MODEL_ENGINE", None)


def test_get():
    response = client.get("/v1/rails/configs")
    assert response.status_code == 200

    # Check that we have at least one config
    result = response.json()
    assert len(result) > 0


def test_1():
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4o",
            "messages": [
                {
                    "content": "hi",
                    "role": "user",
                }
            ],
            "guardrails": {
                "config_id": "config_1",
                "thread_id": "as9d8f7s9d8f7a9s8df79asdf879",
            },
        },
    )
    assert response.status_code == 200
    res = response.json()
    assert "choices" in res
    assert "message" in res["choices"][0]
    assert res["choices"][0]["message"]["content"] == "Hello!"

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4o",
            "messages": [
                {
                    "content": "hi",
                    "role": "user",
                }
            ],
            "guardrails": {
                "config_id": "config_1",
                "thread_id": "as9d8f7s9d8f7a9s8df79asdf879",
            },
        },
    )
    res = response.json()
    assert res["choices"][0]["message"]["content"] == "Hello again!"


@pytest.mark.parametrize(
    "thread_id, status_code",
    [
        (None, 200),  # thread_id is None
        ("a" * 16, 200),  # thread_id is a valid string
        ("abcd", 422),  # thread_id is too short
        ("a" * 256, 422),  # thread_id is too long
        (123, 422),  # thread_id is not a string
    ],
)
def test_thread_id(thread_id, status_code):
    guardrails_data = {"config_id": "config_1"}
    if thread_id is not None:
        guardrails_data["thread_id"] = thread_id
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4o",
            "messages": [{"content": "hi", "role": "user"}],
            "guardrails": guardrails_data,
        },
    )
    assert response.status_code == status_code


@pytest.mark.skip(reason="Should only be run locally when Redis is available.")
def test_with_redis():
    from nemoguardrails.server.datastore.redis_store import RedisStore

    register_datastore(RedisStore("redis://localhost/1"))
    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [
                {
                    "content": "hi",
                    "role": "user",
                }
            ],
            "guardrails": {
                "config_id": "config_1",
                "thread_id": "as9d8f7s9d8f7a9s8df79asdf879",
            },
        },
    )
    assert response.status_code == 200
    res = response.json()
    assert len(res["messages"]) == 1
    assert res["messages"][0]["content"] == "Hello!"

    register_datastore(RedisStore("redis://localhost/1"))

    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [
                {
                    "content": "hi",
                    "role": "user",
                }
            ],
            "guardrails": {
                "config_id": "config_1",
                "thread_id": "as9d8f7s9d8f7a9s8df79asdf879",
            },
        },
    )
    res = response.json()
    assert res["choices"][0]["message"]["content"] == "Hello again!"
